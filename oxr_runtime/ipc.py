"""流程进程（OXR 的接口和流程）和 GPU 进程（识别引擎、版面检测）之间的通信：本机 Unix 套接字
（``multiprocessing.connection``）。

一条连接上可以同时有很多请求在路上：每个请求带一个请求号，回复按号对上，先回的先到。一条消息是一个 JSON 头加若干个
numpy 数组，数组按原始字节各占一帧，不编码成图片格式；收的时候直接收进按头里的形状和类型分配好的数组，可写、不多拷。

两边都不在调用方的线程里等网络：

- :class:`Client` 有一个收线程，回复到了就把对应的 :class:`~concurrent.futures.Future` 结掉；asyncio 里用
  ``asyncio.wrap_future`` 等它。连接断了（GPU 进程退出或重启）时，在路上的请求立刻以 :class:`ConnectionLost` 失败，
  之后的新请求也是，不会挂住；
- :class:`Server` 每条连接一个收线程、一个发线程。处理函数拿到 ``reply``，在哪个线程里调用都行（引擎线程的完成回调
  里也行）：回复先进发线程的队列，调用方不会被对方收得慢卡住。处理函数抛出的异常原样（类型名和消息）传回调用方，
  成为 :class:`RemoteError`。
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from multiprocessing.connection import Client as _connect
from multiprocessing.connection import Connection, Listener

import numpy as np

log = logging.getLogger(__name__)

Reply = Callable[..., None]
"""``reply(header, arrays=())``：把这个请求的回复发回去，每个请求恰好调一次。"""

Handler = Callable[[str, dict, list[np.ndarray], Reply], None]
"""``handler(kind, header, arrays, reply)``：header 里有请求号 ``id``。可以当场 reply，也可以交给别的线程以后 reply。"""


class RemoteError(RuntimeError):
    """服务端处理请求时出错。``kind`` 是服务端异常的类型名。"""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class ConnectionLost(RuntimeError):
    """和服务端的连接断了（服务端进程退出或重启）。在路上的请求和之后的新请求都以它失败。"""


def send(conn: Connection, header: dict, arrays: Sequence[np.ndarray] = ()) -> None:
    """发一条消息：先发 JSON 头（里面记着每个数组的类型和形状），再按顺序发每个数组的原始字节。调用方负责同一条连接
    上的消息不交错。"""
    arrays = [np.ascontiguousarray(a) for a in arrays]
    meta = [{"dtype": a.dtype.str, "shape": list(a.shape)} for a in arrays]
    conn.send_bytes(json.dumps({**header, "_arrays": meta}, ensure_ascii=False).encode())
    for a in arrays:
        conn.send_bytes(a.reshape(-1).view(np.uint8).data if a.size else b"")


def recv(conn: Connection) -> tuple[dict, list[np.ndarray]]:
    """收一条 :func:`send` 发的消息。数组直接收进新分配的数组里。"""
    header = json.loads(conn.recv_bytes())
    arrays = []
    for meta in header.pop("_arrays"):
        array = np.empty(meta["shape"], dtype=np.dtype(meta["dtype"]))
        if array.size:
            conn.recv_bytes_into(array.reshape(-1).view(np.uint8).data)
        else:
            conn.recv_bytes()
        arrays.append(array)
    return header, arrays


class Client:
    """连到 :class:`Server` 的一条连接，线程安全。服务端还没起来时，``connect_timeout`` 秒内反复重连。"""

    def __init__(self, address: str, *, authkey: bytes | None = None, connect_timeout: float = 60.0):
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self._conn = _connect(address, family="AF_UNIX", authkey=authkey)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending: dict[str, Future] = {}
        self._lost: ConnectionLost | None = None
        self._ids = itertools.count()
        self._prefix = f"{os.getpid()}-"
        self._reader = threading.Thread(target=self._read, name="oxr-ipc-client", daemon=True)
        self._reader.start()

    def call(self, kind: str, header: dict, arrays: Sequence[np.ndarray] = ()) -> Future:
        """发一个请求，返回结果是 ``(header, arrays)`` 的 Future；Future 上的 ``request_id`` 给 :meth:`cancel` 用。"""
        request_id = f"{self._prefix}{next(self._ids)}"
        future: Future = Future()
        future.request_id = request_id  # type: ignore[attr-defined]
        with self._lock:
            if self._lost is not None:
                future.set_exception(self._lost)
                return future
            self._pending[request_id] = future
        try:
            with self._send_lock:
                send(self._conn, {**header, "kind": kind, "id": request_id}, arrays)
        except (OSError, EOFError, ValueError) as exc:
            self._fail_all(ConnectionLost(f"发请求时连接断了：{exc}"))
        return future

    def cancel(self, request_id: str) -> None:
        """请服务端取消这个请求。连接已经断了时什么也不做（在路上的请求已经失败了）。"""
        try:
            with self._send_lock:
                send(self._conn, {"kind": "_cancel", "id": request_id})
        except (OSError, EOFError, ValueError):
            pass

    def close(self) -> None:
        self._fail_all(ConnectionLost("连接已关闭"))
        self._conn.close()

    def _read(self) -> None:
        try:
            while True:
                header, arrays = recv(self._conn)
                with self._lock:
                    future = self._pending.pop(header.pop("id"), None)
                if future is None:
                    continue
                if "_error" in header:
                    future.set_exception(RemoteError(header["_error"], header.get("_message", "")))
                else:
                    future.set_result((header, arrays))
        except (EOFError, OSError, ValueError) as exc:
            self._fail_all(ConnectionLost(f"和服务端的连接断了：{exc!r}"))

    def _fail_all(self, exc: ConnectionLost) -> None:
        with self._lock:
            if self._lost is None:
                self._lost = exc
            pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(self._lost)


class Server:
    """在 ``address`` 上收连接，每个请求交给 ``handler``。``on_cancel(request_id)`` 处理取消（可以不给）。"""

    def __init__(self, address: str, handler: Handler, *, on_cancel: Callable[[str], None] | None = None,
                 authkey: bytes | None = None):
        if os.path.exists(address):
            os.unlink(address)
        self._listener = Listener(address, family="AF_UNIX", authkey=authkey)
        self._handler = handler
        self._on_cancel = on_cancel

    def serve_forever(self) -> None:
        while True:
            try:
                conn = self._listener.accept()
            except (OSError, EOFError) as exc:  # 认证失败、对方半路断开：这条不要，接着收
                log.warning("收连接失败：%r", exc)
                continue
            threading.Thread(target=self._serve, args=(conn,), name="oxr-ipc-conn", daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        outbox: queue.SimpleQueue = queue.SimpleQueue()
        writer = threading.Thread(target=_write, args=(conn, outbox), name="oxr-ipc-writer", daemon=True)
        writer.start()
        try:
            while True:
                header, arrays = recv(conn)
                kind = header.pop("kind")
                request_id = header["id"]
                if kind == "_cancel":
                    if self._on_cancel is not None:
                        self._on_cancel(request_id)
                    continue
                reply = _reply_once(outbox, request_id)
                try:
                    self._handler(kind, header, arrays, reply)
                except Exception as exc:  # 这个请求失败，连接照常
                    reply({"_error": type(exc).__name__, "_message": str(exc)})
        except (EOFError, OSError, ValueError):
            pass
        finally:
            outbox.put(None)
            writer.join(timeout=5)
            conn.close()


def _reply_once(outbox: queue.SimpleQueue, request_id: str) -> Reply:
    """这个请求的 ``reply``：只认第一次调用（处理函数报错和迟到的完成回调撞在一起时，不会回两次）。"""
    used = threading.Event()

    def reply(header: dict, arrays: Sequence[np.ndarray] = ()) -> None:
        if used.is_set():
            return
        used.set()
        outbox.put(({**header, "id": request_id}, list(arrays)))

    return reply


def _write(conn: Connection, outbox: queue.SimpleQueue) -> None:
    while True:
        item = outbox.get()
        if item is None:
            return
        try:
            send(conn, *item)
        except (OSError, EOFError, ValueError):
            return
