"""流程进程和 GPU 进程之间的通信（:mod:`oxr_runtime.ipc`）：消息原样往返、很多请求同时在路上时各回各的、服务端报错
原样传回、服务端进程没了时在路上的请求立刻失败（不挂住）、取消能送到服务端。

服务端在子进程里跑（spawn），和线上一样是两个进程。处理函数写在本文件里：子进程按模块名导入它。
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from oxr_runtime import ipc  # noqa: E402


def _handler(kind: str, header: dict, arrays: list, reply) -> None:
    """echo：数组原样加 1 返回；slow：等 header 里的秒数再回；fail：报错；die：整个进程退出；
    cancel_probe：等取消，收到取消就回 cancelled。"""
    if kind == "echo":
        reply({"tag": header["tag"]}, [a + 1 for a in arrays])
    elif kind == "slow":
        threading.Timer(header["seconds"], lambda: reply({"tag": header["tag"]}, [])).start()
    elif kind == "fail":
        raise ValueError("坏请求：" + header["why"])
    elif kind == "die":
        os._exit(3)
    elif kind == "cancel_probe":
        _waiting[header["id"]] = reply


_waiting: dict = {}


def _on_cancel(request_id: str) -> None:
    reply = _waiting.pop(request_id, None)
    if reply is not None:
        reply({"cancelled": True}, [])


def _serve(address: str) -> None:
    server = ipc.Server(address, _handler, on_cancel=_on_cancel)
    server.serve_forever()


@pytest.fixture
def server(tmp_path: Path):
    address = str(tmp_path / "gpu.sock")
    process = multiprocessing.get_context("spawn").Process(target=_serve, args=(address,), daemon=True)
    process.start()
    client = ipc.Client(address, connect_timeout=30)
    yield client, process
    client.close()
    process.kill()
    process.join(10)


def test_round_trip_header_and_arrays(server) -> None:
    """场景：一个请求带两个数组（uint8 图、float32 小表）。
    条件：服务端把每个数组加 1 返回。
    期待：回复的数组形状、类型、内容都对，header 原样带回。"""
    client, _ = server
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    table = np.linspace(0, 1, 7, dtype=np.float32)
    header, arrays = client.call("echo", {"tag": "x"}, [image, table]).result(timeout=30)
    assert header["tag"] == "x"
    assert arrays[0].dtype == np.uint8 and arrays[0].shape == (4, 5, 3) and np.array_equal(arrays[0], image + 1)
    assert arrays[1].dtype == np.float32 and np.array_equal(arrays[1], table + 1)


def test_many_requests_in_flight_get_their_own_replies(server) -> None:
    """场景：一条连接上同时发 200 个请求，其中一些要等一会儿才回。
    条件：回复的先后顺序和发出的顺序不同。
    期待：每个请求拿到的是自己的回复。"""
    client, _ = server
    futures = []
    for i in range(200):
        if i % 10 == 0:
            futures.append((i, client.call("slow", {"tag": i, "seconds": 0.2}, [])))
        else:
            futures.append((i, client.call("echo", {"tag": i}, [np.full(3, i, dtype=np.int64)])))
    for i, future in futures:
        header, arrays = future.result(timeout=60)
        assert header["tag"] == i
        if i % 10:
            assert arrays[0].tolist() == [i + 1] * 3


def test_handler_error_is_raised_in_caller(server) -> None:
    """场景：服务端处理请求时抛异常。
    条件：异常类型 ValueError。
    期待：调用方拿到 RemoteError，里面带着原来的类型名和消息；连接照常能用。"""
    client, _ = server
    with pytest.raises(ipc.RemoteError) as caught:
        client.call("fail", {"why": "缺字段"}, []).result(timeout=30)
    assert caught.value.kind == "ValueError" and "缺字段" in str(caught.value)
    assert client.call("echo", {"tag": 1}, []).result(timeout=30)[0]["tag"] == 1


def test_server_death_fails_pending_requests_quickly(server) -> None:
    """场景：有请求还在路上（要等 30 秒才回），服务端进程整个退出。
    条件：另一个请求让服务端 os._exit。
    期待：在路上的请求几秒内就失败（ConnectionLost），不会等到 30 秒；之后的新请求也立刻失败。"""
    client, process = server
    pending = client.call("slow", {"tag": 1, "seconds": 30}, [])
    client.call("die", {}, [])
    started = time.monotonic()
    with pytest.raises(ipc.ConnectionLost):
        pending.result(timeout=20)
    assert time.monotonic() - started < 10
    with pytest.raises(ipc.ConnectionLost):
        client.call("echo", {"tag": 2}, []).result(timeout=10)


def test_cancel_reaches_server(server) -> None:
    """场景：请求在服务端挂着等，调用方取消。
    条件：服务端收到取消时回 cancelled。
    期待：调用方的这个请求拿到 cancelled 的回复。"""
    client, _ = server
    future = client.call("cancel_probe", {}, [])
    time.sleep(0.5)
    client.cancel(future.request_id)
    header, _ = future.result(timeout=30)
    assert header["cancelled"] is True
