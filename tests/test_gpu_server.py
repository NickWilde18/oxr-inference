"""GPU 进程（:mod:`oxr_runtime.gpu_server`）：排队满了回忙、识别结果怎么回给流程进程、取消送到引擎、显存余量不够时
拒绝启动；以及引擎自己排满时报忙。

引擎和版面检测换成假的，请求经真的 :mod:`oxr_runtime.ipc` 收发（服务端在本进程的线程里），看流程进程那一侧拿到什么。
"""

from __future__ import annotations

import logging
import queue
import threading
import types
from pathlib import Path

import pytest

import test_language_fused  # noqa: F401  导入引擎之前先放好注意力替身（见 test_language_fused）

np = pytest.importorskip("numpy")

from oxr_runtime import gpu_server, ipc  # noqa: E402
from oxr_runtime.engine import Busy, Cancelled, Completion, Engine, Ticket  # noqa: E402

_GIB = 2**30


class _FakeTicket:
    def __init__(self, done):
        self._done = done

    def cancel(self) -> None:
        self._done(Cancelled("调用方取消"))


class _FakeEngine:
    """记下提交的请求，``done`` 留给用例来调；``busy`` 时像排满了那样抛 :class:`Busy`。"""

    config = types.SimpleNamespace(vision=types.SimpleNamespace(patch_size=14, hidden_stride=2))

    def __init__(self, *, busy: bool = False):
        self.busy = busy
        self.submitted: queue.Queue = queue.Queue()

    def submit_resized(self, image, prompt, *, max_tokens, repetition_penalty, done):
        if self.busy:
            raise Busy("排队的请求已有 2048 个")
        self.submitted.put((image, prompt, max_tokens, repetition_penalty, done))
        return _FakeTicket(done)


class _FakeLayout:
    """``infer`` 进来时发信号，等用例放行才返回。"""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def infer(self, images, sizes):
        self.entered.set()
        self.release.wait(30)
        n = images.shape[0]
        return np.zeros((n, 3), dtype=np.int64), np.zeros((n, 3, 4), dtype=np.float32), np.ones((n, 3), np.float32)

    def info(self) -> dict:
        return {"num_classes": 3, "image_size": 8, "max_batch": 8}


def _connect(tmp_path: Path, engine, layout, layout_queue: int = 4) -> tuple[ipc.Client, gpu_server._Service]:
    service = gpu_server._Service(engine, layout, layout_queue)
    address = str(tmp_path / "gpu.sock")
    server = ipc.Server(address, service.handle, on_cancel=service.cancel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return ipc.Client(address, connect_timeout=10), service


def _recognize(client: ipc.Client):
    header = {"prompt": "文字识别", "max_tokens": 64, "repetition_penalty": 1.0}
    return client.call("recognize", header, [np.zeros((28, 28, 3), dtype=np.uint8)])


def _page_batch(pages: int = 1) -> list:
    return [np.zeros((pages, 3, 8, 8), dtype=np.float32), np.full((pages, 2), 8, dtype=np.int64)]


def test_layout_queue_full_replies_busy(tmp_path: Path) -> None:
    """场景：版面检测正在算一批，排队上限 1 批，已经排着 1 批。
    条件：再来一批。
    期待：这一批立刻回忙（流程进程拿到 kind 为 Busy 的 RemoteError），不排进去；前两批放行后照常出结果。"""
    layout = _FakeLayout()
    client, _ = _connect(tmp_path, _FakeEngine(), layout, layout_queue=1)
    running = client.call("layout", {}, _page_batch())
    assert layout.entered.wait(10)
    queued = client.call("layout", {}, _page_batch(2))
    with pytest.raises(ipc.RemoteError) as caught:
        client.call("layout", {}, _page_batch()).result(timeout=10)
    assert caught.value.kind == "Busy"
    layout.release.set()
    assert running.result(timeout=10)[1][0].shape == (1, 3)
    assert queued.result(timeout=10)[1][0].shape == (2, 3)


def test_engine_busy_replies_busy(tmp_path: Path) -> None:
    """场景：引擎里排队的请求满了。
    条件：流程进程发一个识别请求。
    期待：流程进程拿到 kind 为 Busy 的 RemoteError，连接照常能用。"""
    client, _ = _connect(tmp_path, _FakeEngine(busy=True), _FakeLayout())
    with pytest.raises(ipc.RemoteError) as caught:
        _recognize(client).result(timeout=10)
    assert caught.value.kind == "Busy"
    assert client.call("info", {}).result(timeout=10)[0]["patch_size"] == 14


def test_recognize_reply_fields(tmp_path: Path) -> None:
    """场景：识别请求交给引擎，引擎生成完。
    条件：结果有文字、结束原因、提示词 token 数和生成的 token。
    期待：提示词、max_tokens、重复惩罚原样交给引擎；流程进程拿到文字、结束原因、两个 token 数。"""
    engine = _FakeEngine()
    client, _ = _connect(tmp_path, engine, _FakeLayout())
    future = _recognize(client)
    image, prompt, max_tokens, penalty, done = engine.submitted.get(timeout=10)
    assert (image.shape, prompt, max_tokens, penalty) == ((28, 28, 3), "文字识别", 64, 1.0)
    done(Completion(text="第一行", token_ids=[5, 6, 7], finish_reason="stop", prompt_tokens=120, seconds=0.1))
    header, _ = future.result(timeout=10)
    assert {k: header[k] for k in ("text", "finish_reason", "prompt_tokens", "completion_tokens")} == {
        "text": "第一行", "finish_reason": "stop", "prompt_tokens": 120, "completion_tokens": 3}


def test_cancel_reaches_engine(tmp_path: Path) -> None:
    """场景：识别请求在引擎里还没出结果，流程进程那边超时了。
    条件：流程进程发取消。
    期待：引擎里这个请求被取消，流程进程这个请求以 kind 为 Cancelled 的 RemoteError 结束；GPU 进程不再记着它。"""
    engine = _FakeEngine()
    client, service = _connect(tmp_path, engine, _FakeLayout())
    future = _recognize(client)
    engine.submitted.get(timeout=10)
    client.cancel(future.request_id)
    with pytest.raises(ipc.RemoteError) as caught:
        future.result(timeout=10)
    assert caught.value.kind == "Cancelled"
    assert service._tickets == {}


class _QueuedEngine:
    """:meth:`Engine.submit_resized` 用到的几个属性：收件箱、排队列表、上限。"""

    def __init__(self, inbox: int, waiting: int, max_waiting: int):
        self._inbox: queue.Queue = queue.Queue()
        for _ in range(inbox):
            self._inbox.put(object())
        self._waiting = [object()] * waiting
        self.settings = types.SimpleNamespace(max_waiting=max_waiting)

    def _request(self, resized, prompt, max_tokens, repetition_penalty, done):
        return types.SimpleNamespace(cancelled=False)


def test_engine_rejects_when_waiting_is_full() -> None:
    """场景：引擎收件箱里 1 个、已排队 1 个，排队上限 2。
    条件：再提交一个。
    期待：抛 Busy，这个请求不进收件箱，done 不会被调用；上限还有空时照常收下。"""
    calls = []
    full = _QueuedEngine(inbox=1, waiting=1, max_waiting=2)
    with pytest.raises(Busy):
        Engine.submit_resized(full, np.zeros((28, 28, 3), np.uint8), "p", max_tokens=8, done=calls.append)
    assert full._inbox.qsize() == 1 and calls == []

    room = _QueuedEngine(inbox=1, waiting=0, max_waiting=2)
    assert isinstance(Engine.submit_resized(room, np.zeros((28, 28, 3), np.uint8), "p", max_tokens=8,
                                            done=calls.append), Ticket)
    assert room._inbox.qsize() == 2


def test_memory_check_passes_with_margin(caplog) -> None:
    """场景：预热之后整卡还空 3 GiB，要求 2 GiB。
    条件：核对显存。
    期待：不退出，日志里记下本进程占多少、整卡还空多少。"""
    with caplog.at_level(logging.INFO, logger="oxr_runtime.gpu_server"):
        gpu_server._check_memory(3 * _GIB, 140 * _GIB, 100 * _GIB, kv_gb=80, margin_gb=2)
    assert "还空 3.0 GiB" in caplog.text


def test_memory_check_refuses_to_start_and_says_how_much(caplog) -> None:
    """场景：预热之后整卡只空 1.96 GiB，要求 2 GiB（差 0.04 GiB）。
    条件：核对显存。
    期待：以退出码 3 退出；日志里的余量不被四舍五入成和要求一样的数，让 --kv-gb 调小的量向上取到 0.1 GiB（调小这么多
    之后确实够）。"""
    with caplog.at_level(logging.INFO, logger="oxr_runtime.gpu_server"), pytest.raises(SystemExit) as caught:
        gpu_server._check_memory(int(1.96 * _GIB), 140 * _GIB, 100 * _GIB, kv_gb=80, margin_gb=2)
    assert caught.value.code == 3
    assert "显存余量 1.96 GiB，不到要求的 2.00 GiB" in caplog.text
    assert "把 --kv-gb 调小至少 0.1" in caplog.text


def test_memory_check_does_not_suggest_shrinking_kv_below_zero(caplog) -> None:
    """场景：KV 缓存只给了 1 GiB，预热之后整卡只空 0.5 GiB，要求 2 GiB（差 1.5 GiB，比整个 KV 缓存还多）。
    条件：核对显存。
    期待：以退出码 3 退出；不建议把 --kv-gb 调小 1.5（调不了这么多），而是说 KV 缓存全让出来也不够。"""
    with caplog.at_level(logging.INFO, logger="oxr_runtime.gpu_server"), pytest.raises(SystemExit) as caught:
        gpu_server._check_memory(int(0.5 * _GIB), 140 * _GIB, 30 * _GIB, kv_gb=1, margin_gb=2)
    assert caught.value.code == 3
    assert "调小至少" not in caplog.text
    assert "KV 缓存全让出来也不够" in caplog.text
