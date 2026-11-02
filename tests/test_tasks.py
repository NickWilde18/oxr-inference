"""所有流程进程共用的异步任务存储（:class:`oxr_runtime.tasks.SharedTaskStore`）：查询结果和 OXR 自己的
``MemoryTaskStore`` 一样；做完的任务保留 1 小时；接任务的进程退出了，没做完的任务报失败；任务号不是 UUID 的一律查不到；
过期的文件会被删掉。

和 OXR 对照的那一个用例要 OXR 源码（环境变量 ``OXR_SRC``），没有就跳过。
"""

from __future__ import annotations

import asyncio
import copy
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path

import pytest

from oxr_runtime import tasks

_RESULT = {"file_name": "a.pdf", "result": {"markdown": "第一页"}, "metadata": {"pages": 1}}


class _Clock:
    """从现在开始、可以往前拨的时钟。"""

    def __init__(self):
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


def _run(coroutine):
    return asyncio.run(coroutine)


def _dead_pid() -> int:
    process = multiprocessing.get_context("spawn").Process(target=int)
    process.start()
    process.join(30)
    return process.pid


def test_replies_match_oxr_memory_store(tmp_path: Path) -> None:
    """场景：同一串操作（提交、处理中、成功；另一个提交、失败；查不存在的任务；改不存在的任务）。
    条件：分别交给 OXR 的 MemoryTaskStore 和共享存储。
    期待：每一步查到的内容完全相同。"""
    oxr_src = os.environ.get("OXR_SRC")
    if not oxr_src:
        pytest.skip("要 OXR 源码：设环境变量 OXR_SRC")
    sys.path.insert(0, oxr_src)
    from oxr.server.queue.memory_store import MemoryTaskStore

    async def replay(store) -> list:
        a, b, missing = (str(uuid.UUID(int=i)) for i in (1, 2, 3))
        seen = []

        async def look(task_id: str) -> None:  # MemoryTaskStore 返回的是它自己存着的字典，后面的操作会改到它：存一份拷贝
            seen.append(copy.deepcopy(await store.get(task_id)))

        await store.create(a, "a.pdf")
        await look(a)
        await store.set_processing(a)
        await look(a)
        await store.set_success(a, _RESULT)
        await look(a)
        await store.create(b)
        await store.set_failed(b, "坏文件")
        await look(b)
        await look(missing)
        await store.set_processing(missing)
        await store.set_success(missing, _RESULT)
        await look(missing)
        return seen

    assert _run(replay(tasks.SharedTaskStore(tmp_path))) == _run(replay(MemoryTaskStore()))


def test_finished_task_expires_after_an_hour(tmp_path: Path) -> None:
    """场景：任务成功了。
    条件：时钟拨到做完之后 59 分钟、再到 60 分钟。
    期待：59 分钟时还查得到结果，60 分钟时查不到（和查不存在的任务一样）。"""
    clock = _Clock()
    store = tasks.SharedTaskStore(tmp_path, clock=clock)
    task_id = str(uuid.uuid4())
    _run(store.create(task_id, "a.pdf"))
    _run(store.set_success(task_id, _RESULT))
    clock.now += 59 * 60
    assert _run(store.get(task_id))["result"] == _RESULT["result"]
    clock.now += 60
    assert _run(store.get(task_id)) is None


def test_unfinished_task_is_kept_while_its_worker_lives(tmp_path: Path) -> None:
    """场景：任务还在处理，接它的进程（本进程）还活着。
    条件：时钟拨到 2 小时之后。
    期待：照样查得到，状态还是处理中。"""
    clock = _Clock()
    store = tasks.SharedTaskStore(tmp_path, clock=clock)
    task_id = str(uuid.uuid4())
    _run(store.create(task_id))
    _run(store.set_processing(task_id))
    clock.now += 2 * 3600
    assert _run(store.get(task_id))["status"] == "processing"


def test_task_of_dead_worker_fails_then_expires(tmp_path: Path) -> None:
    """场景：任务在处理中，接它的流程进程退出了。
    条件：别的进程查这个任务；再过 1 小时又查。
    期待：先查到失败、错误说明进程退出了请重新提交；1 小时后查不到。"""
    clock = _Clock()
    store = tasks.SharedTaskStore(tmp_path, clock=clock)
    task_id = str(uuid.uuid4())
    _run(store.create(task_id))
    _run(store.set_processing(task_id))
    record = tasks._read_json(tmp_path / f"{task_id}.json")
    record["_owner"] = _dead_pid()
    tasks._write_json(tmp_path / f"{task_id}.json", record)

    found = _run(store.get(task_id))
    assert (found["status"], found["error"]) == ("failed", tasks.ORPHANED)
    clock.now += 3600
    assert _run(store.get(task_id)) is None


def test_task_ids_that_are_not_uuids_are_not_found(tmp_path: Path) -> None:
    """场景：任务目录外面有一个 JSON 文件。
    条件：拿 ``../`` 开头的任务号去查。
    期待：查不到，不会读目录外的文件。"""
    root = tmp_path / "tasks"
    root.mkdir()
    tasks._write_json(tmp_path / "outside.json", {"task_id": "x", "status": "success"})
    store = tasks.SharedTaskStore(root)
    assert _run(store.get("../outside")) is None


def test_expired_files_are_removed_when_a_task_is_submitted(tmp_path: Path) -> None:
    """场景：一个任务成功了，过了 1 小时多没人来取。
    条件：提交一个新任务。
    期待：旧任务的记录和结果文件都删掉了，新任务的还在。"""
    clock = _Clock()
    store = tasks.SharedTaskStore(tmp_path, clock=clock)
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    _run(store.create(old))
    _run(store.set_success(old, _RESULT))
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{old}.json", f"{old}.result.json"])
    clock.now += 3600 + 1  # 扫目录先按文件修改时间粗筛：压线过期的要等下一遍，这里拨过线
    _run(store.create(new))
    assert [p.name for p in tmp_path.iterdir()] == [f"{new}.json"]
