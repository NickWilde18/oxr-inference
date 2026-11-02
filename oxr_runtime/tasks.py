"""所有流程进程共用的异步任务存储，替换 OXR 的 ``MemoryTaskStore``（``oxr/server/queue/memory_store.py``）：方法和
查到的任务记录都和它相同。

OXR 的存储是进程内的字典，多个流程进程时，``/v1/parse/async`` 提交给一个进程，``/v1/parse/async_check`` 落到别的进程
就查不到。这里每个任务在同一个目录里存两个文件，所有流程进程读写这个目录：

- ``<任务号>.json``：任务记录（状态、文件名、错误、metadata），外加接任务的进程号和做完的时刻；
- ``<任务号>.result.json``：成功时的解析结果。先写结果、再把记录改成成功，读的一方看到成功时结果一定已经在了。

每个文件都先写临时文件再改名，读的一方看到的总是完整的一份。一个任务只由接它的进程写（提交、处理中、成功、失败）；
别的进程只在接任务的进程已经退出时把它改成失败。

清理：

- 做完的任务（成功或失败）保留 :data:`TTL` 秒，到期之后查询回「没有这个任务」（和 OXR 查不存在的任务一样）；
- 没做完的任务一直留着；但接它的进程已经退出（流程进程崩溃后 uvicorn 换了新进程）时，查到或清理到它就改成失败
  （:data:`ORPHANED`），再按做完的任务过期；
- 文件在提交新任务时顺带删，每个进程最多每 :data:`SWEEP_INTERVAL` 秒扫一遍目录：先按文件的修改时间挑出可能过期的
  （做完的任务之后不再改，修改时间就是做完的时刻），只读这些的记录。

任务号只认 OXR 生成的 UUID 格式，别的一律当没有（查询的任务号是调用方给的，不能拿来拼出目录外的路径）。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

TTL = 3600
"""做完的任务保留多少秒（和 OXR 演示页面的任务保留时间相同）。"""

SWEEP_INTERVAL = 60
"""每个进程最多多少秒扫一遍目录、删过期的文件。"""

ORPHANED = "处理这个任务的流程进程退出了，请重新提交"
"""接任务的进程退出、任务没做完时，查到的错误。"""

_TASK_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_UNFINISHED = ("queuing", "processing")


class SharedTaskStore:
    """``root`` 是所有流程进程共用的目录（要已经存在）。``clock`` 只在测试里换。"""

    def __init__(self, root: Path, *, ttl: float = TTL, clock: Callable[[], float] = time.time):
        self._root = Path(root)
        self._ttl = ttl
        self._clock = clock
        self._next_sweep = 0.0

    async def create(self, task_id: str, file_name: str | None = None) -> None:
        if not _TASK_ID.fullmatch(task_id):
            raise ValueError(f"任务号不是 UUID：{task_id!r}")
        record = {"record_id": None, "task_id": task_id, "status": "queuing", "file_name": file_name,
                  "result": {}, "error": None, "metadata": {}, "_owner": os.getpid(), "_finished_at": None}
        await asyncio.to_thread(self._create, task_id, record)

    async def set_processing(self, task_id: str) -> None:
        await asyncio.to_thread(self._update, task_id, status="processing")

    async def set_success(self, task_id: str, result: dict[str, Any]) -> None:
        await asyncio.to_thread(self._succeed, task_id, result)

    async def set_failed(self, task_id: str, error: str) -> None:
        await asyncio.to_thread(self._update, task_id, status="failed", error=error, _finished_at=self._clock())

    async def get(self, task_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get, task_id)

    def _create(self, task_id: str, record: dict) -> None:
        now = self._clock()
        if now >= self._next_sweep:
            self._next_sweep = now + SWEEP_INTERVAL
            self._sweep(now)
        _write_json(self._record(task_id), record)

    def _update(self, task_id: str, **changes: Any) -> None:
        record = self._load(task_id)
        if record is not None:  # 和 OXR 一样：没有这个任务就什么也不做
            record.update(changes)
            _write_json(self._record(task_id), record)

    def _succeed(self, task_id: str, result: dict[str, Any]) -> None:
        record = self._load(task_id)
        if record is None:
            return
        _write_json(self._result(task_id), result.get("result", {}))
        record.update(status="success", file_name=result.get("file_name", record["file_name"]),
                      metadata=result.get("metadata", {}), error=None, _finished_at=self._clock())
        _write_json(self._record(task_id), record)

    def _get(self, task_id: str) -> dict[str, Any] | None:
        record = self._load(task_id)
        if record is None:
            return None
        record = self._settle(task_id, record, self._clock())
        if record is None:
            return None
        result: Any = {}
        if record["status"] == "success":
            result = _read_json(self._result(task_id))
            if result is None:  # 刚被别的进程按过期删掉
                return None
        return {k: v for k, v in record.items() if not k.startswith("_")} | {"result": result}

    def _settle(self, task_id: str, record: dict, now: float) -> dict | None:
        """接任务的进程已经退出、还没做完的改成失败；过期的删掉，返回 None。"""
        if record["status"] in _UNFINISHED and not _alive(record["_owner"]):
            record.update(status="failed", error=ORPHANED, _finished_at=now)
            _write_json(self._record(task_id), record)
        if record["_finished_at"] is not None and record["_finished_at"] + self._ttl <= now:
            self._record(task_id).unlink(missing_ok=True)
            self._result(task_id).unlink(missing_ok=True)
            return None
        return record

    def _sweep(self, now: float) -> None:
        with os.scandir(self._root) as entries:
            for entry in entries:
                task_id = entry.name.removesuffix(".json")
                if not _TASK_ID.fullmatch(task_id):
                    continue
                try:
                    stale = entry.stat().st_mtime + self._ttl <= now
                except FileNotFoundError:
                    continue
                record = self._load(task_id) if stale else None
                if record is not None:
                    self._settle(task_id, record, now)

    def _load(self, task_id: str) -> dict | None:
        return _read_json(self._record(task_id)) if _TASK_ID.fullmatch(task_id) else None

    def _record(self, task_id: str) -> Path:
        return self._root / f"{task_id}.json"

    def _result(self, task_id: str) -> Path:
        return self._root / f"{task_id}.result.json"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # 进程在，只是不归本用户
        return True
    return True


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
