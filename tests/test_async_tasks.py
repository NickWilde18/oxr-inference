"""多个流程进程时，OXR 的异步接口（``/v1/parse/async`` 提交、``/v1/parse/async_check`` 查询）在哪个进程都要查得到
同一个任务。

和线上一样，每个流程进程各自导入 OXR 的服务（``oxr_runtime.worker._oxr_app``）、各自跑启动流程。提交在本进程，查询在
另一个子进程（spawn）。要 OXR 源码（环境变量 ``OXR_SRC`` 指过去）和 FastAPI，没有就跳过。
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

OXR_SRC = os.environ.get("OXR_SRC")
if not OXR_SRC:
    pytest.skip("要 OXR 源码：设环境变量 OXR_SRC", allow_module_level=True)
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

# OXR 启动时只校验两个模型地址有没有设；提交的文件不是能解析的 PDF，任务在调模型之前就失败。
_ENV = {"OXR__LAYOUT_ANALYZE_MODEL__URL": "http://in-process.invalid",
        "OXR__OXR_MODEL__URL": "http://in-process.invalid/v1"}


def _app(task_dir: str):
    os.environ.update(_ENV, OXR_RUNTIME_TASK_DIR=task_dir)
    if OXR_SRC not in sys.path:
        sys.path.insert(0, OXR_SRC)
    from oxr_runtime import worker

    return worker._oxr_app()


def _query(task_dir: str, task_id: str, out: str) -> None:
    """另一个流程进程：查这个任务，回复写到 ``out``。"""
    from fastapi.testclient import TestClient

    with TestClient(_app(task_dir)) as client:
        reply = client.post("/v1/parse/async_check", data={"task_id": task_id, "api_key": "test"}).json()
    Path(out).write_text(json.dumps(reply, ensure_ascii=False))


def test_task_submitted_to_one_worker_is_found_by_another(tmp_path: Path) -> None:
    """场景：两个流程进程，任务提交给其中一个。
    条件：查询落到另一个进程。
    期待：查得到这个任务（不是 Task not found）。"""
    from fastapi.testclient import TestClient

    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    out = tmp_path / "reply.json"
    with TestClient(_app(str(task_dir))) as client:
        submitted = client.post("/v1/parse/async", data={"api_key": "test"},
                                files={"file": ("p.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")}).json()
        task_id = submitted["data"]["task_id"]
        other = multiprocessing.get_context("spawn").Process(target=_query, args=(str(task_dir), task_id, str(out)))
        other.start()
        other.join(120)
    assert other.exitcode == 0
    reply = json.loads(out.read_text())
    assert (reply.get("data") or {}).get("task_id") == task_id, reply
