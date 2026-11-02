"""停服务：启动器收到 SIGTERM（容器停止、``kill``、systemd 停服务都是这样）之后，GPU 进程要停掉，临时目录（套接字、
异步任务）要删掉。流程进程只有 1 个时 uvicorn 直接在启动器进程里跑，停法和多个时不同，两种都测。

真起一遍服务（GPU 进程加载模型、预热），要 GPU、权重目录（环境变量 ``OXR_MODEL``）和 OXR 源码（``OXR_SRC``），没有就
跳过。
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODEL, OXR_SRC = os.environ.get("OXR_MODEL"), os.environ.get("OXR_SRC")
if not (MODEL and OXR_SRC):
    pytest.skip("要权重目录和 OXR 源码：设环境变量 OXR_MODEL、OXR_SRC", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("要 GPU", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _children(pid: int) -> list[int]:
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == pid:
            found.append(int(stat.parent.name))
    return found


def _gone(pid: int) -> bool:
    """进程不在了（已经退出的僵尸进程也算）。"""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("workers", [1, 2])
def test_sigterm_stops_gpu_process_and_removes_run_dir(tmp_path: Path, workers: int) -> None:
    """场景：服务起来了，流程进程 1 个或 2 个。
    条件：给启动器发 SIGTERM。
    期待：启动器退出，GPU 进程也停了，启动器的临时目录删掉了。"""
    log = tmp_path / "serve.log"
    with open(log, "wb") as out:
        launcher = subprocess.Popen(
            [sys.executable, "-m", "oxr_runtime.serve", "--model", MODEL, "--oxr-src", OXR_SRC, "--host", "127.0.0.1",
             "--port", str(_free_port()), "--kv-gb", "4", "--max-running", "64", "--workers", str(workers)],
            env=dict(os.environ, TMPDIR=str(tmp_path)), stdout=out, stderr=subprocess.STDOUT,
        )
    gpu = None
    try:
        deadline = time.monotonic() + 900
        while log.read_text(errors="replace").count("Application startup complete") < workers:
            assert launcher.poll() is None, log.read_text(errors="replace")[-3000:]
            assert time.monotonic() < deadline, "15 分钟内没起来"
            time.sleep(1)
        gpu = next(p for p in _children(launcher.pid)
                   if b"oxr_runtime.gpu_server" in Path(f"/proc/{p}/cmdline").read_bytes())

        launcher.send_signal(signal.SIGTERM)
        launcher.wait(120)
        deadline = time.monotonic() + 90
        while not _gone(gpu) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert _gone(gpu), "GPU 进程还在跑"
        assert list(tmp_path.glob("oxr-runtime-*")) == []
    finally:
        if launcher.poll() is None:
            launcher.kill()
        if gpu is not None and not _gone(gpu):
            os.kill(gpu, signal.SIGKILL)
