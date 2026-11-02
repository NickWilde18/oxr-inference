"""启动服务：OXR 官方的 HTTP 接口（``/v1/parse/sync`` 等），识别和版面检测换成本运行时。

    python -m oxr_runtime.serve --model /models/OXR-1.0 --oxr-src /opt/oxr --port 8000

``--oxr-src`` 是 OXR 官方仓库的检出（固定在一个提交上），这里用它的 Python 包 ``oxr``（接口和流程）和
``services/layout_analyze``（版面检测模型），不用它的 vLLM 插件。

进程：

- 一个 GPU 进程（:mod:`oxr_runtime.gpu_server`）：独占这张卡，识别引擎和版面检测在里面，显存启动时一次占好并核对余量；
- ``--workers`` 个流程进程（:mod:`oxr_runtime.worker`，uvicorn 多进程，共用一个监听端口）：OXR 的接口和流程、PDF 渲染、
  各种缩放，经本机 Unix 套接字找 GPU 进程算。流程进程意外退出时 uvicorn 会重新拉起一个；
- GPU 进程意外退出时，本进程让 uvicorn 停下，整个服务以非 0 退出，交给容器重启。

先起 GPU 进程，等它的套接字出现（它核对完显存才建套接字）再起流程进程。套接字放在只有本用户能进的临时目录里，
连接要过一个随机密钥（经环境变量 ``OXR_RUNTIME_IPC_KEY`` 交给子进程）。异步接口的任务也存在这个临时目录里
（:mod:`oxr_runtime.tasks`），所有流程进程共用，服务停下时一起删掉。

OXR 接口那一侧对模型的并发上限（``OXR__OXR_MODEL__MAX_CONCURRENCY``，官方默认 16，是给经 HTTP 调 vLLM 设的）按
GPU 进程的排队上限分给各个流程进程：排队交给引擎，所有流程进程加起来不会超过排队上限。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger("oxr_runtime.serve")

# torch 的编译 worker 池（inductor）默认起几十个子进程；进程因异常退出时它们会把父进程卡住，占着的显存也不放。
# ``OXR_COMPILE=1`` 时要编的只是语言模型和视觉编码器的几个小函数，在本进程里编就够了，关掉这个池。要在导入 torch 之前设。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


class _Stopped(Exception):
    """收到了 SIGTERM。"""


def _raise_stopped(signum: int, frame: object) -> None:
    # 流程进程只有 1 个时，uvicorn 在本进程里跑：收到 SIGTERM 先优雅停下，再换回原来的处理、把 SIGTERM 原样发一遍。
    # 原来的处理是默认的直接结束进程，下面停 GPU 进程、删临时目录的清理就跑不到了，GPU 进程会一直占着显存。
    raise _Stopped


def _wait_for_socket(path: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if process.poll() is not None:
            raise RuntimeError(f"GPU 进程启动时退出了，退出码 {process.returncode}（原因见上面的日志）")
        if time.monotonic() > deadline:
            raise RuntimeError(f"GPU 进程 {timeout:.0f} 秒内没有就绪")
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="权重目录（含 model.safetensors、tokenizer.json）")
    parser.add_argument("--oxr-src", required=True, type=Path, help="OXR 官方仓库的检出")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--kv-gb", type=float, default=16.0)
    parser.add_argument("--max-running", type=int, default=256, help="引擎里同时在跑的序列上限")
    parser.add_argument("--max-waiting", type=int, default=2048, help="引擎里排队的请求上限，满了回忙")
    parser.add_argument("--repetition-span", type=int, default=1024, help="复读提前停的窗口，0 关掉")
    parser.add_argument("--memory-margin-gb", type=float, default=2.0, help="GPU 进程预热之后整卡至少还要空着的显存")
    parser.add_argument("--layout-queue", type=int, default=64, help="GPU 进程里版面检测排队的批数上限，按流程进程数分摊")
    parser.add_argument("--limit-concurrency", type=int, default=256,
                        help="每个流程进程同时处理的 HTTP 请求上限，超出直接回 503（过载时不无限堆积）")
    parser.add_argument("--workers", type=int, default=4, help="流程进程数（OXR 的接口和流程、PDF 渲染、缩放）")
    parser.add_argument("--preprocess-threads", type=int, default=4, help="每个流程进程里做缩放、版面预处理的线程数")
    parser.add_argument("--render-processes", type=int, default=0,
                        help="每个流程进程另开的渲染进程数（oxr_runtime.render，多页文档按页并行渲染）；0 = 用 OXR 自己的")
    parser.add_argument("--startup-timeout", type=float, default=1800, help="等 GPU 进程就绪的秒数")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _raise_stopped)

    run_dir = tempfile.mkdtemp(prefix="oxr-runtime-")
    socket = os.path.join(run_dir, "gpu.sock")
    key = os.urandom(32).hex()
    gpu = subprocess.Popen(
        [sys.executable, "-m", "oxr_runtime.gpu_server", "--model", str(args.model), "--oxr-src", str(args.oxr_src),
         "--socket", socket, "--kv-gb", str(args.kv_gb), "--max-running", str(args.max_running),
         "--max-waiting", str(args.max_waiting), "--repetition-span", str(args.repetition_span),
         "--memory-margin-gb", str(args.memory_margin_gb), "--layout-queue", str(args.layout_queue)],
        env=dict(os.environ, OXR_RUNTIME_IPC_KEY=key),
    )
    stopping = threading.Event()
    try:
        try:
            _wait_for_socket(socket, gpu, args.startup_timeout)
        except RuntimeError as exc:
            log.error("%s", exc)
            sys.exit(1)

        def watch() -> None:
            gpu.wait()
            if not stopping.is_set():
                log.error("GPU 进程意外退出（退出码 %s），停掉整个服务", gpu.returncode)
                os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=watch, name="oxr-gpu-watch", daemon=True).start()

        # OXR 的配置读环境变量（前缀 OXR__），流程进程导入 oxr 之前要设好。两个模型地址只为过启动校验，不会被访问。
        task_dir = os.path.join(run_dir, "tasks")
        os.mkdir(task_dir, mode=0o700)
        os.environ.update(
            OXR_RUNTIME_SOCKET=socket, OXR_RUNTIME_IPC_KEY=key, OXR_RUNTIME_OXR_SRC=str(args.oxr_src),
            OXR_RUNTIME_TASK_DIR=task_dir,
            OXR_RUNTIME_PREPROCESS_THREADS=str(args.preprocess_threads),
            OXR_RUNTIME_RENDER_PROCESSES=str(args.render_processes),
            OXR_RUNTIME_LAYOUT_CONCURRENCY=str(max(1, args.layout_queue // args.workers)),
        )
        os.environ.setdefault("OXR__LAYOUT_ANALYZE_MODEL__URL", "http://in-process.invalid")
        os.environ.setdefault("OXR__OXR_MODEL__URL", "http://in-process.invalid/v1")
        os.environ.setdefault("OXR__OXR_MODEL__MAX_CONCURRENCY", str(max(1, -(-args.max_waiting // args.workers))))

        import uvicorn

        uvicorn.run("oxr_runtime.worker:app", host=args.host, port=args.port, workers=args.workers,
                    limit_concurrency=args.limit_concurrency, log_level="info")
    except _Stopped:
        log.info("收到 SIGTERM，停服务")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # 清理到一半再来的 SIGTERM 不打断清理
        stopping.set()
        if gpu.poll() is None:
            gpu.terminate()
            try:
                gpu.wait(timeout=60)
            except subprocess.TimeoutExpired:
                gpu.kill()
        shutil.rmtree(run_dir, ignore_errors=True)
    if gpu.returncode not in (0, -signal.SIGTERM):
        sys.exit(1)


if __name__ == "__main__":
    main()
