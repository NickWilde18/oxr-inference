"""GPU 进程：独占这张卡，里面跑识别引擎（:class:`~oxr_runtime.engine.Engine`）和版面检测
（:class:`~oxr_runtime.layout.LayoutModel`），通过本机 Unix 套接字（:mod:`oxr_runtime.ipc`）给流程进程服务。

    python -m oxr_runtime.gpu_server --model /models/OXR-1.0 --oxr-src /opt/oxr --socket /tmp/x/gpu.sock

由 :mod:`oxr_runtime.serve` 拉起，认证密钥从环境变量 ``OXR_RUNTIME_IPC_KEY``（十六进制）读。

启动顺序：识别引擎（KV 缓存、CUDA graph、按最坏情况预热）→ 版面检测（各种批大小预热）→ 两边再各占一次峰值（版面检测
录 CUDA Graph 时 PyTorch 清过一次缓存）→ 核对显存余量 → 开始收连接。
显存在开始服务之前一次占好，服务时不再向 GPU 要大块显存；余量不到 ``--memory-margin-gb`` 就退出，日志里写差多少、
怎么调。套接字在核对通过之后才创建，流程进程连得上就说明 GPU 进程已经能干活了。

请求（``kind``）：

- ``recognize``：一张缩放好的图（uint8 ``[高, 宽, 3]``）和提示词、``max_tokens``、``repetition_penalty``；回复文字、
  结束原因、token 数。排队的请求满了回 ``Busy``；
- ``layout``：一批预处理好的整页图（float32 ``[页, 3, 边长, 边长]``）和每页原始尺寸；回复每页的类别、框、置信度。
  一个专用线程按到达顺序一批一批算，排着的超过 ``--layout-queue`` 批时回 ``Busy``；
- ``info``：版面检测的类别数、输入边长、最大批大小，视觉编码器的 patch 边长和合并步长（流程进程缩放图要用）。

识别请求还没出结果时可以取消（交给引擎）。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import queue
import sys
import threading
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # 见 serve.py
# 显存分配器按需把物理显存接到已有的段后面，不为每种形状另开新段：形状各异的批跑下来，占着的显存不会因为碎片越涨越多
# （实测不开这个时，各种压法跑一轮下来 GPU 进程多占约 7 GB，段数 279 → 360）。要在导入 torch 之前设。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

log = logging.getLogger("oxr_runtime.gpu_server")


class _Service:
    """把 IPC 请求转给引擎和版面检测。``handle`` 在连接的收线程里调用，``reply`` 可能在引擎线程里调用。"""

    def __init__(self, engine, layout, layout_queue: int):
        from oxr_runtime.engine import Busy

        self._engine = engine
        self._layout = layout
        self._busy = Busy
        self._lock = threading.Lock()
        self._tickets: dict = {}
        self._jobs: queue.Queue = queue.Queue(maxsize=layout_queue)
        threading.Thread(target=self._layout_loop, name="oxr-layout", daemon=True).start()

    def handle(self, kind: str, header: dict, arrays: list, reply) -> None:
        if kind == "recognize":
            self._recognize(header, arrays[0], reply)
        elif kind == "layout":
            try:
                self._jobs.put_nowait((arrays[0], arrays[1], reply))
            except queue.Full:
                raise self._busy(f"版面检测排着的已有 {self._jobs.maxsize} 批") from None
        elif kind == "info":
            vision = self._engine.config.vision
            reply({**self._layout.info(), "patch_size": vision.patch_size, "hidden_stride": vision.hidden_stride})
        else:
            raise ValueError(f"不认识的请求：{kind}")

    def cancel(self, request_id: str) -> None:
        with self._lock:
            ticket = self._tickets.get(request_id)
        if ticket is not None:
            ticket.cancel()

    def _recognize(self, header: dict, image, reply) -> None:
        request_id = header["id"]
        finished = threading.Event()

        def done(result) -> None:
            finished.set()
            with self._lock:
                self._tickets.pop(request_id, None)
            if isinstance(result, BaseException):
                reply({"_error": type(result).__name__, "_message": str(result)})
            else:
                reply({"text": result.text, "finish_reason": result.finish_reason,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": len(result.token_ids)})

        ticket = self._engine.submit_resized(image, header["prompt"], max_tokens=header["max_tokens"],
                                             repetition_penalty=header["repetition_penalty"], done=done)
        with self._lock:
            if not finished.is_set():  # 引擎可能已经回过了（比如马上就被取消），这时不用再记
                self._tickets[request_id] = ticket

    def _layout_loop(self) -> None:
        while True:
            images, sizes, reply = self._jobs.get()
            try:
                reply({}, list(self._layout.infer(images, sizes)))
            except Exception as exc:  # 这一批失败，接着算下一批
                log.exception("版面检测出错")
                reply({"_error": type(exc).__name__, "_message": str(exc)})


def _check_memory(free: int, total: int, reserved: int, *, kv_gb: float, margin_gb: float) -> None:
    """预热之后核对显存（字节）：整卡还空着的不到 ``margin_gb`` 就以退出码 3 退出，日志里写差多少、怎么调。"""
    gib = 2**30
    log.info("显存：本进程占 %.1f GiB（其中 KV 缓存 %.1f GiB），整卡 %.1f GiB、还空 %.1f GiB",
             reserved / gib, kv_gb, total / gib, free / gib)
    if free < margin_gb * gib:
        short = math.ceil((margin_gb - free / gib) * 10 - 1e-9) / 10  # 向上取到 0.1：照着调小之后确实够
        if short < kv_gb:
            advice = f"把 --kv-gb 调小至少 {short:.1f}，或者让出这张卡上别的进程"
        else:
            advice = "KV 缓存全让出来也不够：让出这张卡上别的进程，或者调低 --memory-margin-gb"
        log.error("显存余量 %.2f GiB，不到要求的 %.2f GiB：%s", free / gib, margin_gb, advice)
        sys.exit(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--oxr-src", required=True, type=Path)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--kv-gb", type=float, default=16.0)
    parser.add_argument("--max-running", type=int, default=256)
    parser.add_argument("--max-waiting", type=int, default=2048)
    parser.add_argument("--repetition-span", type=int, default=1024)
    parser.add_argument("--layout-queue", type=int, default=64)
    parser.add_argument("--memory-margin-gb", type=float, default=2.0,
                        help="预热之后整卡至少还要空着这么多显存（给 cuBLAS、cuDNN 的工作区和零碎分配留的）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 引擎的 prefill 线程和逐字生成线程都要拿 GIL：切换间隔从默认的 5 ms 调到 1 ms，逐字生成每一步等 GIL 的时间短些
    # （实测整条流程按页 16 并发 7.53–7.61 → 7.65–7.69 页/秒，32 并发 7.96 → 8.22）。
    sys.setswitchinterval(0.001)

    import torch

    from oxr_runtime import ipc
    from oxr_runtime.engine import Engine, EngineSettings
    from oxr_runtime.layout import LayoutModel

    engine = Engine(args.model, EngineSettings(kv_cache_gb=args.kv_gb, max_running=args.max_running,
                                               max_waiting=args.max_waiting, repetition_span=args.repetition_span))
    layout = LayoutModel(args.oxr_src, args.model)
    # 版面检测录 CUDA Graph 时 PyTorch 清过一次缓存，把引擎预热时占好的激活显存还了回去：两个都加载完，再各占一次峰值。
    engine.reserve_peak()
    layout.reserve_peak()

    free, total = torch.cuda.mem_get_info()
    _check_memory(free, total, torch.cuda.memory_reserved(), kv_gb=args.kv_gb, margin_gb=args.memory_margin_gb)

    engine.start()
    service = _Service(engine, layout, args.layout_queue)
    key = os.environ.get("OXR_RUNTIME_IPC_KEY")
    server = ipc.Server(args.socket, service.handle, on_cancel=service.cancel,
                        authkey=bytes.fromhex(key) if key else None)
    log.info("开始服务：%s", args.socket)
    server.serve_forever()


if __name__ == "__main__":
    main()
