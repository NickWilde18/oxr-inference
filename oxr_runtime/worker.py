"""流程进程：跑 OXR 的接口和流程（``oxr.server.app``），识别和版面检测都交给 GPU 进程。

:mod:`oxr_runtime.serve` 用 uvicorn 起 ``--workers`` 个这样的进程。每个进程导入本模块时，按 serve.py 传下来的环境变量
（``OXR_RUNTIME_*``）连上 GPU 进程，再把 OXR 里调模型的地方换掉：

- 识别（``OXRModelClient._call_vision_model``；文字、表格、公式、代码、化学式、整页、阅读顺序都走这里）：裁好的图在本
  进程里缩放好（PIL，和参考实现逐位一致），连同提示词发给 GPU 进程。OXR 官方客户端是编码成 PNG、base64、经 HTTP 发给 vLLM。

  - GPU 进程的 ``finish_reason`` 里，``repetition``（复读提前停）按 ``length`` 报给流程：OXR 遇到 ``length`` 会调高
    重复惩罚重试这一块，正是复读该走的恢复路径；
  - 调用方的超时到了就取消 GPU 进程里的请求，不让它在后台接着生成；
  - 缩放放在专用的小线程池里（``OXR_RUNTIME_PREPROCESS_THREADS`` 个线程），不占事件循环；
  - 只支持贪心解码：OXR 默认 temperature=0、top_p=1，请求要别的值时记一条警告、照样贪心。

- 版面检测（``LayoutAnalyzeClient``）：OXR 官方客户端先在事件循环线程上把整页图编成 PNG（``encode_images``），再经 HTTP 发给
  版面检测服务。这里 ``encode_images`` 原样交回整页图，``request_batch`` 在线程池里按官方服务的预处理转成输入张量，发给
  GPU 进程，结果按官方服务的格式整理（见 :mod:`oxr_runtime.layout`）。本进程同时发给 GPU 进程的版面请求不超过
  ``OXR_RUNTIME_LAYOUT_CONCURRENCY`` 批（serve.py 按 GPU 进程的版面排队上限分给各流程进程），多出来的在本进程里排着，
  不会撞上 GPU 进程的上限被回忙；
- 异步接口（``/v1/parse/async`` 提交、``/v1/parse/async_check`` 查询）：OXR 把任务存在进程内存里，查询落到别的流程
  进程就查不到。换成所有流程进程共用的目录（:mod:`oxr_runtime.tasks`），做完的任务保留 1 小时；
- PDF 渲染照用 OXR 自己的：它的进程级锁只锁本进程，几个流程进程之间本来就是并行的。``OXR_RUNTIME_RENDER_PROCESSES``
  大于 0 时，本进程另开几个渲染进程（:mod:`oxr_runtime.render`），一份多页文档按页拆开并行渲染。

GPU 进程断开（退出或重启）时，在路上和之后的请求都立刻失败，不会挂住；serve.py 发现 GPU 进程没了会让整个服务退出。
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from oxr_runtime import ipc, layout, preprocess, tasks

log = logging.getLogger("oxr_runtime.worker")


def to_rgb(image: np.ndarray) -> Image.Image:
    """OXR 流程里的图是 OpenCV 的 BGR（或灰度）数组。官方客户端编码成 PNG 交给 vLLM，那边按 RGB 解码。"""
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.shape[2] == 4:
        image = image[..., :3]
    return Image.fromarray(np.ascontiguousarray(image[..., ::-1]))


def install(client: ipc.Client, *, preprocess_threads: int, layout_concurrency: int) -> None:
    """把 OXR 调识别模型和版面检测的地方换成发给 GPU 进程。在导入 OXR 的服务（``oxr.server.app``）之前或之后调用都行。"""
    from oxr.model import client as oxr_client
    from oxr.model import layout_client as oxr_layout

    info, _ = client.call("info", {}).result(timeout=60)
    pool = ThreadPoolExecutor(max_workers=preprocess_threads, thread_name_prefix="oxr-preprocess")
    layout_slots: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
    resize = functools.partial(_resize, patch_size=info["patch_size"], hidden_stride=info["hidden_stride"])

    async def call_vision_model(
        self: Any,
        prompt: str,
        image: np.ndarray,
        timeout: int,
        model_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> oxr_client.ModelOutput:
        if image is None:
            raise ValueError("An image is required for vision model calls.")

        options = oxr_client.resolve_model_options(model_options)
        if options["temperature"] != 0.0 or options["top_p"] != 1.0:
            log.warning("只支持贪心解码，忽略 temperature=%s top_p=%s", options["temperature"], options["top_p"])

        resized = await asyncio.get_running_loop().run_in_executor(pool, resize, image)
        header = {
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "repetition_penalty": options["repetition_penalty"],
        }
        call = client.call("recognize", header, [resized])
        waiting = asyncio.wrap_future(call)
        try:
            result, _ = await asyncio.wait_for(asyncio.shield(waiting), timeout)
        except (TimeoutError, asyncio.CancelledError):
            client.cancel(call.request_id)
            # 取消之后 GPU 进程还会回一个 Cancelled，没人再等它：取走，免得事件循环报「异常没人取」。
            waiting.add_done_callback(lambda f: f.cancelled() or f.exception())
            raise

        finish = "length" if result["finish_reason"] == "repetition" else result["finish_reason"]
        usage = SimpleNamespace(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        )
        return oxr_client.ModelOutput(result["text"], finish_reason=finish, usage=usage)

    def encode_images(self: Any, images: list[np.ndarray]) -> list[np.ndarray]:
        return list(images)

    async def request_batch(self: Any, images: list[np.ndarray]) -> object:
        loop = asyncio.get_running_loop()
        slots = layout_slots.setdefault(loop, asyncio.Semaphore(layout_concurrency))
        batch, sizes = await loop.run_in_executor(pool, layout.prepare, list(images), info["image_size"])
        async with slots:
            _, (labels, boxes, scores) = await asyncio.wait_for(
                asyncio.wrap_future(client.call("layout", {}, [batch, sizes])), self.config.timeout
            )
        return layout.to_payload(labels, boxes, scores, info["num_classes"])

    oxr_client.OXRModelClient._call_vision_model = call_vision_model  # type: ignore[method-assign]
    oxr_layout.LayoutAnalyzeClient.encode_images = encode_images  # type: ignore[method-assign]
    oxr_layout.LayoutAnalyzeClient.request_batch = request_batch  # type: ignore[method-assign]


def _resize(image: np.ndarray, *, patch_size: int, hidden_stride: int) -> np.ndarray:
    return preprocess.resize(to_rgb(image), patch_size=patch_size, hidden_stride=hidden_stride)


def _create_app() -> Any:
    oxr_src = Path(os.environ["OXR_RUNTIME_OXR_SRC"])
    sys.path.insert(0, str(oxr_src))
    layout.add_source(oxr_src)
    key = os.environ.get("OXR_RUNTIME_IPC_KEY")
    client = ipc.Client(os.environ["OXR_RUNTIME_SOCKET"], authkey=bytes.fromhex(key) if key else None,
                        connect_timeout=60)
    install(client, preprocess_threads=int(os.environ.get("OXR_RUNTIME_PREPROCESS_THREADS", "4")),
            layout_concurrency=int(os.environ.get("OXR_RUNTIME_LAYOUT_CONCURRENCY", "16")))
    processes = int(os.environ.get("OXR_RUNTIME_RENDER_PROCESSES", "0"))
    if processes > 0:
        from oxr_runtime import render

        render.install(processes)
    return _oxr_app()


def _oxr_app() -> Any:
    """导入 OXR 的服务（``oxr.server.app``）。OXR 源码要已经在 ``sys.path`` 里。

    异步接口的任务存到 ``OXR_RUNTIME_TASK_DIR``（:class:`~oxr_runtime.tasks.SharedTaskStore`，所有流程进程共用），
    换掉 OXR 存在进程内存里的那一份。OXR 在服务启动时才创建任务存储，这里换的是它要创建的类。"""
    module = importlib.import_module("oxr.server.app")
    module.MemoryTaskStore = functools.partial(tasks.SharedTaskStore, Path(os.environ["OXR_RUNTIME_TASK_DIR"]))
    return module.app


app = _create_app() if "OXR_RUNTIME_SOCKET" in os.environ else None
"""uvicorn 导入的 ASGI 应用（``oxr_runtime.worker:app``）。只在 serve.py 设好环境变量的进程里创建。"""
