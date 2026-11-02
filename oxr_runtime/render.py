"""PDF 页面渲染放到几个子进程里做，替换 OXR 流程里的 ``pdf_to_images``。

OXR 用一把进程级的锁（``oxr/utils/pdf.py`` 的 ``_PDFIUM_LOCK``：PDFium 不能多线程同时用）把整个进程的渲染串成一条
队，锁里还要把页面编码成 PNG 再解码回来。实测单进程最多约 11 页/秒，并发一高渲染就排队：按页 64 并发时，一页
光渲染的中位数就要 2.7 秒。

这里开几个渲染进程，各自一份 PDFium，进程之间不用互斥：子进程只把页面渲染成 PNG，主进程在调用方的线程里解码
（OpenCV 解码时不占 GIL）；整份 PDF 按页拆成几段，交给几个进程同时渲染。渲染和解码的调用和 OXR 相同（sopdf
``render(dpi=150)`` 出 PNG、``cv2.imdecode`` 成 BGR），图逐位相同。实测 85 页：OXR 自己渲染 7.6 秒，4 个进程
1.3–1.5 秒，8 个进程 0.84 秒。

子进程用 spawn 起：主进程里有 CUDA 和好几个线程，fork 出来的子进程可能卡在别的线程拿着的锁上。这个模块只导入渲染
用到的库，子进程不会带上 torch。

渲染进程意外退出（比如 PDFium 在坏文件上崩溃）时，整个进程池都不能用了：换一个新的池，这份文档重试一次，还是退出就
抛 :class:`RenderError`，由 OXR 当成无效文档返回；同时在渲染的其他文档也会碰到一次池子坏掉，各自在新的池里重试。
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import cv2
import numpy as np
import sopdf

log = logging.getLogger(__name__)

DPI = 150
"""和 OXR ``pdf_to_images`` 的默认分辨率相同。"""

Worker = Callable[[bytes, int, int, "int | None"], "tuple[int, list[bytes]]"]


class RenderError(RuntimeError):
    """渲染进程在这份文档上意外退出，换了新的进程池重试一次还是退出。"""


class PageRenderer:
    """``processes`` 个渲染进程。:meth:`render` 可以从多个线程同时调用。``worker`` 是子进程里执行的渲染函数，
    测试时换成会崩溃的版本。"""

    def __init__(self, processes: int, *, dpi: int = DPI, worker: Worker | None = None):
        if processes < 1:
            raise ValueError(f"渲染进程数至少是 1，现在是 {processes}")
        self.processes = processes
        self._dpi = dpi
        self._worker = worker or _render_pages
        self._lock = threading.Lock()
        self._pool = self._start_pool()
        self._decoders = ThreadPoolExecutor(processes, thread_name_prefix="oxr-render-decode")

    def render(self, pdf_bytes: bytes) -> list[np.ndarray]:
        """渲染整份 PDF，返回每页一张 BGR 图（同 OXR 的 ``pdf_to_images``）。文档打不开时子进程里的异常原样抛出。"""
        for attempt in (1, 2):
            pool = self._pool
            try:
                pngs = self._render_png(pool, pdf_bytes)
                break
            except BrokenProcessPool as exc:
                self._replace(pool)
                if attempt == 2:
                    raise RenderError("渲染进程在这份文档上意外退出") from exc
                log.warning("渲染进程意外退出，换新的进程池重试这份文档")
        if len(pngs) == 1:
            return [_decode(pngs[0])]
        return list(self._decoders.map(_decode, pngs))

    def close(self) -> None:
        """停掉渲染进程和解码线程：等正在渲染的做完，排着的取消。"""
        with self._lock:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._decoders.shutdown(wait=True, cancel_futures=True)

    def _render_png(self, pool: ProcessPoolExecutor, pdf_bytes: bytes) -> list[bytes]:
        # 第一页和总页数一起取回：按页发的场景（一份一页）只要一来一回；页数多时，其余的页再拆给几个进程。
        count, pngs = pool.submit(self._worker, pdf_bytes, self._dpi, 0, 1).result()
        if count > 1:
            parts: list[Future[tuple[int, list[bytes]]]] = [
                pool.submit(self._worker, pdf_bytes, self._dpi, start, stop)
                for start, stop in _split(1, count, self.processes)
            ]
            for part in parts:
                pngs += part.result()[1]
        if len(pngs) != count:
            raise RuntimeError(f"渲染出 {len(pngs)} 页，文档有 {count} 页")
        return pngs

    def _start_pool(self) -> ProcessPoolExecutor:
        pool = ProcessPoolExecutor(self.processes, mp_context=multiprocessing.get_context("spawn"))
        # spawn 起一个进程要导入 sopdf、numpy、cv2（约半秒），先把进程都起好，第一批请求不用等。
        for ready in [pool.submit(_ready) for _ in range(self.processes)]:
            ready.result()
        return pool

    def _replace(self, broken: ProcessPoolExecutor) -> None:
        """换掉坏掉的进程池。几个线程同时发现时只换一次。"""
        with self._lock:
            if self._pool is not broken:
                return
            broken.shutdown(wait=False, cancel_futures=True)
            self._pool = self._start_pool()


def install(processes: int) -> PageRenderer:
    """起 ``processes`` 个渲染进程，替换 OXR 流程里的 ``pdf_to_images``。返回渲染器，服务退出时调它的 ``close``。"""
    from oxr.pipeline import orchestrator

    renderer = PageRenderer(processes)
    orchestrator.pdf_to_images = renderer.render
    return renderer


def _split(start: int, stop: int, parts: int) -> list[tuple[int, int]]:
    """第 ``start`` 到 ``stop - 1`` 页拆成最多 ``parts`` 段首尾相接的区间，各段页数相差不超过 1。"""
    total = stop - start
    if total <= 0:
        return []
    parts = min(parts, total)
    size, extra = divmod(total, parts)
    ranges = []
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _render_pages(pdf_bytes: bytes, dpi: int, start: int, stop: int | None) -> tuple[int, list[bytes]]:
    """子进程里执行：打开文档，把第 ``start`` 到 ``stop - 1`` 页（``stop`` 为 None 时到最后一页）渲染成 PNG。
    返回（文档总页数，各页 PNG）。"""
    with sopdf.open(stream=pdf_bytes) as doc:
        pages = doc.pages
        count = len(pages)
        stop = count if stop is None else min(stop, count)
        return count, [pages[i].render(dpi=dpi) for i in range(start, stop)]


def _decode(png: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def _ready() -> None:
    """进程起好的信号，什么也不做。"""
