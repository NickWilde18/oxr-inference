"""渲染进程池（:mod:`oxr_runtime.render`）：图和 OXR 在本进程里渲染的逐位相同、页序不乱、坏文件和渲染进程崩溃都
干净地报错、关闭后子进程都退出。

对照用 OXR 的渲染写法（``sopdf`` 渲染成 PNG、``cv2.imdecode`` 成 BGR，分辨率 150）；单元测试里导入不了 OXR，照它的
写法在本进程里算一遍。PDF 用 pikepdf 现造：每页尺寸不同、画的方块个数不同，页序错了图就对不上。
"""

from __future__ import annotations

import io
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
sopdf = pytest.importorskip("sopdf")
pikepdf = pytest.importorskip("pikepdf")

from oxr_runtime import render  # noqa: E402

_PAGES = [(612, 792, 1), (300, 500, 2), (842, 595, 3), (400, 400, 4), (612, 792, 5), (200, 900, 6), (500, 300, 7)]


def _pdf(pages: list[tuple[int, int, int]]) -> bytes:
    """每页一个（宽、高、方块个数），方块颜色和位置随序号变。"""
    pdf = pikepdf.new()
    for width, height, rects in pages:
        ops = "".join(f"{0.12 * i:.2f} 0.3 {1 - 0.1 * i:.2f} rg {10 + 25 * i} {10 + 18 * i} 20 14 re f\n" for i in range(rects))
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page, MediaBox=[0, 0, width, height], Contents=pdf.make_stream(ops.encode()),
            Resources=pikepdf.Dictionary(),
        )
        pdf.pages.append(pikepdf.Page(page))
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def _in_process(data: bytes) -> list[np.ndarray]:
    """OXR ``pdf_to_images`` 的写法。"""
    with sopdf.open(stream=data) as doc:
        return [cv2.imdecode(np.frombuffer(page.render(dpi=150), np.uint8), cv2.IMREAD_COLOR) for page in doc.pages]


def _crash_on_marker(pdf_bytes: bytes, dpi: int, start: int, stop: int | None) -> tuple[int, list[bytes]]:
    """渲染进程里用：收到 ``b"crash"`` 时整个进程直接退出（模拟 PDFium 在坏文件上崩溃），否则照常渲染。"""
    if pdf_bytes == b"crash":
        os._exit(13)
    return render._render_pages(pdf_bytes, dpi, start, stop)


def _call(fn, *args, timeout: float = 120):
    """在另一个线程里调用，最多等 ``timeout`` 秒：挂住时用例失败，不会卡住整轮测试。"""
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(fn, *args).result(timeout=timeout)


def test_split_covers_pages_in_order() -> None:
    """场景：把第 start 到 stop-1 页拆给 parts 个进程。
    条件：页数比进程数多、少、相等、为 0 几种。
    期待：各段首尾相接、按页序排、不重不漏，段数不超过进程数，没有空段。"""
    for start, stop, parts in ((1, 60, 4), (1, 3, 4), (0, 4, 4), (1, 1, 4), (5, 6, 3), (0, 7, 1)):
        ranges = render._split(start, stop, parts)
        assert len(ranges) <= parts and all(a < b for a, b in ranges), (start, stop, parts, ranges)
        flat = [page for a, b in ranges for page in range(a, b)]
        assert flat == list(range(start, stop)), (start, stop, parts, ranges)


def test_render_matches_in_process() -> None:
    """场景：7 页（尺寸、内容各不相同）和 1 页的 PDF。
    条件：3 个渲染进程，多页文档按页拆给几个进程。
    期待：和本进程里按 OXR 的写法渲染的图逐位相同、页序相同。"""
    renderer = render.PageRenderer(3)
    try:
        for data in (_pdf(_PAGES), _pdf(_PAGES[2:3])):
            want = _in_process(data)
            got = _call(renderer.render, data)
            assert len(got) == len(want)
            for page, (g, w) in enumerate(zip(got, want)):
                assert g.shape == w.shape and np.array_equal(g, w), page
    finally:
        renderer.close()


def test_bad_document_raises_and_renderer_keeps_working() -> None:
    """场景：交给渲染进程的不是 PDF。
    条件：2 个渲染进程。
    期待：这份文档抛异常（OXR 会当成无效文档返回）；之后的文档照常渲染。"""
    renderer = render.PageRenderer(2)
    try:
        with pytest.raises(Exception):
            _call(renderer.render, b"not a pdf")
        assert len(_call(renderer.render, _pdf(_PAGES[:2]))) == 2
    finally:
        renderer.close()


def test_worker_crash_fails_document_then_recovers() -> None:
    """场景：渲染进程在这份文档上整个退出（模拟 PDFium 崩溃）。
    条件：2 个渲染进程；崩了换新的进程池重试一次，还是崩。
    期待：这份文档抛 RenderError，不挂住；之后的文档在新的进程池里照常渲染，图和本进程渲染的逐位相同。"""
    renderer = render.PageRenderer(2, worker=_crash_on_marker)
    try:
        with pytest.raises(render.RenderError):
            _call(renderer.render, b"crash")
        data = _pdf(_PAGES[:3])
        got = _call(renderer.render, data)
        assert all(np.array_equal(g, w) for g, w in zip(got, _in_process(data))) and len(got) == 3
    finally:
        renderer.close()


def test_close_stops_worker_processes() -> None:
    """场景：渲染过一份文档以后关掉。
    条件：3 个渲染进程。
    期待：close 在限时内返回，渲染进程全部退出。"""
    before = {p.pid for p in multiprocessing.active_children()}
    renderer = render.PageRenderer(3)
    _call(renderer.render, _pdf(_PAGES))
    started = {p.pid for p in multiprocessing.active_children()} - before
    assert started, "渲染进程没有起来"
    _call(renderer.close, timeout=60)
    assert not started & {p.pid for p in multiprocessing.active_children()}
