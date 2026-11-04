"""按块的两个接口：只做版面（``/v1/layout/sync``）、只识别指定的块（``/v1/recognize/sync``）。

``/v1/parse/sync`` 把一页从头做到尾：版面检测 → 去掉页眉页脚 → 阅读顺序 → 每块识别 → 跨块清理 → 拼结果。调用方
只想要其中一部分时（比如有文字层的 PDF 页：文字类块的字从文字层取，只有公式、表格、代码要识别），整页识别里大部分
算力是白花的。这两个接口把流程拆成两段，各段直接调 OXR 自己的函数，结果和 ``/v1/parse/sync`` 走到同一步时的结果
相同：

- ``/v1/layout/sync``：OXR 的 ``analyze_layouts``（版面检测加它的后处理：置信度阈值、NMS）、去页眉页脚（和
  ``feature_config.keep_header_footer`` 同义），再按 OXR 的 ``resolve_reading_orders`` 排阅读顺序。**阅读顺序是识别
  模型对整页算的**（OXR 的版面检测不给顺序），所以每页仍要调一次识别模型；这一次输出很短（每块几个 token）。
  输出不对时的修补、XY-Cut++ 兜底都是 OXR 原样的；
- ``/v1/recognize/sync``：对调用方给的每一块调 OXR 的 ``recognize_block``，裁图、提示词、输出到长度上限时的恢复
  （表格缩小重试、文字调高重复惩罚重试）、各类型的后处理都和 ``/v1/parse/sync`` 同一份代码；公式编号按
  ``/v1/parse/sync`` 的 JSON 那样并进公式内容（``\\tag{…}``）。

**和 /v1/parse/sync 不同的地方**（写在接口说明里，调用方要知道）：

- 版面结果是识别之前的块。``/v1/parse/sync`` 识别完之后还会按内容删掉重复的块（同一段字被两个框各认一遍、大框
  套小框且内容相同等，见 OXR 的 ``overlap_dedupe``、``exact_text_dedupe``、``cross_category_dedupe``）；这些要看识别出
  来的字，版面这一段做不了。所以版面结果可能比 ``/v1/parse/sync`` 多出几块；
- 识别接口不做跨块清理，每块各认各的，按请求的顺序回；
- 图片块（``Picture``）不识别，调用方按框自己裁图。请求里带图片块直接报参数错误，不悄悄跳过。

两个接口的请求格式、错误码和 ``/v1/parse/sync`` 一致（multipart 表单、``api_key`` 必填但不校验、参数错误 1133、处理
失败 1137）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

log = logging.getLogger("oxr_runtime.blocks")

MAX_BLOCKS_PER_REQUEST = 512
"""识别接口一次最多收多少块。一页上 D-FINE 实测最多几十块，留足余量；没有上界的话一个请求就能把识别队列占满。"""

router = APIRouter()


@router.post("/layout/sync")
async def layout_sync(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    feature_config: Optional[str] = Form(None),
) -> JSONResponse:
    """只做版面：每页的块（类型、框、置信度）按阅读顺序排好，``idx`` 就是阅读顺序里的位置。文件和
    ``/v1/parse/sync`` 一样可以是图片或 PDF（PDF 按 OXR 自己的渲染逐页转图，框是那张图上的像素坐标）。"""
    from oxr.config.settings import settings
    from oxr.model.client import model_client
    from oxr.pipeline.errors import get_public_error_message
    from oxr.pipeline.layout import analyze_layouts
    from oxr.pipeline.models import BlockType
    from oxr.pipeline.orchestrator import _decode_document_images
    from oxr.pipeline.reading_order import resolve_reading_orders
    from oxr.server import routes

    keep_header_footer, warnings, errors = routes._parse_feature_config(feature_config)
    if errors:
        return routes._api_error(routes._format_issues("参数错误", errors))

    task_id = str(uuid.uuid4())
    data = await file.read()
    started = time.perf_counter()
    try:
        images = await asyncio.to_thread(_decode_document_images, data, file.filename or "")
        decoded = time.perf_counter()
        pages = await analyze_layouts(images)
        laid_out = time.perf_counter()
        if not keep_header_footer:
            margins = {BlockType.PAGE_HEADER, BlockType.PAGE_FOOTER}
            pages = [[block for block in blocks if block.type not in margins] for blocks in pages]
        if not settings.layout_analyze_model.provides_reading_order:
            orders = await resolve_reading_orders(pages, images, model_client.get_request_semaphore())
            pages = [[blocks[i] for i in order] for blocks, order in zip(pages, orders)]
        ordered = time.perf_counter()
    except Exception as exc:
        log.warning("版面请求失败 task_id=%s error_type=%s", task_id, type(exc).__name__)
        return routes._api_error(get_public_error_message(exc), code=1137)

    result = {
        "file_name": file.filename,
        "pages": [
            {
                "page_num": page_num,
                "page_size": {"w": int(image.shape[1]), "h": int(image.shape[0])},
                "blocks": [
                    {"idx": idx, "type": block.type.value, "bbox": [float(v) for v in block.bbox],
                     "score": block.layout_score}
                    for idx, block in enumerate(blocks)
                ],
            }
            for page_num, (image, blocks) in enumerate(zip(images, pages))
        ],
    }
    metadata = {
        "page_num": len(images),
        "processing_time_ms": _ms(started, ordered),
        "decode_time_ms": _ms(started, decoded),
        "layout_time_ms": _ms(decoded, laid_out),
        "reading_order_time_ms": _ms(laid_out, ordered),
    }
    return _success(task_id, warnings, result, metadata)


@router.post("/recognize/sync")
async def recognize_sync(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    blocks: str = Form(...),
    extra: Optional[str] = Form(None),
    element_formats: Optional[str] = Form(None),
) -> JSONResponse:
    """只识别指定的块。``file`` 是一页（图片，或只有一页的 PDF）；``blocks`` 是 JSON 数组，每项
    ``{"type": "Table", "bbox": [x1, y1, x2, y2]}``，框是这一页图上的像素坐标（和 ``/v1/layout/sync`` 回的一样）。
    ``extra``、``element_formats`` 和 ``/v1/parse/sync`` 同义。结果按请求的顺序回，``idx`` 是请求里的下标。"""
    from oxr.model.client import model_client
    from oxr.pipeline.concurrency import gather_cancel_on_error
    from oxr.pipeline.errors import get_public_error_message
    from oxr.pipeline.orchestrator import _decode_document_images
    from oxr.pipeline.recognition import recognize_block
    from oxr.server import routes

    model_options, extra_errors = routes._parse_extra(extra)
    formats, format_errors = routes._parse_element_formats(element_formats)
    requested, block_errors = parse_blocks(blocks)
    errors = extra_errors + format_errors + block_errors
    if errors:
        return routes._api_error(routes._format_issues("参数错误", errors))

    task_id = str(uuid.uuid4())
    data = await file.read()
    started = time.perf_counter()
    try:
        images = await asyncio.to_thread(_decode_document_images, data, file.filename or "")
    except Exception as exc:
        return routes._api_error(get_public_error_message(exc), code=1137)
    if len(images) != 1:
        return routes._api_error(routes._format_issues("参数错误", [f"file 要是一页，收到 {len(images)} 页"]))
    image = images[0]
    empty = [i for i, block in enumerate(requested) if _crop_is_empty(block.bbox, image.shape)]
    if empty:
        return routes._api_error(routes._format_issues(
            "参数错误", [f"blocks[{i}].bbox 在这一页上裁出来是空的" for i in empty]))

    semaphore = model_client.get_request_semaphore()
    try:
        recognized = await gather_cancel_on_error(*(
            recognize_block(block, image, semaphore, file.filename or "", 0, model_options, formats)
            for block in requested
        ))
    except Exception as exc:
        log.warning("识别请求失败 task_id=%s error_type=%s", task_id, type(exc).__name__)
        return routes._api_error(get_public_error_message(exc), code=1137)

    result = {"file_name": file.filename, "blocks": [serialize(block, idx) for idx, block in enumerate(recognized)]}
    metadata = {"block_num": len(recognized), "processing_time_ms": _ms(started, time.perf_counter())}
    return _success(task_id, [], result, metadata)


def parse_blocks(raw: str) -> tuple[list[Any], list[str]]:
    """``blocks`` 表单字段 → OXR 的 ``Block`` 列表（内容空着，``idx`` 是请求里的下标）。返回（块，错误）。"""
    from oxr.pipeline.models import Block, BlockType, ContentFormat

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return [], ["blocks must be a JSON array."]
    if not isinstance(items, list) or not items:
        return [], ["blocks must be a non-empty JSON array."]
    if len(items) > MAX_BLOCKS_PER_REQUEST:
        return [], [f"blocks has {len(items)} items; at most {MAX_BLOCKS_PER_REQUEST} per request."]

    known = {kind.value: kind for kind in BlockType}
    parsed: list[Any] = []
    errors: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"blocks[{i}] must be an object.")
            continue
        kind = known.get(item.get("type"))
        if kind is None:
            errors.append(f"blocks[{i}].type is invalid; supported values are {', '.join(sorted(known))}.")
            continue
        if kind == BlockType.PICTURE:
            errors.append(f"blocks[{i}].type=Picture is not recognized; crop picture blocks on the caller side.")
            continue
        bbox = item.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            errors.append(f"blocks[{i}].bbox must be [x1, y1, x2, y2] with finite numbers, x2 > x1 and y2 > y1.")
            continue
        parsed.append(Block(idx=i, type=kind, bbox=[float(v) for v in bbox], content="", format=ContentFormat.TEXT))
    return parsed, errors


def serialize(block: Any, idx: int) -> dict:
    """识别完的一块 → ``/v1/parse/sync`` 的 JSON 里那一块的样子（公式编号并进内容，含 ``\\ce`` 的公式类型报成
    ``Chemical Equation``），``idx`` 换成请求里的下标。"""
    from oxr.pipeline.postprocess.formula_postprocess import expand_formula_numbering_blocks
    from oxr.pipeline.postprocess.json_postprocess import serialize_final_blocks

    regions, notes = serialize_final_blocks(expand_formula_numbering_blocks([block]))
    # 编号和公式是同一个框、下标相邻，一定并得进去；并不进去说明 OXR 的规则变了，宁可报错也不悄悄丢字。
    if len(regions) != 1 or notes:
        raise RuntimeError(f"公式编号没能并进公式：regions={len(regions)} notes={len(notes)}")
    region = regions[0]
    region["idx"] = idx
    return region


def _crop_is_empty(bbox: list[float], shape: tuple[int, ...]) -> bool:
    # 和 OXR 的 crop_image 同样取整、夹到图内。
    h, w = shape[:2]
    x1, y1, x2, y2 = (round(v) for v in bbox)
    return min(w, x2) <= max(0, x1) or min(h, y2) <= max(0, y1)


def _success(task_id: str, warnings: list[str], result: dict, metadata: dict) -> JSONResponse:
    return JSONResponse({
        "code": 0, "message": "任务成功", "warnings": warnings,
        "data": {"task_id": task_id, "error": None, "result": result, "metadata": metadata},
    })


def _ms(start: float, end: float) -> int:
    return int((end - start) * 1000)
