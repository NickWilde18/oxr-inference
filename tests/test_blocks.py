"""按块的两个接口（``/v1/layout/sync``、``/v1/recognize/sync``）：走的是 OXR 自己的版面、阅读顺序和逐块识别。

模型换成假的：版面检测回固定的几块，识别模型按提示词回固定的字，阅读顺序回一串编号。要 OXR 源码（环境变量
``OXR_SRC``）、FastAPI 和 OpenCV，没有就跳过。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

OXR_SRC = os.environ.get("OXR_SRC")
if not OXR_SRC:
    pytest.skip("要 OXR 源码：设环境变量 OXR_SRC", allow_module_level=True)
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

_ENV = {"OXR__LAYOUT_ANALYZE_MODEL__URL": "http://in-process.invalid",
        "OXR__OXR_MODEL__URL": "http://in-process.invalid/v1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    os.environ.update(_ENV, OXR_RUNTIME_TASK_DIR=str(tmp_path))
    if OXR_SRC not in sys.path:
        sys.path.insert(0, OXR_SRC)
    from fastapi.testclient import TestClient
    from oxr.model import client as oxr_client
    from oxr.model import prompts
    from oxr.pipeline import layout as oxr_layout
    from oxr.pipeline.models import Block, BlockType, ContentFormat

    from oxr_runtime import worker

    calls: list[tuple[str, tuple[int, ...]]] = []

    async def fake_layouts(images, model_options=None, **kwargs):
        def block(i, kind, box):
            return Block(idx=i, type=kind, bbox=box, content="", format=ContentFormat.TEXT, layout_score=0.9)

        return [[
            block(0, BlockType.TEXT, [10.0, 60.0, 90.0, 80.0]),
            block(1, BlockType.PAGE_HEADER, [10.0, 0.0, 90.0, 8.0]),
            block(2, BlockType.TITLE, [10.0, 20.0, 90.0, 40.0]),
            block(3, BlockType.FORMULA, [10.2, 82.6, 49.5, 95.4]),
        ] for _ in images]

    async def fake_vision(self, prompt, image, timeout, model_options=None, max_tokens=None):
        calls.append((prompt, tuple(image.shape)))
        if prompt == prompts.PROMPT_FORMULA_RECOGNITION:
            text = "$$E=mc^2 \\tag{1}$$"
        elif prompt == prompts.PROMPT_TABLE_RECOGNITION:
            text = "```html\n<table><tr><td>1</td></tr></table>\n```"
        elif "reading order" in prompt:
            # 去掉页眉后剩 3 块（文字、标题、公式），模型说标题在最前、公式在文字前面——和按纵坐标排的结果不同，
            # 用来确认顺序确实取自模型，不是 XY-Cut++ 兜底
            text = "1 2 0"
        else:
            text = "hello"
        return oxr_client.ModelOutput(text, finish_reason="stop")

    monkeypatch.setattr(oxr_layout, "analyze_layouts", fake_layouts)
    monkeypatch.setattr(oxr_client.OXRModelClient, "_call_vision_model", fake_vision)
    with TestClient(worker._oxr_app()) as test_client:
        yield test_client, calls


def _page(h: int = 100, w: int = 100) -> tuple[str, bytes, str]:
    ok, png = cv2.imencode(".png", np.full((h, w, 3), 255, dtype=np.uint8))
    assert ok
    return ("page.png", png.tobytes(), "image/png")


def test_layout_returns_blocks_in_reading_order_without_margins(client) -> None:
    """场景：一页四块，其中一块是页眉；阅读顺序模型说第二块（标题）在最前。
    期待：页眉去掉；剩下三块按模型给的顺序排，idx 是顺序里的位置；没有识别任何块的内容（只调了一次阅读顺序）。"""
    test_client, calls = client
    reply = test_client.post("/v1/layout/sync", data={"api_key": "test"}, files={"file": _page()}).json()

    assert reply["code"] == 0, reply
    page = reply["data"]["result"]["pages"][0]
    assert page["page_size"] == {"w": 100, "h": 100}
    assert [(b["idx"], b["type"]) for b in page["blocks"]] == [(0, "Title"), (1, "Formula"), (2, "Text")]
    assert page["blocks"][0]["bbox"] == [10.0, 20.0, 90.0, 40.0]
    assert len(calls) == 1 and calls[0][1] == (100, 100, 3)  # 只有阅读顺序那一次，送的是整页


def test_layout_keeps_margins_when_asked(client) -> None:
    """场景：``feature_config.keep_header_footer=true``。期待：页眉留着，和 /v1/parse/sync 同义。"""
    test_client, calls = client
    calls.clear()
    reply = test_client.post("/v1/layout/sync", data={"api_key": "test", "feature_config": '{"keep_header_footer": true}'},
                             files={"file": _page()}).json()
    # 4 块时假模型回的 "1 2 0" 缺一个编号，OXR 的修补会按 XY-Cut++ 插回去；这里只看页眉在不在。
    assert reply["code"] == 0, reply
    assert "Page-header" in {b["type"] for b in reply["data"]["result"]["pages"][0]["blocks"]}


def test_recognize_uses_per_type_prompt_and_crop_and_keeps_request_order(client) -> None:
    """场景：一页上要识别一个表格、一个公式、一段文字，请求里就是这个顺序。
    期待：每块用 OXR 对应类型的提示词，裁图按 OXR 的规则（坐标四舍五入）；结果按请求顺序回；表格去掉代码围栏、
    公式编号按 /v1/parse/sync 的 JSON 那样并进内容。"""
    from oxr.model import prompts

    test_client, calls = client
    calls.clear()
    blocks = [
        {"type": "Table", "bbox": [0, 0, 50, 40]},
        {"type": "Formula", "bbox": [10.4, 50.6, 60.5, 70.2]},
        {"type": "Text", "bbox": [0, 80, 100, 100]},
    ]
    reply = test_client.post("/v1/recognize/sync", data={"api_key": "test", "blocks": json.dumps(blocks)},
                             files={"file": _page()}).json()

    assert reply["code"] == 0, reply
    out = reply["data"]["result"]["blocks"]
    assert [(b["idx"], b["type"], b["format"]) for b in out] == [(0, "Table", "html"), (1, "Formula", "latex"),
                                                                    (2, "Text", "text")]
    assert out[0]["content"] == "<table><tr><td>1</td></tr></table>"
    assert out[1]["content"].endswith("\\tag{1}")
    assert out[2]["content"] == "hello"
    by_prompt = {prompt: shape for prompt, shape in calls}
    assert by_prompt[prompts.PROMPT_TABLE_RECOGNITION] == (40, 50, 3)
    assert by_prompt[prompts.PROMPT_FORMULA_RECOGNITION] == (19, 50, 3)  # Python 的 round 是四舍六入五成双：y 51..70、x 10..60
    assert by_prompt[prompts.PROMPT_TEXT_RECOGNITION] == (20, 100, 3)


@pytest.mark.parametrize(("blocks", "message"), [
    ([{"type": "Picture", "bbox": [0, 0, 10, 10]}], "Picture"),
    ([{"type": "Nope", "bbox": [0, 0, 10, 10]}], "type is invalid"),
    ([{"type": "Text", "bbox": [10, 0, 5, 10]}], "x2 > x1"),
    ([{"type": "Text", "bbox": [200, 200, 300, 300]}], "裁出来是空的"),
    ([], "non-empty"),
])
def test_recognize_rejects_bad_blocks(client, blocks, message) -> None:
    """场景：图片块、不认识的类型、反着的框、整个落在页外的框、空数组。期待：参数错误 1133，一个块都不识别。"""
    test_client, calls = client
    calls.clear()
    reply = test_client.post("/v1/recognize/sync", data={"api_key": "test", "blocks": json.dumps(blocks)},
                             files={"file": _page()}).json()
    assert reply["code"] == 1133 and message in reply["message"], reply
    assert calls == []


def test_parse_sync_is_still_mounted(client) -> None:
    """挂上新接口之后，OXR 原来的 /v1/parse/sync 还在、路径没变。"""
    test_client, _ = client
    paths = set(test_client.app.openapi()["paths"])
    assert {"/v1/parse/sync", "/v1/parse/async", "/v1/layout/sync", "/v1/recognize/sync"} <= paths
