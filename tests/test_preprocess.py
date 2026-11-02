"""预处理和官方 vLLM 插件（``oxr_processor.py``）一致。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from oxr_runtime.preprocess import MAX_PIXELS, MIN_PIXELS, _NORMALIZE, smart_resize, to_patches


def _plugin_patches(normalized_chw: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """插件切 patch 的写法逐行照抄（temporal_patch_size=1、hidden_stride=2、patch=16）。"""
    patches = normalized_chw[np.newaxis]
    channel = patches.shape[1]
    patches = patches.reshape(1, 1, channel, grid_h // 2, 2, 16, grid_w // 2, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(grid_h * grid_w, channel * 16 * 16)


def test_patch_layout_matches_plugin() -> None:
    """切 patch 的顺序和插件逐位相同。

    场景：一张 300×170 的随机彩色图，缩放后按插件的写法（通道在前、9 维 transpose）切一遍作为参照。
    期待：``to_patches`` 的输出逐位相同，网格尺寸相同。
    """
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, size=(170, 300, 3), dtype=np.uint8))

    out = to_patches(image)

    h, w = smart_resize(170, 300, 32, MIN_PIXELS, MAX_PIXELS)
    resized = np.asarray(image.resize((w, h), resample=Image.Resampling.BILINEAR, reducing_gap=None))
    chw = ((resized.astype(np.float64) * 0.00392156862745098).astype(np.float32) - np.float32(0.5)) / np.float32(0.5)
    expected = _plugin_patches(chw.transpose(2, 0, 1), h // 16, w // 16)

    assert (out.grid_h, out.grid_w) == (h // 16, w // 16)
    assert out.pixels.dtype == np.float32
    assert np.array_equal(out.pixels, expected)
    assert out.num_tokens == out.grid_h * out.grid_w // 4


def test_normalize_table_matches_rescale_then_normalize() -> None:
    """归一化查表和 transformers 的算法（float64 乘 1/255、转 float32、减均值除标准差）逐位相同。"""
    values = np.arange(256, dtype=np.uint8)
    expected = ((values.astype(np.float64) * 0.00392156862745098).astype(np.float32) - np.float32(0.5)) / np.float32(0.5)
    assert np.array_equal(_NORMALIZE[values], expected)


def test_smart_resize_keeps_bounds_and_factor() -> None:
    """缩放后长宽是 32 的倍数，像素数落在上下限之间，长宽比大致不变。

    场景：一行字那么小的裁图（17,756 像素，测试文档裁图的中位数）、一整页；另有一张 40×4000 的超宽图，高度被
    32 的倍数卡住，长宽比保不住，只查尺寸和上下限。
    """
    for height, width in ((60, 296), (1754, 1241)):
        h, w = smart_resize(height, width, 32, MIN_PIXELS, MAX_PIXELS)
        assert h % 32 == 0 and w % 32 == 0
        assert MIN_PIXELS <= h * w <= MAX_PIXELS
        assert abs(h / w - height / width) / (height / width) < 0.1

    h, w = smart_resize(40, 4000, 32, MIN_PIXELS, MAX_PIXELS)
    assert h % 32 == 0 and w % 32 == 0 and MIN_PIXELS <= h * w <= MAX_PIXELS


def test_grayscale_and_alpha_images_become_rgb() -> None:
    """灰度图、带透明通道的图先转成 RGB，和插件的 ``convert("RGB")`` 一样。"""
    gray = Image.new("L", (120, 80), 128)
    rgba = Image.new("RGBA", (120, 80), (10, 20, 30, 0))

    assert to_patches(gray).pixels.shape[1] == 3 * 16 * 16
    assert np.allclose(to_patches(rgba).pixels[0, :256], _NORMALIZE[10])


def test_patchify_matches_to_patches() -> None:
    """场景：一张随机内容、非正方形的图（缩放后 28×40 个 patch）。
    条件：引擎的做法——先 resize 拿到 uint8，再用 patchify 查 bf16 的归一化表、切 patch。
    期待：和 to_patches 的 float32 结果转 bf16 逐位相同（两条路只差在哪一步转 bf16）。"""
    torch = pytest.importorskip("torch")
    from oxr_runtime.preprocess import normalize_table, patchify, resize, to_patches

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8))

    want = torch.from_numpy(to_patches(image).pixels).to(torch.bfloat16)
    table = torch.from_numpy(normalize_table()).to(torch.bfloat16)
    got = patchify(torch.from_numpy(resize(image)), table)

    assert got.shape == want.shape
    assert torch.equal(got, want)
