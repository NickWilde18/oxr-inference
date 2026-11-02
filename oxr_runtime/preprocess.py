"""图片 → 视觉编码器的输入（patch 序列）。

和官方 vLLM 插件（``oxr_processor.py``）的预处理逐位一致：先按 :func:`smart_resize` 定尺寸，PIL 双线性缩放，
``x / 255`` 之后按均值 0.5、标准差 0.5 归一化，再切成 16×16 的 patch，按 2×2 合并的顺序排好。

插件那条路一张图要约 50 ms（缩放之后 transformers 的图片处理器逐步拷贝、``torch.stack``、numpy 切 patch 时
reshape）。这里把归一化换成查表（256 个取值，算法和插件相同，所以结果逐位相同），切 patch 只做一次 transpose。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

MIN_PIXELS = 200704
"""缩放后的最少像素。模型按这个尺寸训练，调低会让输出变坏（实测：块的输出 41% 变了，6 块陷进复读）。"""
MAX_PIXELS = 921600

_RESCALE = 0.00392156862745098
_MEAN = 0.5
_STD = 0.5


def _normalize_table() -> np.ndarray:
    """0–255 每个取值归一化之后的 float32，算法和 transformers 的 rescale + normalize 相同。"""
    values = np.arange(256, dtype=np.float64) * _RESCALE
    values = values.astype(np.float32)
    mean = np.array(_MEAN, dtype=np.float32)
    std = np.array(_STD, dtype=np.float32)
    return (values - mean) / std


_NORMALIZE = _normalize_table()


def _normalize_table_torchvision() -> np.ndarray:
    """transformers 5 默认的 torchvision 后端的归一化：均值、标准差先乘 255（127.5），再 ``(x - 127.5) / 127.5``。
    和 :func:`_normalize_table` 数学上相同，float32 的舍入不同（末位差约 6e-8）。"""
    values = np.arange(256, dtype=np.float32)
    return (values - np.float32(127.5)) / np.float32(127.5)


_NORMALIZE_TORCHVISION = _normalize_table_torchvision()

BACKEND = os.environ.get("OXR_PREPROCESS", "pil")
"""``pil``：和 OXR 参考实现一致（``modeling_oxr.py`` 在 transformers 5 下指定 PIL 后端）；
``torchvision``：和官方 vLLM 插件实际用的一致（它没指定后端，transformers 5 默认是 torchvision）。
两者缩放的结果有少数像素差一个灰度级。"""


@dataclass(frozen=True)
class Patches:
    """一张图的视觉输入。"""

    pixels: np.ndarray
    """``[grid_h * grid_w, 3 * patch * patch]``，float32，按 2×2 合并的顺序排列。"""
    grid_h: int
    grid_w: int

    @property
    def num_tokens(self) -> int:
        """进语言模型的视觉 token 数（2×2 个 patch 合成一个）。"""
        return self.grid_h * self.grid_w // 4


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """长宽都取 ``factor`` 的倍数，像素数落在 ``[min_pixels, max_pixels]``，尽量保持长宽比。照搬插件的算法。"""
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def resize(
    image: Image.Image,
    *,
    patch_size: int = 16,
    hidden_stride: int = 2,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> np.ndarray:
    """按 :func:`smart_resize` 定的尺寸缩放，返回 ``[H, W, 3]`` 的 uint8（还没归一化）。"""
    if image.mode != "RGB":
        image = image.convert("RGB")

    min_pixels = min(max_pixels, min_pixels)
    width, height = image.size
    resized_h, resized_w = smart_resize(height, width, patch_size * hidden_stride, min_pixels, max_pixels)

    if BACKEND == "torchvision":
        return _resize_torchvision(image, resized_h, resized_w)
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR, reducing_gap=None)
    return np.array(resized, dtype=np.uint8)  # 拷一份可写的：torch.from_numpy 遇到只读数组会告警


def normalize_table() -> np.ndarray:
    """0–255 每个灰度级归一化之后的 float32（随 ``OXR_PREPROCESS`` 选的后端）。"""
    return _NORMALIZE_TORCHVISION if BACKEND == "torchvision" else _NORMALIZE


def to_patches(
    image: Image.Image,
    *,
    patch_size: int = 16,
    hidden_stride: int = 2,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Patches:
    """把一张图变成视觉编码器的输入（全在 CPU 上算）。引擎在 GPU 上做后半段，见 :func:`patchify`。"""
    resized = resize(image, patch_size=patch_size, hidden_stride=hidden_stride, min_pixels=min_pixels,
                     max_pixels=max_pixels)
    grid_h, grid_w = resized.shape[0] // patch_size, resized.shape[1] // patch_size
    pixels = normalize_table()[resized]  # [H, W, 3]
    s, p = hidden_stride, patch_size

    # [H, W, C] → [gh/s, s, p, gw/s, s, p, C] → [gh/s, gw/s, s, s, C, p, p]
    blocks = pixels.reshape(grid_h // s, s, p, grid_w // s, s, p, 3)
    blocks = blocks.transpose(0, 3, 1, 4, 6, 2, 5)
    flat = np.ascontiguousarray(blocks).reshape(grid_h * grid_w, 3 * p * p)

    return Patches(pixels=flat, grid_h=grid_h, grid_w=grid_w)


def patchify(resized: "torch.Tensor", table: "torch.Tensor", *, patch_size: int = 16, hidden_stride: int = 2) -> "torch.Tensor":
    """:func:`to_patches` 的后半段（查表归一化、按 2×2 合并的顺序切 patch），在 ``resized`` 所在的设备上算。

    ``resized`` 是 :func:`resize` 的结果（uint8 ``[H, W, 3]``），``table`` 是 :func:`normalize_table` 转成的张量
    （dtype 决定输出的 dtype）。查表和重排都只是挪数据，结果和 :func:`to_patches` 再转成 ``table`` 的 dtype
    逐位相同。返回 ``[grid_h · grid_w, 3 · patch · patch]``。"""
    grid_h, grid_w = resized.shape[0] // patch_size, resized.shape[1] // patch_size
    s, p = hidden_stride, patch_size
    pixels = table[resized.long()]
    blocks = pixels.view(grid_h // s, s, p, grid_w // s, s, p, 3).permute(0, 3, 1, 4, 6, 2, 5)
    return blocks.reshape(grid_h * grid_w, 3 * p * p)


def _resize_torchvision(image: Image.Image, height: int, width: int) -> np.ndarray:
    """torchvision 的双线性缩放（带抗锯齿），直接在 uint8 上做，和 transformers 的 torchvision 后端相同。返回 [H, W, 3]。"""
    import torch
    import torchvision.transforms.v2.functional as tvF

    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    resized = tvF.resize(tensor, [height, width], interpolation=tvF.InterpolationMode.BILINEAR, antialias=True)
    return resized.permute(1, 2, 0).numpy()
