"""视觉部分：SigLIP2-NaViT 编码器 + 视觉词表头，图片 → 语言模型的 embedding。

结构和参考实现（权重目录的 ``modeling_oxr.py``）一一对应，算子也尽量相同，方便逐位比对：

- patch 嵌入是 16×16、步长 16 的卷积；
- 位置编码：32×32 的原始表按每张图的网格尺寸做 bicubic 插值。参考实现每次前向对每张图都插一遍，
  实测占视觉编码器 GPU 时间的 54%。插值结果只取决于网格尺寸，这里按 ``(h, w)`` 缓存；
- 27 层全注意力（每张图内部），2D 旋转位置编码；token 先按窗口顺序重排、算完再排回来，和参考实现相同
  （全注意力下重排不改变数学结果，只影响浮点累加顺序；保留它是为了逐位比对）；
- 取最后一层的输出（不经过 ``post_layernorm``，参考实现也没用它），每 2×2 个 patch 拼成一行，经线性层 +
  LayerNorm 得到视觉词表上的 logits，softmax 之后乘视觉 embedding 表。

多张图拼成一批算：注意力用 varlen 接口按图切开，其余算子都是逐 token 的，拼批不改变结果。

减少 GPU 操作数：加载权重之后调 :meth:`VisionEncoder.fuse_projections` 把每层的 q/k/v 投影并成一次矩阵乘；
「拆 q/k/v + float32 下套旋转位置编码 + 转回 bf16」写成一个函数，``OXR_COMPILE=1`` 时编成一个 kernel（不编译
时是 6 个：两次转 float32、两次旋转、两次转回）。LayerNorm 不编译：torch 自带的 LayerNorm kernel 比 inductor
生成的快，把残差相加合进去省下的抵不上（同一批 128 张图实测，编译 452 ms、不编译 439 ms）。
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import torch
import torch.nn.functional as F
from torch import nn

from oxr_runtime import attention, language
from oxr_runtime.config import VisionConfig


def _qkv_rotary(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """并起来的 qkv 投影结果拆成 q、k、v，q、k 在 float32 下套旋转位置编码再转回模型的 dtype（参考实现的
    ``apply_rotary_pos_emb_flashatt`` 先把 q、k 转成 float32 再套，结果按 v 的 dtype 存）。"""
    n = qkv.shape[0]
    q, k, v = qkv.view(n, 3, heads, dim).unbind(1)
    # language._rotate 不编译时是 flash-attn 2 的单个 kernel，OXR_COMPILE=1 时是能编进这个函数的纯 PyTorch 写法。
    q = language._rotate(q.float(), cos, sin).to(v.dtype)
    k = language._rotate(k.float(), cos, sin).to(v.dtype)
    # v 写成连续的：它是 qkv 切出来的一段，行不连续，flash-attn 拿到这样的 v 会先自己拷一份连续的再算；
    # 编译后和 q、k 在同一个 kernel 里写出来，不多一次读写。
    return q, k, v.contiguous()


if os.environ.get("OXR_COMPILE") == "1":
    _qkv_rotary = torch.compile(_qkv_rotary, **language.COMPILE_OPTIONS)


_HEAD_CHUNK = 4096
"""视觉词表头一次算多少个视觉 token。4096 行时 logits、float32 的 softmax 加起来约 1.6 GB。"""


@dataclass(frozen=True)
class _GridLayout:
    """一种网格尺寸的位置信息，算一次缓存起来。"""

    position: torch.Tensor
    """``[h * w, hidden]``，已按 2×2 合并的顺序排好的位置编码。"""
    cos: torch.Tensor
    """``[h * w, head_dim / 2]``，已按窗口顺序重排的旋转位置编码。"""
    sin: torch.Tensor
    window_index: torch.Tensor
    """``[h * w / 4]``，合并单元的窗口顺序（参考实现的 ``get_window_index``）。"""


class VisionEncoder(nn.Module):
    """视觉编码器加视觉词表头。权重名和参考实现的 ``visual_tokenizer.*`` 对应，见 :mod:`oxr_runtime.weights`。"""

    def __init__(self, config: VisionConfig, visual_vocab_size: int, indicator_count: int, lm_hidden: int):
        super().__init__()
        self.config = config
        c = config
        self.patch_embedding = nn.Conv2d(3, c.hidden_size, kernel_size=c.patch_size, stride=c.patch_size)
        self.position_embedding = nn.Embedding(c.position_grid * c.position_grid, c.hidden_size)
        self.layers = nn.ModuleList([_EncoderLayer(c) for _ in range(c.num_layers)])
        head_dim = visual_vocab_size - indicator_count
        self.head_proj = nn.Linear(c.hidden_size * c.merge_unit, head_dim, bias=False)
        self.head_norm = nn.LayerNorm(head_dim)
        self.vte = nn.Embedding(visual_vocab_size, lm_hidden)
        self.indicator_count = indicator_count
        self._layouts: dict[tuple[int, int], _GridLayout] = {}

    @property
    def indicator_embeddings(self) -> torch.Tensor:
        """图片开始、结束等 4 个指示符的 embedding：视觉 embedding 表的最后 4 行。"""
        return self.vte.weight[-self.indicator_count :]

    @torch.inference_mode()
    def forward(self, pixels: torch.Tensor, grids: list[tuple[int, int]]) -> torch.Tensor:
        """``pixels`` 是若干张图的 patch 拼在一起（``[sum(h*w), 768]``），``grids`` 是每张图的 ``(h, w)``。

        返回 ``[sum(h*w/4), lm_hidden]``：每张图的视觉 token embedding，顺序同输入。
        """
        c = self.config
        dtype = self.patch_embedding.weight.dtype
        # 要从 CPU 拷到显存的东西（各图的 token 数、新网格尺寸的位置信息）都在发出第一个 kernel 之前拷完：从普通
        # 内存往显存拷，CUDA 要先等这条流上已经发出的 kernel 全部做完，放在中间会让 CPU 一直等到视觉编码算完，
        # 引擎就没法在这段时间里推进逐字生成（见 engine 里 prefill 和 decode 并行的说明）。
        layouts = [self._layout(h, w) for h, w in grids]
        lengths = torch.tensor([h * w for h, w in grids], dtype=torch.int32, device=pixels.device)

        x = self.patch_embedding(pixels.to(dtype).view(-1, 3, c.patch_size, c.patch_size)).reshape(-1, c.hidden_size)
        x = x + torch.cat([layout.position for layout in layouts])

        # 每张图按窗口顺序重排（合并单元为粒度），偏移到拼批后的下标。
        window_index, offset = [], 0
        for layout, (h, w) in zip(layouts, grids, strict=True):
            window_index.append(layout.window_index + offset)
            offset += h * w // c.merge_unit
        window = torch.cat(window_index)
        reverse = torch.argsort(window)

        n = x.shape[0]
        x = x.view(n // c.merge_unit, c.merge_unit, -1)[window].reshape(n, -1)
        cos = torch.cat([layout.cos for layout in layouts])
        sin = torch.cat([layout.sin for layout in layouts])

        cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        max_len = max(h * w for h, w in grids)

        for layer in self.layers:
            x = layer(x, cu_seqlens, max_len, cos, sin)

        x = x.view(n // c.merge_unit, c.merge_unit, -1)[reverse].reshape(n // c.merge_unit, -1)

        # 视觉词表头一行一个视觉 token、6.5 万列：一批 2.5 万个视觉 token 时，float32 的 softmax 就要 6.5 GB。
        # 每行互不相关，分块算，峰值只看块大小。
        out = torch.empty(x.shape[0], self.vte.weight.shape[1], dtype=self.vte.weight.dtype, device=x.device)
        for start in range(0, x.shape[0], _HEAD_CHUNK):
            out[start : start + _HEAD_CHUNK] = self._head(x[start : start + _HEAD_CHUNK])
        return out

    def fuse_projections(self) -> None:
        """每层的 q/k/v 投影并成一个矩阵，拆掉原来的三个；视觉词表头的权重补到 64 的倍数行。加载权重之后调一次。

        视觉词表头输出 65532 列（65536 减去 4 个指示符），不是 8 的倍数，cuBLAS 只能选很慢的 kernel：H200
        2.75 万行时 60 ms（276 TFLOPS）。权重补 4 行 0、输出 65536 列只要 25 ms，前 65532 列和不补时逐位相同
        （64 到 2.75 万行各测过一次，一个元素都不差），:meth:`_head` 只取前 65532 列。"""
        for layer in self.layers:
            layer.fuse_projections()
        weight = self.head_proj.weight
        pad = -weight.shape[0] % 64
        if pad:
            with torch.no_grad():
                padded = torch.cat((weight, weight.new_zeros(pad, weight.shape[1])))
            self.head_proj.weight = nn.Parameter(padded, requires_grad=False)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head_norm(self.head_proj(x)[:, : self.head_norm.normalized_shape[0]])
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        # 指示符那 4 列补 0 再乘整张表，和参考实现的矩阵形状相同（只乘前面的行结果相同，但 GEMM 的分块不同，
        # 末位会有差别）。
        probs = F.pad(probs, (0, self.indicator_count))
        return probs @ self.vte.weight

    def _layout(self, h: int, w: int) -> _GridLayout:
        key = (h, w)
        if key not in self._layouts:
            self._layouts[key] = self._build_layout(h, w)
        return self._layouts[key]

    def _build_layout(self, h: int, w: int) -> _GridLayout:
        c = self.config
        s = c.hidden_stride
        device = self.position_embedding.weight.device

        # 位置编码：原始表插值到 (h, w)，再排成 2×2 合并的顺序（参考实现 Siglip2VisionEmbeddings.forward）。
        table = self.position_embedding.weight.reshape(c.position_grid, c.position_grid, -1).unsqueeze(0).permute(0, 3, 1, 2)
        pe = F.interpolate(table, size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(h // s, s, w // s, s, -1).permute(0, 2, 1, 3, 4).reshape(h * w, -1)

        # 2D 旋转位置编码（参考实现 rot_pos_emb）：行号、列号各占一半维度。
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        hpos = hpos.reshape(h // s, s, w // s, s).permute(0, 2, 1, 3).flatten()
        wpos = wpos.reshape(h // s, s, w // s, s).permute(0, 2, 1, 3).flatten()
        # 频率在 float32 下现算（参考实现的 VisionRotaryEmbedding.forward 也是每次现算）：不做成 buffer，否则
        # 整个模型 .to(bfloat16) 时会被一起转成 bf16。
        rotary_dim = c.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float) / rotary_dim))
        freqs = torch.outer(torch.arange(max(h, w), device=device, dtype=torch.float), inv_freq)
        rotary = freqs[torch.stack([hpos, wpos], dim=-1)].flatten(1)  # [h*w, head_dim/2]

        window_index = self._window_index(h, w).to(device)
        units = h * w // c.merge_unit
        rotary = rotary.view(units, c.merge_unit, -1)[window_index].reshape(h * w, -1)

        return _GridLayout(
            position=pe.contiguous(),
            cos=rotary.cos().contiguous(),
            sin=rotary.sin().contiguous(),
            window_index=window_index,
        )

    def _window_index(self, h: int, w: int) -> torch.Tensor:
        """参考实现 ``get_window_index`` 的单图版本：合并单元按窗口逐个排列。"""
        c = self.config
        win = c.window_size // c.hidden_stride // c.patch_size
        gh, gw = h // c.hidden_stride, w // c.hidden_stride
        index = torch.arange(gh * gw).reshape(gh, gw)
        pad_h, pad_w = win - gh % win, win - gw % win
        padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        nh, nw = (gh + pad_h) // win, (gw + pad_w) // win
        padded = padded.reshape(nh, win, nw, win).permute(0, 2, 1, 3).reshape(-1)
        return padded[padded != -100]


class _EncoderLayer(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.c = c
        self.layer_norm1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.out_proj = nn.Linear(c.hidden_size, c.hidden_size)
        self.fc1 = nn.Linear(c.hidden_size, c.intermediate_size)
        self.fc2 = nn.Linear(c.intermediate_size, c.hidden_size)

    def fuse_projections(self) -> None:
        c = self.c
        with torch.no_grad():
            weight = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
            bias = torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias))
        self.qkv_proj = nn.Linear(c.hidden_size, 3 * c.hidden_size, device=weight.device, dtype=weight.dtype)
        self.qkv_proj.weight = nn.Parameter(weight, requires_grad=False)
        self.qkv_proj.bias = nn.Parameter(bias, requires_grad=False)
        del self.q_proj, self.k_proj, self.v_proj

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[0]
        heads, dim = self.c.num_heads, self.c.head_dim

        q, k, v = _qkv_rotary(self.qkv_proj(self.layer_norm1(x)), cos, sin, heads, dim)
        attn = attention.varlen_bidirectional(q, k, v, cu_seqlens, max_len)
        x = x + self.out_proj(attn.reshape(n, -1))

        h = self.fc2(F.gelu(self.fc1(self.layer_norm2(x)), approximate="tanh"))
        return x + h
