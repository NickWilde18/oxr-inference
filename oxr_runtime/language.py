"""语言模型：Qwen3 结构（28 层、GQA 16/8、head_dim 128、q/k 各自 RMSNorm、RoPE θ=1e6）。

两种前向：

- :meth:`LanguageModel.prefill`：若干个序列的整段输入拼在一起算（varlen 因果注意力），KV 写进分页缓存；
- :meth:`LanguageModel.decode`：每个序列各进一个新 token，注意力读分页缓存。

数值上照 vLLM 的做法：RoPE 的 cos/sin 先算成 float32 再存成模型的 dtype，套用时在 float32 下算；RMSNorm 在
float32 下归一化、乘权重前转回模型 dtype（和 HF、vLLM 相同）；残差相加在模型 dtype 下做（vLLM 的
``fused_add_rms_norm`` 也是先加、按模型 dtype 存，再拿存下来的值归一化）。

减少 GPU 操作数（逐字生成时每一步要下发的 kernel 数）：

- 加载权重之后调 :meth:`LanguageModel.fuse_projections`，q/k/v 三个投影、gate/up 两个投影各并成一次矩阵乘
  （vLLM 的 ``QKVParallelLinear``、``MergedColumnParallelLinear`` 也是这样并的）；
- 「q/k 各自 RMSNorm + 旋转位置编码 + K/V 写进分页缓存」是一个手写的 Triton kernel（见
  :mod:`oxr_runtime.kernels`）；
- 其余矩阵乘之间的逐元素操作写成两个函数：残差相加加 RMSNorm、SiLU 乘法。环境变量 ``OXR_COMPILE=1`` 时用
  ``torch.compile`` 把每个函数各编成一个 kernel；不设时照常逐个操作算，结果是同一组运算，只是 kernel 多。编译
  后归约的顺序可能不同，末位会有差别，所以打开前后要整批重测准确率。
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import torch
import torch.nn.functional as F
from torch import nn

from oxr_runtime import attention, kernels
from oxr_runtime.config import LanguageConfig
from oxr_runtime.kv_cache import BLOCK_SIZE, KVCache


@dataclass
class PrefillBatch:
    """一次 prefill 的输入：若干个序列首尾相接。"""

    embeds: torch.Tensor
    """``[T, hidden]``。"""
    positions: torch.Tensor
    """``[T]``，每个 token 在它自己序列里的位置。"""
    cu_seqlens: torch.Tensor
    """``[n + 1]``，int32。"""
    max_len: int
    slots: torch.Tensor
    """``[T]``，每个 token 的 KV 写到缓存的哪一格（页号 × 256 + 页内偏移）。"""


@dataclass
class DecodeBatch:
    """一步 decode 的输入：每个序列一个新 token。"""

    embeds: torch.Tensor
    """``[B, hidden]``。"""
    positions: torch.Tensor
    """``[B]``。"""
    slots: torch.Tensor
    """``[B]``。"""
    seq_lens: torch.Tensor
    """``[B]``，int32，含这个新 token 之后的长度。"""
    block_table: torch.Tensor
    """``[B, max_blocks]``，int32。"""


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """前后两半旋转（GPT-NeoX 式），在 float32 下算、按 ``x`` 的 dtype 存。``x`` 是 ``[n, heads, dim]``，
    ``cos``、``sin`` 是 ``[n, dim / 2]``。

    不编译时用 :func:`oxr_runtime.attention.rotary`（flash-attn 2 有现成的单个 kernel）；``OXR_COMPILE=1`` 时换成
    下面的纯 PyTorch 写法，好让它和前后的归一化编进同一个 kernel。两种写法算式相同。"""
    return attention.rotary(x.unsqueeze(0), cos, sin).squeeze(0)


def rotate_traceable(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """:func:`_rotate` 的纯 PyTorch 写法，算式和 flash-attn 2 的 ``apply_rotary_emb``（不交错）相同。"""
    half = cos.shape[-1]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    c = cos.float()[:, None, :]
    s = sin.float()[:, None, :]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)


def _add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual + x`` 按 ``x`` 的 dtype 存下来作为新的残差，再归一化。返回（归一化结果，新的残差）。"""
    residual = x + residual
    return _rms_norm(residual, weight, eps), residual


def _silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


COMPILE_OPTIONS = {"dynamic": None, "fullgraph": True}
"""``OXR_COMPILE=1`` 时编译逐元素函数的选项（视觉编码器也用这一份）。

``dynamic=None`` 是 torch 的自动模式：第一次按具体形状编，之后哪一维变过（批大小、prefill 的 token 数）就把
哪一维改成可变的重编一次；隐藏维度、头数这些从来不变的维度一直按常数编，生成的 kernel 能按固定尺寸展开。
``dynamic=True`` 把所有维度和整数参数都当成可变的：语言模型这边 kernel 更慢，视觉编码器那边 torch 2.8 的
inductor 甚至在旋转的切片上推导取值范围时报错。``fullgraph`` 保证编不进去时直接报错，而不是悄悄退回逐个操作。
CUDA Graph 录制前的预热会触发编译，录下来的是编好的 kernel。"""

if os.environ.get("OXR_COMPILE") == "1":
    _rotate = rotate_traceable
    _add_rms_norm = torch.compile(_add_rms_norm, **COMPILE_OPTIONS)
    _silu_mul = torch.compile(_silu_mul, **COMPILE_OPTIONS)


@dataclass
class ExtendBatch:
    """一步投机解码的验证：每个序列进同样多（S）个新 token，注意力读分页缓存。"""

    embeds: torch.Tensor
    """``[B · S, hidden]``，按序列排、序列内按位置排。"""
    positions: torch.Tensor
    """``[B · S]``。"""
    slots: torch.Tensor
    """``[B · S]``。"""
    seq_lens: torch.Tensor
    """``[B]``，int32，含这 S 个新 token 之后的长度。"""
    block_table: torch.Tensor
    """``[B, max_blocks]``，int32。"""
    tokens_per_seq: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


class LanguageModel(nn.Module):
    """权重名和参考实现的 ``llm.*`` 对应，见 :mod:`oxr_runtime.weights`。"""

    def __init__(self, config: LanguageConfig, max_positions: int):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float) / config.head_dim)
        )
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float), inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def cast_rope(self, dtype: torch.dtype) -> None:
        """cos/sin 表存成模型的 dtype（vLLM 的 cos_sin_cache 也是这样存的）。加载权重之后调一次。"""
        self.rope_cos = self.rope_cos.to(dtype)
        self.rope_sin = self.rope_sin.to(dtype)

    def fuse_projections(self) -> None:
        """每层的 q/k/v、gate/up 投影各并成一个矩阵，拆掉原来的几个。加载权重之后、录 CUDA Graph 之前调一次。"""
        for layer in self.layers:
            layer.fuse_projections()

    @torch.inference_mode()
    def prefill(self, batch: PrefillBatch, cache: KVCache) -> torch.Tensor:
        """返回 ``[T, hidden]``，最后一层归一化之后的隐状态。"""
        cos, sin = self.rope_cos[batch.positions], self.rope_sin[batch.positions]
        x = batch.embeds
        residual = None

        for index, layer in enumerate(self.layers):
            x, residual = layer.prefill(x, residual, cos, sin, batch, cache, index)

        return _add_rms_norm(x, residual, self.norm.weight, self.norm.eps)[0]

    @torch.inference_mode()
    def decode(self, batch: DecodeBatch, cache: KVCache) -> torch.Tensor:
        """返回 ``[B, hidden]``。"""
        cos, sin = self.rope_cos[batch.positions], self.rope_sin[batch.positions]
        x = batch.embeds
        residual = None

        for index, layer in enumerate(self.layers):
            x, residual = layer.decode(x, residual, cos, sin, batch, cache, index)

        return _add_rms_norm(x, residual, self.norm.weight, self.norm.eps)[0]

    @torch.inference_mode()
    def extend(self, batch: ExtendBatch, cache: KVCache) -> torch.Tensor:
        """返回 ``[B · S, hidden]``：每个新 token 位置上最后一层归一化之后的隐状态。"""
        cos, sin = self.rope_cos[batch.positions], self.rope_sin[batch.positions]
        x = batch.embeds
        residual = None

        for index, layer in enumerate(self.layers):
            x, residual = layer.extend(x, residual, cos, sin, batch, cache, index)

        return _add_rms_norm(x, residual, self.norm.weight, self.norm.eps)[0]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


class _DecoderLayer(nn.Module):
    def __init__(self, c: LanguageConfig):
        super().__init__()
        self.c = c
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)
        self.scale = c.head_dim**-0.5

    def fuse_projections(self) -> None:
        c = self.c
        with torch.no_grad():
            qkv = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight))
            gate_up = torch.cat((self.gate_proj.weight, self.up_proj.weight))
        self.qkv_proj = nn.Linear(c.hidden_size, qkv.shape[0], bias=False, device=qkv.device, dtype=qkv.dtype)
        self.gate_up_proj = nn.Linear(
            c.hidden_size, gate_up.shape[0], bias=False, device=gate_up.device, dtype=gate_up.dtype
        )
        self.qkv_proj.weight = nn.Parameter(qkv, requires_grad=False)
        self.gate_up_proj.weight = nn.Parameter(gate_up, requires_grad=False)
        del self.q_proj, self.k_proj, self.v_proj, self.gate_proj, self.up_proj

    def _attention_inputs(
        self, x: torch.Tensor, residual: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor,
        cache: KVCache, index: int, slots: torch.Tensor, *, need_kv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """残差相加、RMSNorm、q/k/v 投影，再交给 Triton kernel 做 q/k 归一化和旋转、把 K/V 写进缓存。
        返回（q，``need_kv`` 时这几个 token 的 k、v，新的残差）。"""
        norm = self.input_layernorm
        if residual is None:
            residual, h = x, norm(x)
        else:
            h, residual = _add_rms_norm(x, residual, norm.weight, norm.eps)

        c = self.c
        q, k, v = kernels.qk_norm_rope_store(
            self.qkv_proj(h), self.q_norm.weight, self.k_norm.weight, cos, sin, c.rms_norm_eps,
            cache.k[index], cache.v[index], slots, c.num_heads, c.num_kv_heads, c.head_dim, need_kv=need_kv,
        )
        return q, k, v, residual

    def _mlp(self, attn: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = self.post_attention_layernorm
        h, residual = _add_rms_norm(self.o_proj(attn), residual, norm.weight, norm.eps)
        return self.down_proj(_silu_mul(self.gate_up_proj(h))), residual

    def prefill(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: PrefillBatch,
        cache: KVCache,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, residual = self._attention_inputs(x, residual, cos, sin, cache, index, batch.slots, need_kv=True)
        attn = attention.varlen(q, k, v, batch.cu_seqlens, batch.max_len, causal=True, scale=self.scale)
        return self._mlp(attn.reshape(x.shape[0], -1), residual)

    def decode(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: DecodeBatch,
        cache: KVCache,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, _, _, residual = self._attention_inputs(x, residual, cos, sin, cache, index, batch.slots, need_kv=False)
        attn = attention.paged_decode(
            q, cache.k[index], cache.v[index], batch.seq_lens, batch.block_table, scale=self.scale
        )
        return self._mlp(attn.reshape(x.shape[0], -1), residual)

    def extend(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: ExtendBatch,
        cache: KVCache,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, _, _, residual = self._attention_inputs(x, residual, cos, sin, cache, index, batch.slots, need_kv=False)
        b, s = batch.block_table.shape[0], batch.tokens_per_seq
        attn = attention.paged_extend(
            q.view(b, s, *q.shape[1:]), cache.k[index], cache.v[index], batch.seq_lens, batch.block_table,
            scale=self.scale,
        )
        return self._mlp(attn.reshape(x.shape[0], -1), residual)


def slot_of(blocks: list[int], position: int) -> int:
    """第 ``position`` 个 token（从 0 数）在缓存里的格号。"""
    return blocks[position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE
