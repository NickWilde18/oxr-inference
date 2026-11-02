"""注意力和旋转位置编码的算子，两套实现可选：

- ``fa2``：flash-attn 2.x（``flash_attn``）。官方只发 torch 2.8 / CUDA 12 的预编译包，再新的 torch 要自己编译；
- ``fa4``：flash-attn 4（``flash_attn.cute``，CuTeDSL 写的，第一次调用时现编内核），支持 Hopper 和 Blackwell，
  不挑 torch 版本。它不带旋转位置编码的算子，这里在 float32 下用 PyTorch 算。

环境变量 ``OXR_ATTENTION`` 选择语言模型用哪套；不设时 flash-attn 2 装了就用它，否则用 4。两套接口在这里统一成
几个函数，模型代码不关心用的是哪套。

视觉编码器的注意力（每张图内部、不分先后）单独用 :func:`varlen_bidirectional`，环境变量
``OXR_VISION_ATTENTION`` 选择；不设时 flash-attn 4 装了就用它。视觉的头维度是 72：flash-attn 2 只有 64、96、
128 几档 kernel，72 维按 96 算，白算四分之一；flash-attn 4 是给 Hopper 写的。H200 上实测，128 张裁图的
一层注意力 flash-attn 2 要 3.43 ms，4 要 2.24 ms。两者结果的末位有差别（约 22% 的元素差一个 bf16 舍入单位）。
flash-attn 4 只装 ``flash_attn/cute`` 这一个子包，和 flash-attn 2 装在一起不冲突。
"""

from __future__ import annotations

import importlib.util
import os

import torch

def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


BACKEND = os.environ.get("OXR_ATTENTION") or ("fa2" if _installed("flash_attn.flash_attn_interface") else "fa4")
VISION_BACKEND = os.environ.get("OXR_VISION_ATTENTION") or ("fa4" if _installed("flash_attn.cute") else BACKEND)

for _name, _value in (("OXR_ATTENTION", BACKEND), ("OXR_VISION_ATTENTION", VISION_BACKEND)):
    if _value not in ("fa2", "fa4"):
        raise ValueError(f"{_name} 只能是 fa2 或 fa4，现在是 {_value!r}")

if "fa2" in (BACKEND, VISION_BACKEND):
    from flash_attn import flash_attn_varlen_func as _fa2_varlen
if BACKEND == "fa2":
    from flash_attn import flash_attn_with_kvcache as _fa2_kvcache
    from flash_attn.layers.rotary import apply_rotary_emb as _fa2_rotary
if "fa4" in (BACKEND, VISION_BACKEND):
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen


def varlen(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int, *,
    causal: bool, scale: float | None = None,
) -> torch.Tensor:
    """若干个序列首尾相接（``[T, heads, dim]``），各序列内部做注意力。``cu_seqlens`` 是 int32。"""
    if BACKEND == "fa2":
        return _fa2_varlen(q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, softmax_scale=scale, causal=causal)

    out = _fa4_varlen(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len,
        softmax_scale=scale, causal=causal,
    )
    return out[0] if isinstance(out, tuple) else out


def varlen_bidirectional(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int
) -> torch.Tensor:
    """视觉编码器用：若干张图的 token 首尾相接（``[T, heads, dim]``），每张图内部做不分先后的注意力。"""
    if VISION_BACKEND == "fa2":
        return _fa2_varlen(q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, causal=False)

    out = _fa4_varlen(
        q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=False,
    )
    return out[0] if isinstance(out, tuple) else out


def paged_decode(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor,
    block_table: torch.Tensor, *, scale: float | None = None,
) -> torch.Tensor:
    """每个序列一个新 token（``q`` 是 ``[B, heads, dim]``），读分页缓存。``seq_lens``、``block_table`` 是 int32，
    ``seq_lens`` 含这个新 token。返回 ``[B, heads, dim]``。"""
    if BACKEND == "fa2":
        out = _fa2_kvcache(
            q.unsqueeze(1), k_cache, v_cache, cache_seqlens=seq_lens, block_table=block_table,
            softmax_scale=scale, causal=True,
        )
        return out.squeeze(1)

    out = _fa4_varlen(
        q.unsqueeze(1), k_cache, v_cache, seqused_k=seq_lens, page_table=block_table, softmax_scale=scale, causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.squeeze(1)


def paged_extend(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor,
    block_table: torch.Tensor, *, scale: float | None = None,
) -> torch.Tensor:
    """每个序列一次进 S 个新 token（``q`` 是 ``[B, S, heads, dim]``，这些 token 的 K、V 已经写进分页缓存），
    因果注意力读分页缓存：第 j 个新 token 只看到它自己和它前面的（``seq_lens`` 含这 S 个）。返回 ``[B, S, heads, dim]``。
    投机解码验证那一步用（见 :mod:`oxr_runtime.speculative`）。"""
    if BACKEND == "fa2":
        return _fa2_kvcache(
            q, k_cache, v_cache, cache_seqlens=seq_lens, block_table=block_table, softmax_scale=scale, causal=True,
        )

    out = _fa4_varlen(q, k_cache, v_cache, seqused_k=seq_lens, page_table=block_table, softmax_scale=scale, causal=True)
    return out[0] if isinstance(out, tuple) else out


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """前后两半旋转（GPT-NeoX 式）。``x`` 是 ``[1, n, heads, dim]``，``cos``、``sin`` 是 ``[n, dim / 2]``。
    在 float32 下算、按 ``x`` 的 dtype 存。"""
    if BACKEND == "fa2":
        return _fa2_rotary(x, cos, sin)

    half = cos.shape[-1]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half : 2 * half]
    c = cos.float()[None, :, None, :]
    s = sin.float()[None, :, None, :]
    rotated = torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    if rotated.shape[-1] < x.shape[-1]:
        rotated = torch.cat((rotated, xf[..., 2 * half :]), dim=-1)
    return rotated.to(x.dtype)
