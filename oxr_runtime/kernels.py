"""手写的 Triton kernel：语言模型每层注意力前面那一串小操作合成一个。

:func:`qk_norm_rope_store` 接在 q/k/v 投影（一次矩阵乘）后面，一个 kernel 做完：

- q 的每个头：RMSNorm（float32 下归一化、乘权重前转回 bf16）→ 旋转位置编码（float32 下算、按 bf16 存）→ 写出；
- k 的每个头：同样的归一化和旋转，直接写进分页 KV 缓存；
- v 的每个头：原样写进分页 KV 缓存。

这一段交给 torch.compile 时是 7 个 kernel，再加两次写缓存（索引赋值），每层 9 个；逐字生成的每一步 28 层就是
250 多个，几乎都只有几微秒，主要耗在 kernel 之间的空隙上。prefill 时注意力要用这几个 token 自己的 k、v，
``need_kv=True`` 时另外写出连续的 k、v。

每个 token 起三个程序，分别算 q 的全部头、k 的全部头、v 的全部头。一个头一个程序时（3.8 万个 token 就是 120 万个
程序，每个只搬 256 字节），时间都耗在访存延迟上。H200 上（CUDA Graph 里）实测，prefill 3.8 万个 token 每层
0.20 ms（一个头一个程序 0.78 ms，逐个操作的 PyTorch 写法 3.9 ms），逐字生成批 4–1,024 时每层 1.7–4.4 µs
（一个头一个程序 1.7–21.6 µs）。

舍入点和逐个操作的 PyTorch 写法（:func:`qk_norm_rope_store_reference`，CPU 上用它）相同，平方和的加法顺序也相同：
两边都固定成一个头一个程序时 Triton 归约的顺序（见 :func:`_sum_squares`）。实测 4 到 3.8 万个 token，q、k、v 和
写进缓存的值，这个 kernel、一个头一个程序的写法、PyTorch 写法三者逐位相同；PyTorch 写法要是直接用 ``mean``，每一百万个
输出约有几个末位不同。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_squares(x, ROWS: tl.constexpr):
    """``x`` 是 ``ROWS`` 个头的前半或后半（``[ROWS, 64]``，float32），返回每个头这 64 个数的平方和（``[ROWS]``）。

    加法顺序固定成「一个程序只算一个头」时 Triton 归约的顺序：64 个数分在两个 warp 的 32 个线程上，warp 内按蝶形
    跨度 16、8、4、2、1 两两相加，最后两个 warp 相加。直接 ``tl.sum(axis=1)`` 时二维布局让每个线程先在线程内
    累加几个数，顺序变了，实测每一百万个输出约有两个末位不同。"""
    y = tl.reshape(x * x, (ROWS, 2, 2, 2, 2, 2, 2))  # 第 1–6 维依次是元素号的第 5–0 位
    y = tl.sum(y, axis=2)  # 第 4 位（跨度 16）
    y = tl.sum(y, axis=2)  # 第 3 位
    y = tl.sum(y, axis=2)  # 第 2 位
    y = tl.sum(y, axis=2)  # 第 1 位
    y = tl.sum(y, axis=2)  # 第 0 位
    return tl.sum(y, axis=1)  # 第 5 位：两个 warp


@triton.jit
def _norm_rope(x1, x2, weight, cos, sin, t, eps, ROWS: tl.constexpr, D: tl.constexpr):
    """第 ``t`` 个 token 的 ``ROWS`` 个头，前后两半 ``x1``、``x2``（``[ROWS, D/2]``，bf16）：RMSNorm 再旋转，
    返回旋转后的两半（float32，调用方转 bf16）。"""
    half = tl.arange(0, D // 2)
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    var = (_sum_squares(x1, ROWS) + _sum_squares(x2, ROWS)) / D
    r = tl.rsqrt(var + eps)[:, None]
    w1 = tl.load(weight + half).to(tl.float32)[None, :]
    w2 = tl.load(weight + D // 2 + half).to(tl.float32)[None, :]
    # 归一化的结果先转 bf16，再和权重相乘、按 bf16 存（同 _rms_norm：weight * h.to(dtype)）。
    y1 = (w1 * (x1 * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y2 = (w2 * (x2 * r).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    c = tl.load(cos + t * (D // 2) + half).to(tl.float32)[None, :]
    s = tl.load(sin + t * (D // 2) + half).to(tl.float32)[None, :]
    return y1 * c - y2 * s, y1 * s + y2 * c


@triton.jit
def _qk_norm_rope_store_kernel(
    qkv, q_out, k_out, v_out, k_cache, v_cache, slots, q_weight, k_weight, cos, sin, eps,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, NEED_KV: tl.constexpr,
):
    tl.static_assert(D == 128, "_sum_squares 的加法顺序按每半 64 个数写死")
    t = tl.program_id(0)
    part = tl.program_id(1)  # 0：q 的全部头；1：k 的全部头；2：v 的全部头
    half = tl.arange(0, D // 2)
    row = qkv + t * (H + 2 * KV) * D

    if part == 0:
        q_heads = tl.arange(0, H)[:, None] * D + half[None, :]
        q1, q2 = _norm_rope(tl.load(row + q_heads), tl.load(row + q_heads + D // 2), q_weight, cos, sin, t, eps, H, D)
        q_dst = q_out + t * H * D + q_heads
        tl.store(q_dst, q1.to(tl.bfloat16))
        tl.store(q_dst + D // 2, q2.to(tl.bfloat16))
    elif part == 1:
        k_heads = tl.arange(0, KV)[:, None] * D + half[None, :]
        k_src = row + H * D + k_heads
        k1, k2 = _norm_rope(tl.load(k_src), tl.load(k_src + D // 2), k_weight, cos, sin, t, eps, KV, D)
        k_dst = k_cache + tl.load(slots + t) * KV * D + k_heads
        tl.store(k_dst, k1.to(tl.bfloat16))
        tl.store(k_dst + D // 2, k2.to(tl.bfloat16))
        if NEED_KV:
            k_dst = k_out + t * KV * D + k_heads
            tl.store(k_dst, k1.to(tl.bfloat16))
            tl.store(k_dst + D // 2, k2.to(tl.bfloat16))
    else:
        v_heads = tl.arange(0, KV)[:, None] * D + tl.arange(0, D)[None, :]
        v = tl.load(row + (H + KV) * D + v_heads)
        tl.store(v_cache + tl.load(slots + t) * KV * D + v_heads, v)
        if NEED_KV:
            tl.store(v_out + t * KV * D + v_heads, v)


def qk_norm_rope_store(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
    num_heads: int, num_kv_heads: int, head_dim: int, *, need_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``qkv`` 是 ``[n, (H + 2·KV)·D]`` 的投影结果（bf16、连续）；``cos``、``sin`` 是这 n 个 token 的
    ``[n, D/2]``；``k_cache``、``v_cache`` 是这一层的分页缓存（``[页数, 页大小, KV, D]``），第 i 个 token 写进
    第 ``slots[i]`` 格。返回 ``q``（``[n, H, D]``），``need_kv`` 时还有连续的 ``k``、``v``（``[n, KV, D]``）。"""
    args = (qkv, q_weight, k_weight, cos, sin, eps, k_cache, v_cache, slots, num_heads, num_kv_heads, head_dim)
    if not qkv.is_cuda:
        return qk_norm_rope_store_reference(*args, need_kv=need_kv)
    return _launch(*args, need_kv=need_kv)


def _launch(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
    num_heads: int, num_kv_heads: int, head_dim: int, *, need_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    n = qkv.shape[0]
    q = qkv.new_empty(n, num_heads, head_dim)
    k = qkv.new_empty(n, num_kv_heads, head_dim) if need_kv else q
    v = qkv.new_empty(n, num_kv_heads, head_dim) if need_kv else q
    if n:
        _qk_norm_rope_store_kernel[(n, 3)](
            qkv, q, k, v, k_cache, v_cache, slots, q_weight, k_weight, cos, sin, eps,
            H=num_heads, KV=num_kv_heads, D=head_dim, NEED_KV=need_kv, num_warps=4,
        )
    return (q, k, v) if need_kv else (q, None, None)


def qk_norm_rope_store_reference(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    eps: float, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
    num_heads: int, num_kv_heads: int, head_dim: int, *, need_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """:func:`qk_norm_rope_store` 的 PyTorch 写法（同一套算式），CPU 上用。"""
    n, half = qkv.shape[0], head_dim // 2
    q, k, v = qkv.view(n, num_heads + 2 * num_kv_heads, head_dim).split((num_heads, num_kv_heads, num_kv_heads), 1)

    def norm_rope(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        h = x.float()
        var = (_sum_squares_reference(h[..., :half]) + _sum_squares_reference(h[..., half:])) / head_dim
        h = h * torch.rsqrt(var + eps)[..., None]
        y = (weight * h.to(x.dtype)).float()
        c, s = cos.float()[:, None, :], sin.float()[:, None, :]
        y1, y2 = y[..., :half], y[..., half:]
        return torch.cat((y1 * c - y2 * s, y1 * s + y2 * c), dim=-1).to(x.dtype)

    q, k = norm_rope(q, q_weight), norm_rope(k, k_weight)
    flat = (-1, num_kv_heads, head_dim)
    k_cache.view(flat)[slots] = k
    v_cache.view(flat)[slots] = v
    return (q, k, v.contiguous()) if need_kv else (q, None, None)


def _sum_squares_reference(x: torch.Tensor) -> torch.Tensor:
    """最后一维 64 个数的平方和，加法顺序同 :func:`_sum_squares`（先两两相加跨度 16、8、4、2、1 的，最后前后两个 32 相加）。"""
    y = (x * x).unflatten(-1, (2, -1))
    while y.shape[-1] > 1:
        y = y.unflatten(-1, (2, -1))
        y = y[..., 0, :] + y[..., 1, :]
    y = y[..., 0]
    return y[..., 0] + y[..., 1]
