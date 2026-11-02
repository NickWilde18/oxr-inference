"""视觉编码器合并 q/k/v 投影、合并逐元素操作之后，算出来的仍是 SigLIP2-NaViT 的结果。

flash-attn 只收 GPU 上的 bf16/fp16：沿用 ``test_language_fused`` 放进去的纯 PyTorch ``oxr_runtime.attention``。
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

import test_language_fused  # noqa: E402,F401  —— 放好注意力的替身之后才能导入视觉编码器
from oxr_runtime.config import VisionConfig  # noqa: E402
from oxr_runtime.vision import VisionEncoder  # noqa: E402

_CONFIG = VisionConfig(
    hidden_size=48, intermediate_size=80, num_layers=3, num_heads=4, patch_size=4, hidden_stride=2,
    window_size=16, layer_norm_eps=1e-6, position_grid=4,
)


def _encoder() -> VisionEncoder:
    torch.manual_seed(0)
    enc = VisionEncoder(_CONFIG, visual_vocab_size=40, indicator_count=4, lm_hidden=24)
    with torch.no_grad():
        for name, p in enc.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in name and name.endswith("weight") else 0.2 * torch.randn_like(p))
    return enc.eval().requires_grad_(False)


def _reference_layers(enc: VisionEncoder, x: torch.Tensor, grids: list[tuple[int, int]], cos, sin) -> torch.Tensor:
    """逐层手写（合并之前的权重、float32）：LayerNorm → q/k/v → float32 下旋转 → 每张图内部全注意力 → 残差 → MLP。"""
    c = _CONFIG
    heads, dim = c.num_heads, c.head_dim
    bounds = [0]
    for h, w in grids:
        bounds.append(bounds[-1] + h * w)
    half = cos.shape[-1]

    def rope(t):
        t1, t2 = t[..., :half], t[..., half:]
        cc, ss = cos[:, None, :], sin[:, None, :]
        return torch.cat((t1 * cc - t2 * ss, t1 * ss + t2 * cc), dim=-1)

    for layer in enc.layers:
        h = F.layer_norm(x, (c.hidden_size,), layer.layer_norm1.weight, layer.layer_norm1.bias, c.layer_norm_eps)
        q = rope((h @ layer.q_proj.weight.T + layer.q_proj.bias).view(-1, heads, dim))
        k = rope((h @ layer.k_proj.weight.T + layer.k_proj.bias).view(-1, heads, dim))
        v = (h @ layer.v_proj.weight.T + layer.v_proj.bias).view(-1, heads, dim)
        out = []
        for a, b in zip(bounds, bounds[1:]):
            scores = torch.einsum("nhd,mhd->hnm", q[a:b], k[a:b]) * dim**-0.5
            out.append(torch.einsum("hnm,mhd->nhd", scores.softmax(-1), v[a:b]))
        x = x + torch.cat(out).reshape(x.shape[0], -1) @ layer.out_proj.weight.T + layer.out_proj.bias
        h = F.layer_norm(x, (c.hidden_size,), layer.layer_norm2.weight, layer.layer_norm2.bias, c.layer_norm_eps)
        h = F.gelu(h @ layer.fc1.weight.T + layer.fc1.bias, approximate="tanh")
        x = x + h @ layer.fc2.weight.T + layer.fc2.bias
    return x


def test_fused_vision_matches_reference_float32() -> None:
    """场景：两张图（4×8、8×4 个 patch）拼成一批过视觉编码器。
    条件：合并 q/k/v 投影，float32。
    期待：最后一层的输出和逐层手写的参考实现一致（误差在 float32 舍入以内）。"""
    enc = _encoder()
    grids = [(4, 8), (8, 4)]
    torch.manual_seed(1)
    pixels = torch.randn(sum(h * w for h, w in grids), 3 * _CONFIG.patch_size**2)

    # 参考：编码器自己的 patch 嵌入、位置编码、窗口重排都照用，只把各层换成手写的（用合并之前的权重）。
    unfused = copy.deepcopy(enc)
    enc.fuse_projections()
    seen = {}
    first = enc.layers[0].register_forward_hook(lambda m, a, o: seen.update(x=a[0], cos=a[3], sin=a[4]))
    last = enc.layers[-1].register_forward_hook(lambda m, a, o: seen.update(out=o))
    enc(pixels, grids)
    first.remove()
    last.remove()

    reference = _reference_layers(unfused, seen["x"], grids, seen["cos"], seen["sin"])

    assert not hasattr(enc.layers[0], "q_proj")
    torch.testing.assert_close(seen["out"], reference, atol=1e-4, rtol=1e-4)


def test_compiled_qkv_rotary_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """场景：``OXR_COMPILE=1`` 时「拆 q/k/v + 旋转」按线上的编译选项（``language.COMPILE_OPTIONS``）编出来。
    条件：bf16、真实的头数和头维度（16 头 × 72），token 数 64、100、37 各跑一次（每批的 token 数都不同）。
    期待：能编过（不抛 InductorError），结果和逐个操作算的差在 bf16 一个舍入单位以内。"""
    from oxr_runtime import language, vision

    monkeypatch.setattr(language, "_rotate", language.rotate_traceable)  # 编译时用纯 PyTorch 的旋转，同线上
    torch._dynamo.reset()
    qkv_rotary = torch.compile(vision._qkv_rotary, **language.COMPILE_OPTIONS)
    torch.manual_seed(4)
    heads, dim = 16, 72
    for n in (64, 100, 37):
        qkv = torch.randn(n, 3 * heads * dim).bfloat16()
        cos, sin = torch.randn(n, dim // 2), torch.randn(n, dim // 2)
        got, want = qkv_rotary(qkv, cos, sin, heads, dim), vision._qkv_rotary(qkv, cos, sin, heads, dim)
        for g, e in zip(got, want, strict=True):
            torch.testing.assert_close(g.float(), e.float(), atol=2e-2, rtol=2**-7)


def test_padded_head_gives_same_embeddings() -> None:
    """场景：视觉词表头的权重补到 64 的倍数行（这里 36 列补到 64 列）之后再过词表头。
    条件：float32，同一批视觉 token。
    期待：输出的视觉 embedding 和不补时一致（补的那几列只是让矩阵乘对齐，结果里不用）。CPU 的矩阵乘换了输出
    宽度会换分块，末位可能不同，所以这里比到 float32 的舍入误差以内；GPU（cuBLAS）上前 65532 列逐位相同是另外
    实测的。"""
    enc = _encoder()
    torch.manual_seed(5)
    x = torch.randn(10, _CONFIG.hidden_size * _CONFIG.merge_unit)
    before = enc._head(x)

    enc.fuse_projections()

    assert enc.head_proj.weight.shape[0] % 64 == 0
    torch.testing.assert_close(enc._head(x), before, atol=1e-6, rtol=1e-5)
