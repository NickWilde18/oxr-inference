"""语言模型合并投影、合并逐元素操作之后，算出来的仍是 Qwen3 的结果。

这些用例在 CPU 上用 float32 和参考实现比，flash-attn 只收 GPU 上的 bf16/fp16：不管装没装 flash-attn，都先放一个
纯 PyTorch 的 ``oxr_runtime.attention`` 顶替，再导入语言模型。
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

_stub = types.ModuleType("oxr_runtime.attention")

def _attend(q, k, v, scale, causal):
    """q ``[n, H, D]``，k、v ``[m, KVH, D]``；GQA 把 KV 头重复到 H 个。causal 时第 i 个 q 对齐到倒数位置。"""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("nhd,mhd->hnm", q.float(), k.float()) * (scale or q.shape[-1] ** -0.5)
    if causal:
        n, m = q.shape[0], k.shape[0]
        mask = torch.arange(m)[None, :] > (torch.arange(n)[:, None] + m - n)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.einsum("hnm,mhd->nhd", scores.softmax(-1), v.float()).to(q.dtype)

def _varlen(q, k, v, cu_seqlens, max_len, *, causal, scale=None):
    bounds = cu_seqlens.tolist()
    return torch.cat([_attend(q[a:b], k[a:b], v[a:b], scale, causal) for a, b in zip(bounds, bounds[1:])])

def _paged_decode(q, k_cache, v_cache, seq_lens, block_table, *, scale=None):
    out = []
    for i, length in enumerate(seq_lens.tolist()):
        k = k_cache[block_table[i]].flatten(0, 1)[:length]
        v = v_cache[block_table[i]].flatten(0, 1)[:length]
        out.append(_attend(q[i : i + 1], k, v, scale, causal=True))
    return torch.cat(out)

def _rotary(x, cos, sin):
    """``[1, n, heads, dim]``，前后两半旋转，float32 下算。"""
    half = cos.shape[-1]
    xf = x.float()
    c, s = cos.float()[None, :, None, :], sin.float()[None, :, None, :]
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).to(x.dtype)

def _paged_extend(q, k_cache, v_cache, seq_lens, block_table, *, scale=None):
    """q ``[B, S, H, D]``：第 b 个序列的 S 个新 token 排在它长度 seq_lens[b] 的末尾，因果遮挡。"""
    out = []
    for b, length in enumerate(seq_lens.tolist()):
        k = k_cache[block_table[b]].flatten(0, 1)[:length]
        v = v_cache[block_table[b]].flatten(0, 1)[:length]
        out.append(_attend(q[b], k, v, scale, causal=True))
    return torch.stack(out)

def _varlen_bidirectional(q, k, v, cu_seqlens, max_len):
    return _varlen(q, k, v, cu_seqlens, max_len, causal=False)

_stub.varlen, _stub.paged_decode, _stub.rotary = _varlen, _paged_decode, _rotary
_stub.varlen_bidirectional = _varlen_bidirectional
_stub.paged_extend = _paged_extend
sys.modules["oxr_runtime.attention"] = _stub

from oxr_runtime import language  # noqa: E402
from oxr_runtime.config import LanguageConfig  # noqa: E402
from oxr_runtime.kv_cache import BLOCK_SIZE, KVCache  # noqa: E402

_CONFIG = LanguageConfig(
    hidden_size=64, intermediate_size=96, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=16,
    rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=50,
)


def _model(dtype: torch.dtype) -> language.LanguageModel:
    torch.manual_seed(0)
    model = language.LanguageModel(_CONFIG, max_positions=64)
    with torch.no_grad():
        for name, p in model.named_parameters():
            # 归一化的权重在 1 附近，其余是小的随机数，免得几层之后数值爆掉。
            p.copy_(1 + 0.1 * torch.randn_like(p) if "norm" in name else 0.2 * torch.randn_like(p))
    model = model.to(dtype).eval().requires_grad_(False)
    model.cast_rope(dtype)
    return model


def _reference(model: language.LanguageModel, embeds: torch.Tensor) -> torch.Tensor:
    """一个序列从头算到尾，float32，按 HF 的 Qwen3 写法逐层算；用合并之前的权重。返回最后一层归一化后的隐状态。"""
    c = _CONFIG
    n = embeds.shape[0]

    def norm(x, w):
        return w.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps))

    inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float) / c.head_dim))
    freqs = torch.outer(torch.arange(n, dtype=torch.float), inv_freq)
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]

    def rope(x):
        x1, x2 = x[..., : c.head_dim // 2], x[..., c.head_dim // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    x = embeds.float()
    for layer in model.layers:
        h = norm(x, layer.input_layernorm.weight)
        q = rope(norm((h @ layer.q_proj.weight.float().T).view(n, c.num_heads, c.head_dim), layer.q_norm.weight))
        k = rope(norm((h @ layer.k_proj.weight.float().T).view(n, c.num_kv_heads, c.head_dim), layer.k_norm.weight))
        v = (h @ layer.v_proj.weight.float().T).view(n, c.num_kv_heads, c.head_dim)
        rep = c.num_heads // c.num_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("nhd,mhd->hnm", q, k) * c.head_dim**-0.5
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
        attn = torch.einsum("hnm,mhd->nhd", scores.softmax(-1), v).reshape(n, -1)
        x = x + attn @ layer.o_proj.weight.float().T
        h = norm(x, layer.post_attention_layernorm.weight)
        mlp = F.silu(h @ layer.gate_proj.weight.float().T) * (h @ layer.up_proj.weight.float().T)
        x = x + mlp @ layer.down_proj.weight.float().T
    return norm(x, model.norm.weight)


def _run(model: language.LanguageModel, seqs: list[torch.Tensor], extra: list[torch.Tensor]):
    """几个序列一起 prefill，再每个序列各 decode 一个 token。返回（prefill 的隐状态，decode 的隐状态）。"""
    cache = KVCache(_CONFIG, num_blocks=len(seqs) + 1, dtype=seqs[0].dtype, device=torch.device("cpu"))
    blocks = [cache.allocate(1) for _ in seqs]
    lens = [s.shape[0] for s in seqs]

    positions = torch.cat([torch.arange(n) for n in lens])
    slots = torch.tensor([language.slot_of(b, p) for b, n in zip(blocks, lens) for p in range(n)])
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    hidden = model.prefill(language.PrefillBatch(torch.cat(seqs), positions, cu, max(lens), slots), cache)

    step = language.DecodeBatch(
        torch.stack(extra),
        torch.tensor(lens),
        torch.tensor([language.slot_of(b, n) for b, n in zip(blocks, lens)]),
        torch.tensor([n + 1 for n in lens], dtype=torch.int32),
        torch.tensor(blocks, dtype=torch.int32),
    )
    return hidden, model.decode(step, cache)


def test_fused_layers_match_reference_float32() -> None:
    """场景：两个序列（5 个、3 个 token）一起 prefill，再各 decode 一步。
    条件：合并 q/k/v、gate/up 投影，float32。
    期待：prefill 每个 token、decode 的新 token，隐状态都和逐层手写的 Qwen3 参考实现一致（误差在 float32 舍入以内）。"""
    model = _model(torch.float32)
    torch.manual_seed(1)
    seqs = [torch.randn(5, _CONFIG.hidden_size), torch.randn(3, _CONFIG.hidden_size)]
    extra = [torch.randn(_CONFIG.hidden_size), torch.randn(_CONFIG.hidden_size)]
    expected_prefill = torch.cat([_reference(model, s) for s in seqs])
    expected_decode = torch.stack([_reference(model, torch.cat([s, e[None]]))[-1] for s, e in zip(seqs, extra)])

    model.fuse_projections()
    prefill, decode = _run(model, seqs, extra)

    assert not hasattr(model.layers[0], "q_proj")
    torch.testing.assert_close(prefill, expected_prefill, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(decode, expected_decode, atol=1e-4, rtol=1e-4)


def test_fuse_projections_keeps_bf16_results() -> None:
    """场景：bf16 下同一批输入，合并投影前后各跑一遍。
    条件：合并前的模型直接用（逐个投影），合并后的用 qkv_proj、gate_up_proj。
    期待：两次的隐状态逐位相同。合并只是把几个矩阵首尾相接，每个输出元素的点积不变；CPU 上的矩阵乘按输出列
    独立算，所以连舍入都不变（GPU 上的 cuBLAS 可能换分块，那里要整批重测准确率，不在这里保证）。"""
    model = _model(torch.bfloat16)
    torch.manual_seed(2)
    seqs = [torch.randn(6, _CONFIG.hidden_size).bfloat16(), torch.randn(2, _CONFIG.hidden_size).bfloat16()]
    extra = [torch.randn(_CONFIG.hidden_size).bfloat16(), torch.randn(_CONFIG.hidden_size).bfloat16()]

    for layer in model.layers:  # 合并之前：用拆开的投影临时拼一个 qkv_proj / gate_up_proj 的等价物
        layer.qkv_proj = lambda h, l=layer: torch.cat((l.q_proj(h), l.k_proj(h), l.v_proj(h)), dim=-1)
        layer.gate_up_proj = lambda h, l=layer: torch.cat((l.gate_proj(h), l.up_proj(h)), dim=-1)
    before = _run(model, seqs, extra)
    for layer in model.layers:
        del layer.qkv_proj, layer.gate_up_proj

    model.fuse_projections()
    after = _run(model, seqs, extra)

    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_elementwise_matches_eager(dtype: torch.dtype) -> None:
    """场景：``OXR_COMPILE=1`` 时两个逐元素函数按线上的编译选项（``language.COMPILE_OPTIONS``）编出来。
    条件：随机输入，批大小 7 和 33 各跑一次（动态形状，不应因为批大小变化就算错）。
    期待：float32 下和逐个操作算的一致。bf16 下不要求逐位相同：inductor 把函数内部的中间结果留在 float32，
    不像逐个操作那样每步舍入到 bf16。这里要求编译版离「全程 float32 算」的结果不比逐个操作的远。"""
    c = _CONFIG
    funcs = {
        "add_rms_norm": language._add_rms_norm,
        "silu_mul": language._silu_mul,
    }
    compiled = {k: torch.compile(f, **language.COMPILE_OPTIONS) for k, f in funcs.items()}
    torch.manual_seed(3)
    nw = (1 + 0.1 * torch.randn(c.hidden_size)).to(dtype)
    for n in (7, 33):
        x, r = torch.randn(n, c.hidden_size, dtype=dtype), torch.randn(n, c.hidden_size, dtype=dtype)
        gate_up = torch.randn(n, 2 * c.intermediate_size, dtype=dtype)
        cases = {
            "add_rms_norm": (x, r, nw, c.rms_norm_eps),
            "silu_mul": (gate_up,),
        }
        for name, args in cases.items():
            eager, got = _tuple(funcs[name](*args)), _tuple(compiled[name](*args))
            if dtype == torch.float32:
                for w, g in zip(eager, got):
                    torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-5)
                continue
            exact = _tuple(funcs[name](*(a.float() if isinstance(a, torch.Tensor) else a for a in args)))
            for w, g, e in zip(eager, got, exact):
                eager_err = (w.float() - e).abs().max().item()
                compiled_err = (g.float() - e).abs().max().item()
                assert compiled_err <= eager_err * 1.01 + 1e-6, (name, n, compiled_err, eager_err)


def _tuple(x):
    return x if isinstance(x, tuple) else (x,)


def test_embed_places_visual_rows_and_indicators() -> None:
    """场景：一段输入里有普通 token、4 种指示符和两段视觉占位（共 5 个），视觉 embedding 有 5 行。
    条件：用不需要 CPU 等 GPU 的写法（torch.where + 前缀和）拼 embedding。
    期待：和按掩码逐个赋值（参考实现 merge_multimodal 的做法）的结果逐位相同：视觉占位依次取第 0–4 行，
    指示符取视觉 embedding 表最后 4 行，其余取语言模型的词表。"""
    from oxr_runtime.config import INDICATOR_IDS, VISUAL_ATOM_ID
    from oxr_runtime.model import OXRModel

    model = OXRModel.__new__(OXRModel)
    torch.nn.Module.__init__(model)
    model.language = _model(torch.float32)
    model.vision = types.SimpleNamespace(indicator_embeddings=torch.randn(4, _CONFIG.hidden_size))
    a = VISUAL_ATOM_ID
    ids = torch.tensor([3, INDICATOR_IDS[0], a, a, a, INDICATOR_IDS[1], 7, INDICATOR_IDS[2], a, a, INDICATOR_IDS[3], 9])
    visual = torch.randn(5, _CONFIG.hidden_size)

    got = model.embed(ids, visual)

    want = model.language.embed_tokens(ids.clamp(min=0)).clone()
    for i, indicator in enumerate(INDICATOR_IDS):
        want[ids == indicator] = model.vision.indicator_embeddings[i]
    want[ids == a] = visual
    assert torch.equal(got, want)


def test_extend_matches_full_sequence_float32() -> None:
    """场景：两个序列（各 8、6 个 token）先 prefill 前 5、3 个 token，再一次 extend 各 3 个新 token（投机解码验证那一步）。
    条件：float32，合并投影。
    期待：extend 出来的这 3 个位置的隐状态，和整段一次从头算（手写参考实现）的对应位置一致。"""
    model = _model(torch.float32)
    torch.manual_seed(6)
    full = [torch.randn(8, _CONFIG.hidden_size), torch.randn(6, _CONFIG.hidden_size)]
    expected = torch.cat([_reference(model, seq)[-3:] for seq in full])
    model.fuse_projections()

    cache = KVCache(_CONFIG, num_blocks=3, dtype=torch.float32, device=torch.device("cpu"))
    blocks = [cache.allocate(1) for _ in full]
    heads = [seq[:-3] for seq in full]
    lens = [h.shape[0] for h in heads]
    model.prefill(language.PrefillBatch(
        torch.cat(heads), torch.cat([torch.arange(n) for n in lens]),
        torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32), max(lens),
        torch.tensor([language.slot_of(b, p) for b, n in zip(blocks, lens) for p in range(n)]),
    ), cache)

    positions = [p for n in lens for p in range(n, n + 3)]
    batch = language.ExtendBatch(
        torch.cat([seq[-3:] for seq in full]), torch.tensor(positions),
        torch.tensor([language.slot_of(b, p) for b, n in zip(blocks, lens) for p in range(n, n + 3)]),
        torch.tensor([n + 3 for n in lens], dtype=torch.int32), torch.tensor(blocks, dtype=torch.int32), 3,
    )
    got = model.extend(batch, cache)

    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
