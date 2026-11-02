"""逐字生成录好的图要带着回放那条流的高优先级跑：普通优先级流上排着一串 flash-attn 4 注意力（同视觉编码）时，
回放一步不能排到它们后面。

torch 2.11 实例化图时加了 ``cudaGraphInstantiateFlagUseNodePriority``：图里的 kernel 按录制时那条流的优先级调度，
回放时放进哪条流不改变优先级（torch 2.8 按回放的流）。图要是录在普通优先级的流上，和视觉编码同时跑时每个小 kernel 都要等前面的注意力 kernel，
H200 上实测一步从 7 ms 变成 100 多 ms。背景负载换成大矩阵乘时两种录法都不会排到后面，所以这里用注意力。

需要 GPU 和 flash-attn 4，没有就跳过。``oxr_runtime.decode_graph`` 在用例里面才导入：它会连带导入语言模型和
注意力模块，收集阶段导入会抢在 ``test_language_fused`` 放注意力替身之前。这里的语言模型是个只有几十层小矩阵乘的
替身，用不到注意力。
"""

from __future__ import annotations

import statistics
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("要 GPU", allow_module_level=True)
fa4 = pytest.importorskip("flash_attn.cute").flash_attn_varlen_func

from oxr_runtime.config import LanguageConfig  # noqa: E402
from oxr_runtime.kv_cache import KVCache  # noqa: E402

_CONFIG = LanguageConfig(
    hidden_size=64, intermediate_size=96, num_layers=1, num_heads=4, num_kv_heads=2, head_dim=16,
    rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=512,
)


class _FakeLanguage(torch.nn.Module):
    """只有 DecodeGraphs 用到的几个接口；一步是 40 层小矩阵乘加 ReLU，录下来的图有八十多个小 kernel。"""

    def __init__(self, hidden: int = 256, vocab: int = 512, layers: int = 40) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in range(layers))
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def decode(self, batch, cache):
        h = batch.embeds
        for layer in self.layers:
            h = torch.relu(layer(h))
        return h

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


def test_replay_keeps_stream_priority_under_load() -> None:
    """场景：普通优先级流上排着 40 次 flash-attn 4 双向注意力（128 段 × 1024 token、16 头、72 维，同视觉编码），
    逐字生成的图在高优先级流上回放一步、等 token 回来。
    条件：DecodeGraphs 用这条高优先级流构造，回放也在这条流上。
    期待：这一步比单独回放最多多等 5 次注意力的时间；排到它们后面时，八十多个小 kernel 每个都要等，远超这个数。"""
    from oxr_runtime.decode_graph import DecodeGraphs

    high, low = torch.cuda.Stream(priority=-1), torch.cuda.Stream()
    language = _FakeLanguage().cuda().bfloat16().eval()
    cache = KVCache(_CONFIG, num_blocks=4, dtype=torch.bfloat16, device=torch.device("cuda"))
    with torch.inference_mode():
        graphs = DecodeGraphs(language, cache, max_batch=4, max_positions=256, stream=high)

        def replay() -> float:
            started = time.perf_counter()
            with torch.cuda.stream(high):
                _, tokens = graphs.run([1, 2], [0, 0], [0, 1], [1, 1], [[0], [0]])
                tokens.cpu()
            return time.perf_counter() - started

        for _ in range(5):
            replay()
        solo = statistics.median(replay() for _ in range(10))

        q = torch.randn(128 * 1024, 16, 72, device="cuda", dtype=torch.bfloat16)
        cu = torch.arange(0, 128 * 1024 + 1, 1024, device="cuda", dtype=torch.int32)

        def attend() -> None:
            fa4(q, q, q, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=1024, max_seqlen_k=1024)

        with torch.cuda.stream(low):
            attend()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.cuda.stream(low):
            attend()
        torch.cuda.synchronize()
        one = time.perf_counter() - started

        with torch.cuda.stream(low):
            for _ in range(40):
                attend()
        contended = replay()
        torch.cuda.synchronize()

    assert contended < solo + 5 * one, (
        f"注意力排着时回放一步 {contended * 1e3:.1f} ms；单独回放 {solo * 1e3:.2f} ms，一次注意力 {one * 1e3:.2f} ms"
    )
