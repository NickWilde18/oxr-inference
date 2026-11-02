"""逐字生成那一步用 CUDA Graph 回放。

不用它时，一步要由 CPU 依次下发约 1,100 个小 GPU 操作（28 层、每层约 40 个），实测一步 36 ms，几乎都耗在
下发上。启动时按几档批大小把这一步各录一遍，之后每一步只回放一次。

录下来的是固定形状和固定显存地址，所以：

- 输入（这一步的 token、位置、写缓存的格号、序列长度、KV 页表）每一步先用 numpy 拼进几块锁页内存，再异步
  拷进几块固定的显存，然后回放。页表一次拼好整块，不按序列逐行建张量：一千个序列逐行建张量要几毫秒；
- 选下一个 token 的 argmax 也录在图里，每一步只把 token 号拷回 CPU；
- 实际的序列数凑不满这一档时，空位指向一页专门留出来的「草稿页」：写缓存写进草稿页，注意力只读草稿页的
  第一格，算出来的那几行丢掉，碰不到真实序列的缓存；
- 每档的输出（logits）一直留着引用，录完的显存不会被别的档复用掉；
- 图录在回放用的那条流（引擎里是逐字生成的高优先级流）上。torch 2.11 实例化图时加了
  ``cudaGraphInstantiateFlagUseNodePriority``（torch v2.11.0 的 ``aten/src/ATen/cuda/CUDAGraph.cpp:183``，v2.8.0
  没有），图里的 kernel 按录制那条流的优先级调度，回放时放进哪条流不起作用：录在普通优先级的流上时，和视觉编码
  同时跑一步要 100 多 ms，录在高优先级流上是 7 ms（H200 实测，批 128）。

结果和不用 CUDA Graph 时算的是同一组操作；批大小凑到档位上，矩阵乘的分块可能不同，末位偶尔会有差别。
"""

from __future__ import annotations

import numpy as np
import torch

from oxr_runtime.kv_cache import BLOCK_SIZE, KVCache, pack_page_table
from oxr_runtime.language import DecodeBatch, ExtendBatch, LanguageModel

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024)


class DecodeGraphs:
    """各档批大小的录制结果。:meth:`run` 选能装下的最小一档回放；超过最大一档返回 None，调用方照常算。"""

    def __init__(self, language: LanguageModel, cache: KVCache, *, max_batch: int, max_positions: int,
                 stream: torch.cuda.Stream):
        """``stream`` 是回放用的流，图就录在这条流上（见模块说明）。"""
        self.language = language
        self.cache = cache
        self.stream = stream
        device = language.lm_head.weight.device

        scratch = cache.allocate(1)
        if scratch is None:
            raise RuntimeError("KV 缓存连一页草稿页都分不出来")
        self.scratch_block = scratch[0]
        self.scratch_slot = self.scratch_block * BLOCK_SIZE

        self.buckets = [b for b in BUCKETS if b < max_batch] + [max_batch]
        self.width = (max_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
        top = self.buckets[-1]

        self.ids = torch.zeros(top, dtype=torch.long, device=device)
        self.positions = torch.zeros(top, dtype=torch.long, device=device)
        self.slots = torch.full((top,), self.scratch_slot, dtype=torch.long, device=device)
        self.seq_lens = torch.ones(top, dtype=torch.int32, device=device)
        self.table = torch.full((top, self.width), self.scratch_block, dtype=torch.int32, device=device)

        # 和上面一一对应的锁页内存。调用方在下一次 run 之前一定已经取走了这一次的 token（取的时候会同步），
        # 所以这一次的异步拷贝已经做完，下一次可以直接覆盖。
        self._staging = [(dst, torch.empty_like(dst, device="cpu").pin_memory())
                         for dst in (self.ids, self.positions, self.slots, self.seq_lens, self.table)]
        self._host = [host.numpy() for _, host in self._staging]

        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        pool = torch.cuda.graph_pool_handle()

        # 从大到小录：大的那档先占好显存池，小的复用池里的空间。
        for bucket in reversed(self.buckets):
            self._graphs[bucket] = self._capture(bucket, pool)

    def _forward(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch = DecodeBatch(
            self.language.embed_tokens(self.ids[:bucket]),
            self.positions[:bucket],
            self.slots[:bucket],
            self.seq_lens[:bucket],
            self.table[:bucket],
        )
        logits = self.language.logits(self.language.decode(batch, self.cache))
        return logits, logits.argmax(dim=-1)

    def _capture(self, bucket: int, pool: object) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
        # 先在回放用的流上跑两遍预热（triton 的编译、cuBLAS 选算法都在这时完成），录制时不能再有这些动作。
        _warm_up(self.stream, lambda: self._forward(bucket))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
            logits, tokens = self._forward(bucket)
        return graph, logits, tokens

    def run(
        self, ids: list[int], positions: list[int], slots: list[int], seq_lens: list[int], tables: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """回放一步，返回 ``[len(ids), vocab]`` 的 logits 和贪心选出的 ``[len(ids)]`` 个 token（都指向这一档的
        输出，下一次回放前要用完）。"""
        n = len(ids)
        bucket = next((b for b in self.buckets if b >= n), None)
        if bucket is None:
            return None

        h_ids, h_positions, h_slots, h_seq_lens, h_table = self._host
        h_ids[:n] = ids
        h_ids[n:bucket] = 0
        h_positions[:n] = positions
        h_positions[n:bucket] = 0
        h_slots[:n] = slots
        h_slots[n:bucket] = self.scratch_slot
        h_seq_lens[:n] = seq_lens
        h_seq_lens[n:bucket] = 1

        pack_page_table(h_table[:bucket], tables, self.scratch_block)

        for dst, host in self._staging:
            dst[:bucket].copy_(host[:bucket], non_blocking=True)

        graph, logits, tokens = self._graphs[bucket]
        graph.replay()
        return logits[:n], tokens[:n]


EXTEND_BUCKETS = (1, 2, 4, 8)


class ExtendGraphs:
    """投机解码验证那一步（每个序列固定 ``tokens_per_seq`` 个 token）用 CUDA Graph 回放，做法同 :class:`DecodeGraphs`：
    批大小凑到 1、2、4、8 几档；凑不满一档时，空位的 token 写进草稿页的前几格，注意力也只读这几格。不录图时这一步
    要从 Python 一个个发出 28 层的 kernel（torch.compile 编出的函数每调用一次光是检查 guard 就约 0.1 ms），实测一步
    13 ms，比它省下的逐字生成步数还贵。"""

    def __init__(self, language: LanguageModel, cache: KVCache, *, max_batch: int, tokens_per_seq: int,
                 max_positions: int, scratch_block: int, stream: torch.cuda.Stream):
        """``stream`` 是回放用的流，图就录在这条流上（同 :class:`DecodeGraphs`）。"""
        self.language = language
        self.cache = cache
        self.stream = stream
        self.tokens_per_seq = s = tokens_per_seq
        device = language.lm_head.weight.device
        self.buckets = [b for b in EXTEND_BUCKETS if b < max_batch] + [max_batch]
        self.width = (max_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.scratch_block = scratch_block
        top = self.buckets[-1]
        scratch_slots = torch.arange(s, device=device) + scratch_block * BLOCK_SIZE

        self.ids = torch.zeros(top * s, dtype=torch.long, device=device)
        self.positions = torch.arange(s, device=device).repeat(top)
        self.slots = scratch_slots.repeat(top)
        self.seq_lens = torch.full((top,), s, dtype=torch.int32, device=device)
        self.table = torch.full((top, self.width), scratch_block, dtype=torch.int32, device=device)
        self._scratch_slots = scratch_slots.cpu().numpy()

        self._staging = [(dst, torch.empty_like(dst, device="cpu").pin_memory())
                         for dst in (self.ids, self.positions, self.slots, self.seq_lens, self.table)]
        self._host = [host.numpy() for _, host in self._staging]

        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        pool = torch.cuda.graph_pool_handle()
        for bucket in reversed(self.buckets):
            self._graphs[bucket] = self._capture(bucket, pool)

    def _forward(self, bucket: int) -> torch.Tensor:
        s, n = self.tokens_per_seq, bucket * self.tokens_per_seq
        batch = ExtendBatch(
            self.language.embed_tokens(self.ids[:n]), self.positions[:n], self.slots[:n], self.seq_lens[:bucket],
            self.table[:bucket], s,
        )
        return self.language.logits(self.language.extend(batch, self.cache)).argmax(dim=-1).view(bucket, s)

    def _capture(self, bucket: int, pool: object) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        _warm_up(self.stream, lambda: self._forward(bucket))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
            tokens = self._forward(bucket)
        return graph, tokens

    def run(
        self, ids: list[int], positions: list[int], slots: list[int], seq_lens: list[int], tables: list[list[int]]
    ) -> torch.Tensor | None:
        """``ids``、``positions``、``slots`` 是 ``len(seq_lens) × tokens_per_seq`` 个，按序列排。回放一次，返回
        ``[序列数, tokens_per_seq]`` 的贪心 token（指向这一档的输出，下一次回放前要用完）；序列数超过最大一档返回 None。"""
        n = len(seq_lens)
        bucket = next((b for b in self.buckets if b >= n), None)
        if bucket is None:
            return None
        s = self.tokens_per_seq
        real, padded = n * s, bucket * s

        h_ids, h_positions, h_slots, h_seq_lens, h_table = self._host
        h_ids[:real] = ids
        h_ids[real:padded] = 0
        h_positions[:real] = positions
        h_positions[real:padded] = np.tile(np.arange(s), bucket - n)
        h_slots[:real] = slots
        h_slots[real:padded] = np.tile(self._scratch_slots, bucket - n)
        h_seq_lens[:n] = seq_lens
        h_seq_lens[n:bucket] = s
        pack_page_table(h_table[:bucket], tables, self.scratch_block)

        for dst, host in self._staging:
            length = padded if dst.shape[0] == self.ids.shape[0] else bucket
            dst[:length].copy_(host[:length], non_blocking=True)

        graph, tokens = self._graphs[bucket]
        graph.replay()
        return tokens[:n]


def _warm_up(stream: torch.cuda.Stream, forward) -> None:
    """在 ``stream`` 上把要录的那一步跑两遍，再让当前流等它做完。"""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
