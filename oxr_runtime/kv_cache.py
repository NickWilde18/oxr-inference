"""语言模型的 KV 缓存：按页（block）分配，每页 256 个 token。

OXR 的语言模型只有 0.6B，KV 很小（每个 token 约 112 KB），但一页 OCR 的输出长短差别很大：逐块识别多数几十个
token，整页识别中位数约 900、陷进复读时到 8192 上限。按最长预留会浪费上百 GB，所以按页分配，用完就还。

页大小 256 是 flash-attn 分页接口的要求（``page_block_size`` 必须是 256 的倍数）。
"""

from __future__ import annotations

import itertools
import threading

import numpy as np
import torch

from oxr_runtime.config import LanguageConfig

BLOCK_SIZE = 256


class KVCache:
    """所有层共用一张页表：同一个序列在每一层占的页号相同。要页、还页可以从多个线程调用（引擎的 prefill 线程准入时
    要页，逐字生成线程扩页、还页）。"""

    def __init__(self, config: LanguageConfig, num_blocks: int, *, dtype: torch.dtype, device: torch.device):
        shape = (config.num_layers, num_blocks, BLOCK_SIZE, config.num_kv_heads, config.head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))
        self._lock = threading.Lock()

    @staticmethod
    def bytes_per_block(config: LanguageConfig, dtype: torch.dtype) -> int:
        element = torch.empty((), dtype=dtype).element_size()
        return 2 * config.num_layers * BLOCK_SIZE * config.num_kv_heads * config.head_dim * element

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int] | None:
        """要 ``count`` 页；不够时返回 None，一页都不拿。"""
        with self._lock:
            if count > len(self._free):
                return None
            return [self._free.pop() for _ in range(count)]

    def release(self, blocks: list[int]) -> None:
        with self._lock:
            self._free.extend(reversed(blocks))


def blocks_for(tokens: int) -> int:
    """放下 ``tokens`` 个 token 要几页。"""
    return (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def pack_page_table(out: np.ndarray, tables: list[list[int]], fill: int) -> None:
    """把各序列的页号写进页表 ``out``（``[行数, 列数]``，行数不少于序列数）：第 i 行前面是第 i 个序列的页号，
    其余格子（包括多出来的行）都填 ``fill``。

    一次算出所有页号的行、列再整块写，不按序列逐行写：一千个序列时逐行写要几毫秒，整块写不到 0.1 ms。
    """
    n = len(tables)
    lengths = np.fromiter(map(len, tables), dtype=np.int64, count=n)
    pages = np.fromiter(itertools.chain.from_iterable(tables), dtype=out.dtype, count=int(lengths.sum()))
    rows = np.repeat(np.arange(n), lengths)
    cols = np.arange(pages.size) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    out.fill(fill)
    out[rows, cols] = pages
