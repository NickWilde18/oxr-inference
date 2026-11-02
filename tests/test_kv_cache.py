"""KV 缓存的页表拼装。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from oxr_runtime.kv_cache import pack_page_table  # noqa: E402


def test_pack_page_table_rows_and_padding() -> None:
    """场景：三个序列，页数分别是 2、0、3，页表比序列多一行、比最长的多两列，里面原来有上一步的旧值。
    条件：用草稿页号 99 填空。
    期待：每行开头是这个序列的页号、顺序不变，其余格子（包括没有序列的那一行）全是 99，旧值不留。"""
    out = np.full((4, 5), -7, dtype=np.int32)

    pack_page_table(out, [[3, 8], [], [5, 1, 4]], fill=99)

    assert out.tolist() == [
        [3, 8, 99, 99, 99],
        [99, 99, 99, 99, 99],
        [5, 1, 4, 99, 99],
        [99, 99, 99, 99, 99],
    ]


def test_pack_page_table_matches_row_by_row() -> None:
    """场景：一千个序列、页数随机（0–6 页），和逐行写的结果比。
    期待：完全相同。"""
    rng = np.random.default_rng(0)
    tables = [rng.integers(0, 5000, size=rng.integers(0, 7)).tolist() for _ in range(1000)]
    expected = np.full((1024, 8), 7, dtype=np.int32)
    for i, pages in enumerate(tables):
        expected[i, : len(pages)] = pages

    out = np.zeros((1024, 8), dtype=np.int32)
    pack_page_table(out, tables, fill=7)

    assert np.array_equal(out, expected)
