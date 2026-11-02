"""复读检测：输出末尾一段是不是同一段内容在循环。

实测整页识别有约 3.5% 的页（85 页里 3 页）陷进复读：末尾 2000 字里同一个 20 字片段重复上百次，一直写到
``max_tokens``。检测到就停在第一遍循环之后，后面的都是重复，没有信息。
"""

from __future__ import annotations

import numpy as np


def repetition_cut(output: list[int], *, span: int, max_period: int, every: int) -> int | None:
    """``output`` 末尾 ``span`` 个 token 是周期不超过 ``max_period`` 的循环时，返回该截在哪（保留第一遍循环）；
    不是就返回 None。只在长度是 ``every`` 的倍数时检查，省掉每步都比一遍。``span`` 为 0 时不检查。"""
    if not span or len(output) < span or len(output) % every:
        return None

    tail = np.asarray(output[-span:])
    for period in range(1, max_period + 1):
        if np.array_equal(tail[period:], tail[:-period]):
            return len(output) - span + period
    return None
