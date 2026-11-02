"""不用草稿模型的投机解码：从一条序列自己已经写出的内容里找重复片段来猜接下来的几个 token。

OCR 的长输出主要是表格（HTML 的 ``<tr><td>…</td>`` 一行行重复）：评估集里最长的一条（1607 个 token 的表格）
按「最近一次出现的末尾 n 个 token（n 从 4 退到 1）后面接的是什么」来猜，每次最多猜 8 个，平均一次验证能前进
2.87 个 token；普通文字几乎猜不中（约 1.07 个）。贪心解码下，验证时模型自己给出的 token 和猜的逐个比对，
猜中的前缀直接收下，再加上验证那一步本来就会给出的一个，所以结果和一个一个生成的相同（验证用的 kernel 和逐字
生成不同，近似平局的地方末位舍入可能让结果不同，见 :mod:`oxr_runtime.engine`）。
"""

from __future__ import annotations

_MAX_N = 4


class NgramProposer:
    """一条序列的 n 元组索引：末尾 n 个 token（n = 1–4）→ 它们上一次出现之后的那个位置。"""

    def __init__(self, tokens: list[int]):
        self._index: dict[tuple[int, ...], int] = {}
        for position in range(1, len(tokens)):
            self._add(tokens, position)

    def _add(self, tokens: list[int], position: int) -> None:
        """``tokens[position]`` 已经有了：以它前面的 n 个 token 结尾的 n 元组，下次就从 ``position`` 接着猜。"""
        for n in range(1, min(_MAX_N, position) + 1):
            self._index[tuple(tokens[position - n : position])] = position

    def extend(self, tokens: list[int], old_length: int) -> None:
        """``tokens`` 从 ``old_length`` 往后是新追加的，补进索引。"""
        for position in range(max(old_length, 1), len(tokens)):
            self._add(tokens, position)

    def propose(self, tokens: list[int], k: int) -> list[int]:
        """按最长的能匹配上的末尾 n 元组，猜接下来最多 ``k`` 个 token；没匹配上返回空列表。"""
        for n in range(min(_MAX_N, len(tokens)), 0, -1):
            start = self._index.get(tuple(tokens[-n:]))
            if start is not None:
                return tokens[start : start + k]
        return []


def accept(draft: list[int], predicted: list[int]) -> list[int]:
    """``predicted[j]`` 是模型在「上一个 token + draft[:j]」之后选出的 token（``len(predicted) == len(draft) + 1``）。
    返回这一步收下的 token：draft 里和模型选的一致的最长前缀，再加上模型在那之后选的一个。"""
    accepted = 0
    while accepted < len(draft) and draft[accepted] == predicted[accepted]:
        accepted += 1
    return draft[:accepted] + [predicted[accepted]]
