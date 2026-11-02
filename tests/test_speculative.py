"""投机解码的猜测和验收（纯 Python，不碰模型）。"""

from __future__ import annotations

from oxr_runtime.speculative import NgramProposer, accept


def test_proposes_continuation_of_latest_match() -> None:
    """场景：表格那样一段结构重复的输出：<td> 1 </td> <td> 2 </td> <td>。
    条件：末尾是 "</td> <td>"，它上一次出现后面接的是 "2 </td>"。
    期待：按最长的末尾 n 元组、最近一次出现来猜，猜出 "2 </td> <td>"（最多 3 个）。"""
    td, end, one, two = 10, 11, 1, 2
    tokens = [td, one, end, td, two, end, td]
    proposer = NgramProposer(tokens)

    assert proposer.propose(tokens, 3) == [two, end, td]


def test_no_match_gives_empty_draft() -> None:
    """场景：末尾的 token 前面从没出现过。期待：不猜（空列表），调用方照常一个一个生成。"""
    tokens = [1, 2, 3, 4]
    assert NgramProposer(tokens).propose(tokens, 8) == []


def test_extend_matches_rebuilding_from_scratch() -> None:
    """场景：先对前一半建索引，再把后一半追加进去。期待：和对整段一次建好的索引猜出的一样。"""
    tokens = [5, 6, 7, 5, 6, 8, 5, 6, 7, 9, 5, 6]
    incremental = NgramProposer(tokens[:5])
    incremental.extend(tokens, 5)

    for k in (1, 2, 4):
        assert incremental.propose(tokens, k) == NgramProposer(tokens).propose(tokens, k)


def test_accept_takes_matching_prefix_plus_one() -> None:
    """场景：猜了 4 个，模型在第 3 个上选得不一样。
    期待：收下前 2 个猜对的，再加上模型在第 3 个位置选的那个；全猜对时多收一个模型给的；一个都不对时只收模型的。"""
    assert accept([7, 8, 9, 10], [7, 8, 3, 0, 0]) == [7, 8, 3]
    assert accept([7, 8], [7, 8, 5]) == [7, 8, 5]
    assert accept([7, 8], [1, 8, 5]) == [1]
    assert accept([], [4]) == [4]
