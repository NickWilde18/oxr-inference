"""复读检测。"""

from __future__ import annotations

from oxr_runtime.repetition import repetition_cut


def test_cuts_after_first_cycle() -> None:
    """末尾是同一段 20 个 token 循环（实测复读页的样子）：截在第一遍循环之后。"""
    head = list(range(1000, 1300))
    cycle = list(range(20))
    output = head + cycle * 100  # 300 + 2000 = 2300
    output = output[: len(output) - len(output) % 64]  # 只在 64 的倍数时检查

    cut = repetition_cut(output, span=1024, max_period=128, every=64)

    assert cut is not None
    assert output[:cut][-20:] == output[cut - 20 : cut]
    # 截完之后不再有一整段 1024 的循环
    assert cut <= len(output) - 1024 + 20


def test_ordinary_text_is_not_cut() -> None:
    """正常的输出（不循环）、只循环了一小段（比如表格里几十个空单元格）都不截。"""
    varied = [i * 7919 % 50000 for i in range(2048)]
    short_loop = varied[:1500] + [5, 6, 7, 8] * 50 + varied[:348]

    assert repetition_cut(varied, span=1024, max_period=128, every=64) is None
    assert repetition_cut(short_loop[:2048], span=1024, max_period=128, every=64) is None


def test_disabled_and_off_schedule() -> None:
    """span 为 0 时关掉；长度不是检查间隔的倍数时不检查。"""
    loop = [1, 2] * 1024
    assert repetition_cut(loop, span=0, max_period=128, every=64) is None
    assert repetition_cut(loop[:-1], span=1024, max_period=128, every=64) is None
    assert repetition_cut(loop, span=1024, max_period=128, every=64) is not None
