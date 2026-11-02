"""画 README 里的性能对比图（英文、中文各一套），输出到 docs/images/。

    python scripts/plot_performance.py

要 matplotlib；中文图用文泉驿正黑（WenQuanYi Zen Hei），没有就换成别的中文字体。数字是 README「性能」一节的实测值
（一张 H200，85 页，1,819 个识别请求，关闭缓存）。柱高是各遍的平均（只有范围的取中点），细线是各遍的最小到最大。
图里的文字转成路径，看图的人不用装字体。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

# 整条流程（页/秒）：按页 16 并发、按页 64 并发、3 份 PDF 整份同时发。官方每种发法一遍，本项目两遍的列两个数。
PIPELINE = {
    "official_default": [[1.44], [1.44], [1.24]],
    "official_tuned": [[2.19], [2.06], [1.95]],
    "ours": [[7.79, 7.78], [8.27], [5.16, 6.07]],
}
# 识别推理（请求/秒）：只记了范围。
RECOGNITION = {"vllm": (64, 82), "ours": (180, 190)}

COLORS = {"official_default": "#b4bac4", "official_tuned": "#6b7280", "vllm": "#6b7280", "ours": "#2563eb"}

TEXT = {
    "en": {
        "font": "DejaVu Sans",
        "groups": ["Per page,\n16 concurrent", "Per page,\n64 concurrent", "3 whole PDFs\nat once"],
        "series": {"official_default": "OXR official, default", "official_tuned": "OXR official, tuned",
                   "ours": "oxr-inference"},
        "pipeline_title": "Full pipeline on one H200",
        "pipeline_axis": "pages/s (higher is better)",
        "recognition_title": "Recognition inference, 1,819 requests at once",
        "recognition_axis": "requests/s (higher is better)",
        "rows": {"vllm": "vLLM 0.25.1 + OXR plugin\n(fastest config)", "ours": "oxr-inference"},
    },
    "zh": {
        "font": "WenQuanYi Zen Hei",
        "groups": ["按页发\n16 并发", "按页发\n64 并发", "3 份 PDF\n整份同时发"],
        "series": {"official_default": "OXR 官方部署，默认参数", "official_tuned": "OXR 官方部署，调参",
                   "ours": "本项目"},
        "pipeline_title": "整条流程，一张 H200",
        "pipeline_axis": "页/秒（越高越好）",
        "recognition_title": "识别推理，1,819 个请求一次发出",
        "recognition_axis": "请求/秒（越高越好）",
        "rows": {"vllm": "vLLM 0.25.1 + OXR 插件\n（最快配置）", "ours": "本项目"},
    },
}


def _label(values: list[float]) -> str:
    # 按十进制四舍五入到一位小数：1.95 要显示成 2.0（直接 f"{1.95:.1f}" 是 1.9，和表里的数对不上）
    lo, hi = (Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) for v in (min(values), max(values)))
    return f"{lo}" if lo == hi else f"{lo}–{hi}"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _pipeline(t: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    for i, key in enumerate(PIPELINE):
        for g, runs in enumerate(PIPELINE[key]):
            x = g + (i - 1) * width
            mean = sum(runs) / len(runs)
            ax.bar(x, mean, width * 0.92, color=COLORS[key], label=t["series"][key] if g == 0 else None, zorder=2)
            if len(runs) > 1:
                ax.vlines(x, min(runs), max(runs), color="#111827", linewidth=1.2, zorder=3)
            ax.text(x, max(runs) + 0.12, _label(runs), ha="center", va="bottom", fontsize=10,
                    color="#111827", fontweight="bold" if key == "ours" else "normal")
    ax.set_xticks(range(3), t["groups"], fontsize=10.5)
    ax.set_ylabel(t["pipeline_axis"], fontsize=10.5)
    ax.set_ylim(0, 9.5)
    ax.yaxis.grid(True, color="#e5e7eb", zorder=0)
    ax.set_title(t["pipeline_title"], fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def _recognition(t: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for y, key in enumerate(("ours", "vllm")):
        lo, hi = RECOGNITION[key]
        ax.barh(y, (lo + hi) / 2, 0.55, color=COLORS[key], zorder=2)
        ax.hlines(y, lo, hi, color="#111827", linewidth=1.2, zorder=3)
        ax.text(hi + 3, y, f"{lo}–{hi}", va="center", fontsize=10.5, color="#111827",
                fontweight="bold" if key == "ours" else "normal")
    ax.set_yticks(range(2), [t["rows"]["ours"], t["rows"]["vllm"]], fontsize=10.5)
    ax.set_xlabel(t["recognition_axis"], fontsize=10.5)
    ax.set_xlim(0, 215)
    ax.xaxis.grid(True, color="#e5e7eb", zorder=0)
    ax.set_title(t["recognition_title"], fontsize=13, loc="left", pad=10)
    _style(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "oxr-inference"  # 每次生成的 SVG 相同，不在 git 里白白产生差异
    for lang, t in TEXT.items():
        plt.rcParams["font.family"] = [t["font"]]
        for name, draw in (("pipeline-throughput", _pipeline), ("recognition-throughput", _recognition)):
            fig = draw(t)
            fig.savefig(OUT / f"{name}-{lang}.svg", facecolor="white", metadata={"Date": None})
            plt.close(fig)


if __name__ == "__main__":
    main()
