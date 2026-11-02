"""模型结构参数，从权重目录的 config.json 读出。

只取推理要用的那几项；config.json 里别的字段（训练用的、HF 自动映射用的）不管。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    """视觉编码器（SigLIP2-NaViT）。"""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    patch_size: int
    hidden_stride: int
    """相邻 hidden_stride × hidden_stride 个 patch 合成一个视觉 token。"""
    window_size: int
    layer_norm_eps: float
    position_grid: int
    """原始位置编码表是 position_grid × position_grid 的网格，按每张图的网格尺寸插值。"""

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def merge_unit(self) -> int:
        return self.hidden_stride * self.hidden_stride


@dataclass(frozen=True)
class LanguageConfig:
    """语言模型（Qwen3 结构）。"""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int


@dataclass(frozen=True)
class ModelConfig:
    vision: VisionConfig
    language: LanguageConfig
    visual_vocab_size: int
    """视觉词表大小，最后 4 个是图片起止等指示符（见 :data:`INDICATOR_IDS`）。"""
    eos_token_ids: tuple[int, ...]

    @staticmethod
    def load(model_dir: str | Path) -> ModelConfig:
        root = Path(model_dir)
        raw = json.loads((root / "config.json").read_text())
        vit = raw["vit_config"]
        llm = raw["llm_config"]

        eos: list[int] = []
        generation = root / "generation_config.json"
        if generation.exists():
            value = json.loads(generation.read_text()).get("eos_token_id", [])
            eos = list(value) if isinstance(value, list) else [value]
        if not eos:
            eos = [raw["eos_token_id"]]

        image_size, patch_size = vit["image_size"], vit["patch_size"]

        return ModelConfig(
            vision=VisionConfig(
                hidden_size=vit["hidden_size"],
                intermediate_size=vit["intermediate_size"],
                num_layers=vit["num_hidden_layers"],
                num_heads=vit["num_attention_heads"],
                patch_size=patch_size,
                hidden_stride=vit["hidden_stride"],
                window_size=vit["window_size"],
                layer_norm_eps=vit["layer_norm_eps"],
                position_grid=image_size // patch_size,
            ),
            language=LanguageConfig(
                hidden_size=llm["hidden_size"],
                intermediate_size=llm["intermediate_size"],
                num_layers=llm["num_hidden_layers"],
                num_heads=llm["num_attention_heads"],
                num_kv_heads=llm["num_key_value_heads"],
                head_dim=llm["head_dim"],
                rms_norm_eps=llm["rms_norm_eps"],
                rope_theta=llm["rope_theta"],
                vocab_size=llm["vocab_size"],
            ),
            visual_vocab_size=raw["visual_vocab_size"],
            eos_token_ids=tuple(eos),
        )


# 输入序列里的占位 id：负数，不会和文本 token 撞。和参考实现（modeling_oxr.py）一致。
VISUAL_ATOM_ID = -300
"""一个视觉 token 的位置。"""
INDICATOR_IDS = (-301, -302, -303, -304)
"""图片开始、图片结束、视频开始、视频结束。它们的 embedding 是视觉词表最后 4 行。"""
