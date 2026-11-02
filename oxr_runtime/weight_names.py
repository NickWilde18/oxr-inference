"""参考实现的权重名 → 本仓库模块的参数名。

参考实现（``modeling_oxr.py``）的模块嵌套更深（注意力、MLP 各是一个子模块）；这里拆得更平，按下表改名。
"""

from __future__ import annotations

import re

_VIT = "visual_tokenizer.vit.vision_model."
_LLM = "llm.model."

# (参考实现里的名字的正则, 这里的名字)。按顺序匹配，第一个命中的生效。
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(re.escape(_VIT) + r"embeddings\.patch_embedding\.(weight|bias)$"), r"vision.patch_embedding.\1"),
    (re.compile(re.escape(_VIT) + r"embeddings\.position_embedding\.weight$"), r"vision.position_embedding.weight"),
    (re.compile(re.escape(_VIT) + r"encoder\.layers\.(\d+)\.self_attn\.(q|k|v|out)_proj\.(weight|bias)$"),
     r"vision.layers.\1.\2_proj.\3"),
    (re.compile(re.escape(_VIT) + r"encoder\.layers\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$"), r"vision.layers.\1.\2.\3"),
    (re.compile(re.escape(_VIT) + r"encoder\.layers\.(\d+)\.(layer_norm1|layer_norm2)\.(weight|bias)$"),
     r"vision.layers.\1.\2.\3"),
    (re.compile(r"visual_tokenizer\.head\.0\.weight$"), r"vision.head_proj.weight"),
    (re.compile(r"visual_tokenizer\.head\.1\.(weight|bias)$"), r"vision.head_norm.\1"),
    (re.compile(r"vte\.weight$"), r"vision.vte.weight"),
    (re.compile(re.escape(_LLM) + r"embed_tokens\.weight$"), r"language.embed_tokens.weight"),
    (re.compile(re.escape(_LLM) + r"norm\.weight$"), r"language.norm.weight"),
    (re.compile(r"llm\.lm_head\.weight$"), r"language.lm_head.weight"),
    (re.compile(re.escape(_LLM) + r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$"), r"language.layers.\1.\2_proj.weight"),
    (re.compile(re.escape(_LLM) + r"layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$"), r"language.layers.\1.\2.weight"),
    (re.compile(re.escape(_LLM) + r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$"), r"language.layers.\1.\2_proj.weight"),
    (re.compile(re.escape(_LLM) + r"layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$"),
     r"language.layers.\1.\2.weight"),
]

# 参考实现里有、推理用不到的权重：视觉编码器最后的 LayerNorm（参考实现取的是它之前的输出）。
_UNUSED = {_VIT + "post_layernorm.weight", _VIT + "post_layernorm.bias"}


def map_name(name: str) -> str | None:
    """参考实现的权重名 → 这里的参数名；推理用不到的返回 None。名字不认识时报错。"""
    if name in _UNUSED:
        return None
    for pattern, target in _RULES:
        if pattern.match(name):
            return pattern.sub(target, name)
    raise KeyError(f"不认识的权重：{name}")
