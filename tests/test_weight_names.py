"""权重名映射：参考实现的每个权重都有去处，没有两个映射到同一个参数。"""

from __future__ import annotations

import json
import os

import pytest

from oxr_runtime.weight_names import map_name

_VIT = "visual_tokenizer.vit.vision_model."


def _reference_names(vision_layers: int = 27, language_layers: int = 28) -> list[str]:
    """按参考实现的模块结构列出全部权重名（和 model.safetensors 的头一致，752 个）。"""
    names = [
        _VIT + "embeddings.patch_embedding.weight", _VIT + "embeddings.patch_embedding.bias",
        _VIT + "embeddings.position_embedding.weight", _VIT + "post_layernorm.weight", _VIT + "post_layernorm.bias",
        "visual_tokenizer.head.0.weight", "visual_tokenizer.head.1.weight", "visual_tokenizer.head.1.bias",
        "vte.weight", "llm.model.embed_tokens.weight", "llm.model.norm.weight", "llm.lm_head.weight",
    ]
    for i in range(vision_layers):
        p = f"{_VIT}encoder.layers.{i}."
        for m in ("q_proj", "k_proj", "v_proj", "out_proj"):
            names += [f"{p}self_attn.{m}.weight", f"{p}self_attn.{m}.bias"]
        for m in ("fc1", "fc2"):
            names += [f"{p}mlp.{m}.weight", f"{p}mlp.{m}.bias"]
        for m in ("layer_norm1", "layer_norm2"):
            names += [f"{p}{m}.weight", f"{p}{m}.bias"]
    for i in range(language_layers):
        p = f"llm.model.layers.{i}."
        names += [f"{p}self_attn.{m}.weight" for m in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        names += [f"{p}mlp.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        names += [f"{p}input_layernorm.weight", f"{p}post_attention_layernorm.weight"]
    return names


def test_every_reference_weight_maps_to_one_parameter() -> None:
    names = _reference_names()
    assert len(names) == 752

    targets = [map_name(n) for n in names]
    used = [t for t in targets if t is not None]

    assert len(used) == len(set(used)), "有两个权重映射到同一个参数"
    assert sorted(n for n, t in zip(names, targets, strict=True) if t is None) == [
        _VIT + "post_layernorm.bias", _VIT + "post_layernorm.weight",
    ]
    assert map_name(_VIT + "encoder.layers.3.self_attn.out_proj.bias") == "vision.layers.3.out_proj.bias"
    assert map_name("llm.model.layers.27.mlp.down_proj.weight") == "language.layers.27.down_proj.weight"
    assert map_name("visual_tokenizer.head.1.bias") == "vision.head_norm.bias"


def test_unknown_weight_is_an_error() -> None:
    with pytest.raises(KeyError):
        map_name("llm.model.layers.0.self_attn.rotary_emb.inv_freq")


@pytest.mark.skipif(not os.environ.get("OXR_SAFETENSORS_HEADER"), reason="需要 model.safetensors 的头（JSON）")
def test_real_header_names_match_reference_list() -> None:
    """真实权重文件的头里的名字，和按结构列出的完全相同。``OXR_SAFETENSORS_HEADER`` 指向头的 JSON。"""
    header = json.load(open(os.environ["OXR_SAFETENSORS_HEADER"]))
    header.pop("__metadata__", None)
    assert sorted(header) == sorted(_reference_names())
