"""OXR 整体：视觉编码器 + 语言模型，以及把视觉 token 拼进输入序列的那一步。"""

from __future__ import annotations

import torch
from torch import nn

from oxr_runtime.config import INDICATOR_IDS, VISUAL_ATOM_ID, ModelConfig
from oxr_runtime.language import LanguageModel
from oxr_runtime.vision import VisionEncoder


class OXRModel(nn.Module):
    def __init__(self, config: ModelConfig, max_positions: int):
        super().__init__()
        self.config = config
        self.vision = VisionEncoder(
            config.vision,
            visual_vocab_size=config.visual_vocab_size,
            indicator_count=len(INDICATOR_IDS),
            lm_hidden=config.language.hidden_size,
        )
        self.language = LanguageModel(config.language, max_positions)

    @torch.inference_mode()
    def embed(self, ids: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """输入序列的 embedding。``ids`` 里的负数是占位：:data:`VISUAL_ATOM_ID` 依次换成 ``visual`` 的各行，
        :data:`INDICATOR_IDS` 换成视觉 embedding 表最后 4 行。做法同参考实现的 ``merge_multimodal``。

        全程不让 CPU 等 GPU：不用布尔掩码赋值（``out[mask] = …`` 要先把掩码里有几个 True 拷回 CPU），改用
        ``torch.where``；第 k 个视觉占位取 ``visual`` 的第 k 行，k 由前缀和算出。取值都是原样拷贝，结果不变。"""
        out = self.language.embed_tokens(ids.clamp(min=0))
        indicators = self.vision.indicator_embeddings.to(out.dtype)

        for i, indicator in enumerate(INDICATOR_IDS):
            out = torch.where((ids == indicator).unsqueeze(-1), indicators[i], out)

        if visual.shape[0]:
            atoms = ids == VISUAL_ATOM_ID
            order = (atoms.cumsum(0) - 1).clamp(min=0)
            out = torch.where(atoms.unsqueeze(-1), visual.to(out.dtype)[order], out)

        return out
