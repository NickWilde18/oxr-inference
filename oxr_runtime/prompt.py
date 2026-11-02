"""一次识别请求的输入序列：一条 user 消息，先图后文字。

和 OXR 官方客户端发给 vLLM 的完全一样（``oxr/model/client.py``：``[image_url, text]``），chat 模板是权重目录的
``chat_template.jinja``：

    <|im_start|>user\\n<image>{提示词}<|im_end|>\\n<|im_start|>assistant\\n

``<image>`` 换成「图片开始 + N 个视觉 token + 图片结束」三种占位（见 :mod:`oxr_runtime.config`），两边的文字
分开分词，做法同参考实现的 ``_tokenize_with_visual_placeholder``。
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from oxr_runtime.config import INDICATOR_IDS, VISUAL_ATOM_ID

_BEFORE_IMAGE = "<|im_start|>user\n"
_AFTER_PROMPT = "<|im_end|>\n<|im_start|>assistant\n"


class PromptBuilder:
    def __init__(self, model_dir: str | Path):
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self._prefix = self._encode(_BEFORE_IMAGE)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def build(self, prompt: str, visual_tokens: int) -> list[int]:
        """一张图加一段提示词的输入 id，视觉部分是占位。"""
        image = [INDICATOR_IDS[0], *([VISUAL_ATOM_ID] * visual_tokens), INDICATOR_IDS[1]]
        return [*self._prefix, *image, *self._encode(prompt + _AFTER_PROMPT)]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
