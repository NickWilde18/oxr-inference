"""从权重目录的 ``model.safetensors`` 加载到 :class:`~oxr_runtime.model.OXRModel`。

权重名的对应关系见 :mod:`oxr_runtime.weight_names`。每个参数都必须恰好被赋值一次：少了或多了都报错，
不静默跳过。
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from oxr_runtime.model import OXRModel
from oxr_runtime.weight_names import map_name

def load_weights(model: OXRModel, model_dir: str | Path, device: torch.device) -> None:
    """把权重拷进 ``model`` 的参数。dtype 以 ``model`` 的参数为准。"""
    params = dict(model.named_parameters())
    assigned: set[str] = set()

    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt", device=str(device)) as f:
        for name in f.keys():  # noqa: SIM118 —— safe_open 不支持直接迭代
            target = map_name(name)
            if target is None:
                continue
            if target not in params:
                raise KeyError(f"权重 {name} 映射到 {target}，模型里没有这个参数")
            if target in assigned:
                raise KeyError(f"参数 {target} 被赋值了两次（{name}）")

            tensor = f.get_tensor(name)
            param = params[target]
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(f"{name} 的形状 {tuple(tensor.shape)} 和 {target} 的 {tuple(param.shape)} 不一致")

            with torch.no_grad():
                param.copy_(tensor.to(param.dtype))
            assigned.add(target)

    missing = sorted(set(params) - assigned)
    if missing:
        raise KeyError(f"这些参数没有对应的权重：{missing[:8]}{' …' if len(missing) > 8 else ''}")
