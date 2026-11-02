"""版面检测（OXR 的 D-FINE）在 GPU 进程里跑，不另起 HTTP 服务。

OXR 官方部署里，版面检测是单独的服务（OXR 源码里的 ``services/layout_analyze``），OXR 的流程把整页图编成 PNG、经
HTTP 发过去。两个进程各用各的显存，谁也不知道对方要多少：实测显存快满时，版面检测碰上没见过的批大小要临时多要
显存，申请失败（``CUBLAS_STATUS_ALLOC_FAILED``、``CUDNN_STATUS_INTERNAL_ERROR``），整份文档报错。这里它和识别
引擎在同一个进程里，显存在启动时一次占好、一起核对。

结果和官方服务逐位相同：

- 模型、权重、CUDA graph、推理直接用官方服务的那一套（``dfine_la.runtime.ModelRuntime``），只是不走 HTTP；
- 官方服务把收到的 PNG 解码成 BGR 再转 RGB。PNG 是无损的，这里在流程进程里直接把同一张 BGR 整页图转 RGB，交给官方
  的预处理（``dfine_la.preprocess.preprocess_images``：缩放补边、转 float），省掉编码和解码（见 :func:`prepare`）；
- 推理结果按官方服务的格式整理（``format_predictions``；OXR 不带置信度参数，不过滤），OXR 的后处理照常解析（见
  :func:`to_payload`）。

显存：1 页的批走官方的 CUDA graph，2 页以上的批官方是直接算的。启动时把 2 到 ``max_batch`` 页的批都先算一遍，显存
一次占好（单独实测：只算过 1 页时 1.2 GB，2–8 页都算过 4.25 GB；见 :meth:`LayoutModel.reserve_peak`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MAX_BATCH = 8
"""和 OXR 版面检测的默认批大小（``OXR__LAYOUT_ANALYZE_MODEL__BATCH_SIZE``）相同。"""


def add_source(oxr_src: Path) -> None:
    """让官方版面检测服务的包（``dfine_la``）能导入。"""
    root = str(Path(oxr_src) / "services" / "layout_analyze")
    if root not in sys.path:
        sys.path.insert(0, root)


class LayoutModel:
    """GPU 进程里的版面检测。:meth:`infer` 同一时刻只能一个线程调用（GPU 进程里由一个专用线程调用）。"""

    def __init__(self, oxr_src: Path, model_dir: Path, *, max_batch: int = MAX_BATCH):
        import torch

        add_source(oxr_src)
        from dfine_la.config import Settings
        from dfine_la.runtime import ModelRuntime

        weights = Path(model_dir) / "layout_analysis" / "last.pth"
        self._torch = torch
        self.runtime = ModelRuntime(Settings(model_path=weights, metadata_path=weights.with_name("metadata.json")))
        self.max_batch = max_batch
        self.image_size = self.runtime.settings.image_size
        self.num_classes = self.runtime.settings.num_classes
        self._stream = torch.cuda.Stream()
        self.reserve_peak()

    def reserve_peak(self) -> None:
        """2 到 ``max_batch`` 页的批各算一遍，让显存分配器把各种批大小要的显存占好。同一进程里之后再录 CUDA Graph
        （录图前 PyTorch 会清一次缓存）时，录完要再调一次。"""
        for batch in range(2, self.max_batch + 1):
            blank = np.full((batch, 3, self.image_size, self.image_size), 114 / 255, dtype=np.float32)
            self.infer(blank, np.full((batch, 2), self.image_size, dtype=np.int64))

    def infer(self, images: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``images`` 是 :func:`prepare` 的结果（float32 ``[页, 3, 边长, 边长]``），``sizes`` 是每页原始的（宽、高）。
        返回每页的类别、框、置信度（官方服务 ``ModelRuntime.infer`` 的输出）。"""
        if not 1 <= images.shape[0] <= self.max_batch:
            raise ValueError(f"一批 {images.shape[0]} 页，上限 {self.max_batch}")
        torch = self._torch
        with torch.cuda.stream(self._stream):
            outputs, _ = self.runtime.infer((torch.from_numpy(images), torch.from_numpy(sizes)))
        return outputs

    def info(self) -> dict:
        return {"num_classes": self.num_classes, "image_size": self.image_size, "max_batch": self.max_batch}


def prepare(images: list[np.ndarray], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """流程进程里：OXR 的整页图（BGR uint8）→ 官方服务解码之后那样的 RGB → 官方的预处理。返回 float32
    ``[页, 3, 边长, 边长]`` 和每页原始的（宽、高）。要先 :func:`add_source`。"""
    import cv2
    from dfine_la.preprocess import preprocess_images

    batch, sizes = preprocess_images([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images], image_size)
    return batch.numpy(), sizes.numpy()


def to_payload(labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, num_classes: int) -> dict:
    """推理结果整理成官方服务回给 OXR 的 JSON 结构（不带置信度参数时的那一种）。要先 :func:`add_source`。"""
    from dfine_la.confidence import format_predictions, parse_confidence_policy

    confidence, overrides, params = parse_confidence_policy(None, None, num_classes)
    return format_predictions((labels, boxes, scores), confidence, overrides, params)
