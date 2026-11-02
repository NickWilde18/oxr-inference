# oxr-inference：OXR 的 HTTP 接口（/v1/parse/sync 等），识别模型的推理换成本仓库的运行时
# （PyTorch + flash-attn 4 + 几个 Triton kernel）。
#
# 镜像里有三样东西：
# - OXR 官方源码（固定提交）：用它的接口、流程和版面检测模型；
# - OXR-1.0 权重（约 2.9 GB，含版面检测 D-FINE），构建时下载并逐个校验 sha256；
# - 本仓库的运行时：一个 GPU 进程（oxr_runtime.gpu_server）跑识别和版面检测，几个流程进程（oxr_runtime.worker）
#   跑 OXR 的接口和流程。
#
# 版本：torch 2.11.0 + CUDA 13.0，和 OXR 版面检测（D-FINE）说明文档写的版本相同，版面检测的结果和官方服务逐页相同。
# 视觉编码器和语言模型的注意力都用 flash-attn 4.0.0b37（见 oxr_runtime/attention.py；flash-attn 2 没有 torch 2.11
# 的预编译包），其余依赖见 docker/requirements.lock，其中只有 flash-attn 4 是预发布版（它还没有正式版）。
#
# 驱动：CUDA 13.x 要 580 以上的驱动。NVIDIA 官方发布说明：CUDA 13.0 正式版配套的 Linux x86_64 驱动是
# ≥ 580.65.06；按 CUDA 小版本兼容，13.x 的程序在 ≥ 580 的驱动上能跑。
#
# 许可证：本仓库和 OXR 都是 Apache 2.0 加 OXR 附加条款。镜像里各部分的许可证原文都在 /licenses，说明见
# THIRD_PARTY_NOTICES.md；其中 NVIDIA CUTLASS DSL（flash-attn 4 依赖）是 NVIDIA 的专有许可，所以不公开发布镜像。

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ── 下载源码和权重 ─────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS fetch
ARG OXR_COMMIT=c1a1dcb
# 权重的下载地址；不给时用 docker/fetch_weights.py 的默认（ModelScope）。逐个校验 sha256，换地址不会换掉内容。
ARG WEIGHTS_BASE_URL=
WORKDIR /fetch
COPY docker/fetch_weights.py docker/weights_manifest.json ./
RUN python fetch_weights.py --manifest weights_manifest.json --out /models/OXR-1.0 \
        ${WEIGHTS_BASE_URL:+--base-url "$WEIGHTS_BASE_URL"}
# 基础镜像没有 git：按固定提交下载源码包。codeload 的包根目录是 OXR-<完整提交号>。
RUN python -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
        "https://codeload.github.com/SoMarkAI/OXR/tar.gz/${OXR_COMMIT}" oxr.tar.gz \
    && mkdir src && tar -xzf oxr.tar.gz -C src && mv src/OXR-* /opt-oxr

# ── 运行镜像 ───────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE}
# gcc：Triton 第一次编 kernel 时要编一个 C 写的启动器（手写的 qk_norm_rope_store 和 OXR_COMPILE=1 编出来的
# kernel 都要）；
# libgl1、libglib2.0-0：OXR 依赖的 opencv-python（非 headless 版）要用。
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir -r /tmp/requirements.lock

COPY --from=fetch /opt-oxr /opt/oxr
COPY --from=fetch /models/OXR-1.0 /models/OXR-1.0
RUN mkdir /licenses \
    && cp /opt/oxr/LICENSE /licenses/OXR-LICENSE \
    && cp /opt/oxr/NOTICE /licenses/OXR-NOTICE \
    && cp /opt/oxr/services/layout_analyze/THIRD_PARTY_NOTICES.md /licenses/layout-analysis-THIRD_PARTY_NOTICES.md \
    && cp /models/OXR-1.0/LICENSE /licenses/OXR-1.0-weights-LICENSE
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /licenses/oxr-inference/
COPY docker/list_licenses.py /tmp/list_licenses.py
RUN python /tmp/list_licenses.py > /licenses/python-packages.md && rm /tmp/list_licenses.py

COPY oxr_runtime /opt/runtime/oxr_runtime
ENV PYTHONPATH=/opt/runtime \
    OXR_ATTENTION=fa4 \
    OXR_COMPILE=1 \
    TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
    PYTHONUNBUFFERED=1
# 默认跑法：语言模型的逐元素操作用 torch.compile 编译，并按逐步舍入算（和不编译时、和 vLLM 的
# 舍入点一致）。升级 torch 时要复测 TORCHINDUCTOR_EMULATE_PRECISION_CASTS 这个开关还在、结果还对。

EXPOSE 8000
# KV 缓存大小、同时在跑的上限按部署的显存改（H200 上实测用的是 KV 80 GiB、上限 1,024）。
ENTRYPOINT ["python", "-m", "oxr_runtime.serve", "--model", "/models/OXR-1.0", "--oxr-src", "/opt/oxr", "--port", "8000"]
CMD ["--kv-gb", "16", "--max-running", "256"]
