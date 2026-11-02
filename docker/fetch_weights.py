"""构建镜像时下载 OXR-1.0 权重，按 ``weights_manifest.json`` 逐个校验大小和 sha256，对不上就失败。

清单来自模型仓库的元数据（下载、校验通过的那 24 个文件）。默认从 ModelScope 下载（实测有的网络里
HF 的大文件 CDN 只有 0.05 MiB/s）。构建机能直连 HF 时，用 ``--base-url`` 换成 HF 的 resolve 地址。
大文件分 8 段 Range 并行下载（单连接速度波动很大，实测同一文件 0.6–39.5 MiB/s），每段最多重试 5 次。

用法：python fetch_weights.py --manifest weights_manifest.json --out /models/OXR-1.0 [--base-url …]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

DEFAULT_BASE = "https://modelscope.cn/models/SoMark/OXR-1.0/resolve/master/"
_PARTS = 8
_LARGE = 256 << 20


def _download_range(url: str, dst: str, lo: int, hi: int) -> None:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if os.path.getsize(dst) == hi - lo + 1:
                return
        except OSError as exc:
            print(f"重试 {os.path.basename(dst)}（第 {attempt + 1} 次）：{type(exc).__name__}", flush=True)
    raise RuntimeError(f"{dst} 下载失败")


def fetch(url: str, dst: str, size: int) -> str:
    """下载到 ``dst``，返回 sha256。"""
    h = hashlib.sha256()
    if size >= _LARGE:
        step = -(-size // _PARTS)
        ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(_PARTS)]
        with cf.ThreadPoolExecutor(_PARTS) as pool:
            list(pool.map(lambda r: _download_range(url, f"{dst}.part{r[0]}", r[1], r[2]), ranges))
        with open(dst, "wb") as out:
            for i, _, _ in ranges:
                with open(f"{dst}.part{i}", "rb") as part:
                    while chunk := part.read(1 << 20):
                        out.write(chunk)
                        h.update(chunk)
                os.remove(f"{dst}.part{i}")
        return h.hexdigest()

    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    for path, want in sorted(manifest.items()):
        dst = os.path.join(args.out, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        started = time.monotonic()
        digest = fetch(args.base_url + path, dst, want["size"])
        size = os.path.getsize(dst)
        if size != want["size"] or digest != want["sha256"]:
            print(f"校验失败 {path}：期望 {want['size']} {want['sha256'][:12]}，实际 {size} {digest[:12]}", flush=True)
            sys.exit(3)
        print(f"OK {path} {size} {size / 2**20 / max(time.monotonic() - started, 1e-3):.1f} MiB/s", flush=True)
    print(f"权重校验通过：{len(manifest)} 个文件", flush=True)


if __name__ == "__main__":
    main()
