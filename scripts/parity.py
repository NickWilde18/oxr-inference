"""跑一遍运行时，和存下的 vLLM 输出逐块比对，顺带计时。

请求集是 OXR 流程发给识别模型的那些请求（渲染、版面分析都已做好），存成 pickle：一个列表，每条是一个 dict，有
``doc``、``page``、``kind``（块的类型，整页识别是 ``whole_page``）、``idx``、``img``（OpenCV 的 BGR 图）、
``prompt``、``max_tokens``。基线是同一请求集在 vLLM 上的结果，JSON：``{"results": [...]}``，每条有 ``doc``、
``page``、``kind``、``idx``、``ok``、``text``，顺序和请求集一致。本仓库不附带请求集和基线（测试文档有版权）。

用法::

    python scripts/parity.py --model /models/OXR-1.0 --reqs reqs.pkl --baseline vllm.json --mode blocks --out rt.json

比对时要关掉复读提前停（默认就是关的，``--repetition-span 0``），否则陷进复读的那几块必然不一致。

``--gpu-server`` 时不在本进程里起引擎，而是拉起一个 GPU 进程（:mod:`oxr_runtime.gpu_server`），和线上的流程进程一样：
图在本进程里缩放（``oxr_runtime.worker`` 的做法），经本机套接字发过去。
"""

from __future__ import annotations

import argparse
import difflib
import functools
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# torch 的编译 worker 池（inductor）默认起几十个子进程；进程因异常退出时它们会把父进程卡住，占着的显存也不放。
# ``OXR_COMPILE=1`` 时要编的只是语言模型和视觉编码器的几个小函数，在本进程里编就够了，关掉这个池。要在导入 torch 之前设。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import cv2
from PIL import Image

from oxr_runtime.engine import Completion, Engine, EngineSettings


@dataclass
class _Result:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class _InProcess:
    """在本进程里起引擎。"""

    def __init__(self, args: argparse.Namespace):
        settings = EngineSettings(
            kv_cache_gb=args.kv_gb, max_running=args.max_running, repetition_span=args.repetition_span,
            cuda_graphs=not args.no_cuda_graphs, speculative_tokens=args.speculative_tokens,
        )
        self.engine = Engine(args.model, settings)
        self.engine.start()

    def prepare(self, r: dict) -> Image.Image:
        # 先把图转好（BGR → RGB），预处理和计时分开：比对的是推理，不是 PNG 编解码。
        return Image.fromarray(cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB))

    def submit(self, image: Image.Image, r: dict, done) -> None:
        def finished(x: Completion | BaseException) -> None:
            done(x if isinstance(x, BaseException) else
                 _Result(x.text, x.finish_reason, x.prompt_tokens, len(x.token_ids), x.seconds))

        self.engine.submit(image, r["prompt"], max_tokens=r["max_tokens"], done=finished)

    def reset_stats(self) -> None:
        self.engine.profile = type(self.engine.profile)()
        self.engine.preemptions = 0
        self.engine.speculative_steps = self.engine.speculative_taken = 0

    def stats(self) -> dict:
        e = self.engine
        return {"preemptions": e.preemptions, "speculative": {"steps": e.speculative_steps, "tokens": e.speculative_taken},
                "profile": e.profile.summary()}

    def stop(self) -> None:
        self.engine.stop()


class _ThroughGpuServer:
    """拉起一个 GPU 进程，经本机套接字发请求；图在本进程里缩放，和线上的流程进程一样。"""

    def __init__(self, args: argparse.Namespace, max_waiting: int):
        from oxr_runtime import ipc, serve, worker

        self._dir = tempfile.mkdtemp(prefix="oxr-parity-")
        socket = os.path.join(self._dir, "gpu.sock")
        key = os.urandom(32).hex()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "oxr_runtime.gpu_server", "--model", args.model, "--oxr-src", args.oxr_src,
             "--socket", socket, "--kv-gb", str(args.kv_gb), "--max-running", str(args.max_running),
             "--max-waiting", str(max_waiting), "--repetition-span", str(args.repetition_span)],
            env=dict(os.environ, OXR_RUNTIME_IPC_KEY=key),
        )
        serve._wait_for_socket(socket, self.process, timeout=1800)
        self.client = ipc.Client(socket, authkey=bytes.fromhex(key))
        info, _ = self.client.call("info", {}).result(timeout=60)
        self._resize = functools.partial(worker._resize, patch_size=info["patch_size"],
                                         hidden_stride=info["hidden_stride"])

    def prepare(self, r: dict):
        return r["img"]  # 流程进程拿到的就是 OpenCV 的 BGR 图，缩放算在计时里（引擎内缩放时也算在里面）

    def submit(self, image, r: dict, done) -> None:
        resized = self._resize(image)
        start = time.monotonic()
        future = self.client.call("recognize", {"prompt": r["prompt"], "max_tokens": r["max_tokens"],
                                                "repetition_penalty": 1.0}, [resized])

        def finished(f) -> None:
            try:
                h, _ = f.result()
            except BaseException as exc:  # 识别失败也算一条结果，照常往下数
                done(exc)
                return
            done(_Result(h["text"], h["finish_reason"], h["prompt_tokens"], h["completion_tokens"],
                         time.monotonic() - start))

        future.add_done_callback(finished)

    def reset_stats(self) -> None:
        pass

    def stats(self) -> dict:
        return {}

    def stop(self) -> None:
        self.client.close()
        self.process.terminate()
        self.process.wait(timeout=60)
        shutil.rmtree(self._dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reqs", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--mode", choices=("blocks", "page"), default="blocks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kv-gb", type=float, default=16.0)
    parser.add_argument("--max-running", type=int, default=256)
    parser.add_argument("--repetition-span", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--submit-threads", type=int, default=4,
                        help="并行预处理、提交的线程数（和 serve 的 --preprocess-threads 默认值相同）")
    parser.add_argument("--speculative-tokens", type=int, default=EngineSettings.speculative_tokens,
                        help="投机解码每条序列最多猜几个 token，0 关掉")
    parser.add_argument("--repeat", type=int, default=1,
                        help="模型只加载一次，同一批请求连跑几遍，每遍单独计时（比对和输出文件用最后一遍）")
    parser.add_argument("--gpu-server", action="store_true",
                        help="经 GPU 进程跑（oxr_runtime.gpu_server），不在本进程里起引擎；要同时给 --oxr-src")
    parser.add_argument("--oxr-src", help="OXR 官方仓库的检出（--gpu-server 时 GPU 进程里加载版面检测要用）")
    args = parser.parse_args()
    if args.gpu_server and not args.oxr_src:
        parser.error("--gpu-server 要同时给 --oxr-src")
    if args.gpu_server and (args.no_cuda_graphs or args.speculative_tokens != EngineSettings.speculative_tokens):
        parser.error("--gpu-server 时用 GPU 进程的默认设置，不能改 --no-cuda-graphs、--speculative-tokens")

    reqs = [r for r in pickle.load(open(args.reqs, "rb")) if (r["kind"] == "whole_page") == (args.mode == "page")]
    if args.limit:
        reqs = reqs[: args.limit]

    backend = _ThroughGpuServer(args, max(2048, len(reqs))) if args.gpu_server else _InProcess(args)
    images = [backend.prepare(r) for r in reqs]

    def run_once() -> tuple[list[_Result | BaseException | None], float, float]:
        results: list[_Result | BaseException | None] = [None] * len(reqs)
        remaining = threading.Semaphore(0)

        def done_for(i: int):
            def done(result: _Result | BaseException) -> None:
                results[i] = result
                remaining.release()

            return done

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.submit_threads) as pool:
            for i, (r, image) in enumerate(zip(reqs, images, strict=True)):
                pool.submit(backend.submit, image, r, done_for(i))
        submitted = time.monotonic()
        for _ in reqs:
            remaining.acquire()
        return results, time.monotonic() - start, submitted - start

    walls = []
    for rep in range(args.repeat):
        backend.reset_stats()
        results, wall, submit_s = run_once()
        walls.append(wall)
        if args.repeat > 1:  # 每一遍的输出都留一份：批次组成不同，结果会有零星差别，准确率要逐遍看
            json.dump({"summary": {"rep": rep, "wall_s": round(wall, 2)}, "results": _rows(reqs, results)},
                      open(f"{args.out.removesuffix('.json')}.rep{rep}.json", "w"), ensure_ascii=False)
    stats = backend.stats()
    backend.stop()

    ok = [x for x in results if isinstance(x, _Result)]
    errors = [x for x in results if not isinstance(x, _Result)]
    pages = len({(r["doc"], r["page"]) for r in reqs})
    summary = {
        "mode": args.mode, "requests": len(reqs), "pages": pages, "errors": len(errors),
        "wall_s": round(wall, 2), "submit_s": round(submit_s, 2),
        "req_per_s_each": [round(len(reqs) / w, 1) for w in walls],
        "req_per_s": round(len(reqs) / wall, 1), "pages_per_s": round(pages / wall, 2),
        "gen_tok_per_s": round(sum(x.completion_tokens for x in ok) / wall),
        "prompt_tok_per_s": round(sum(x.prompt_tokens for x in ok) / wall),
        "finish": dict(Counter(x.finish_reason for x in ok)),
        **stats,
    }
    if errors:
        summary["first_error"] = repr(errors[0])[:300]

    if args.baseline:
        base = json.load(open(args.baseline))["results"]
        base = [b for b in base if (b["kind"] == "whole_page") == (args.mode == "page")][: len(reqs)]
        same, ratios = 0, []
        for r, b, x in zip(reqs, base, results, strict=True):
            assert (r["doc"], r["page"], r["kind"], r["idx"]) == (b["doc"], b["page"], b["kind"], b["idx"]), "请求集和基线顺序不一致"
            if not isinstance(x, _Result) or not b["ok"]:
                continue
            if x.text == b["text"]:
                same += 1
            else:
                ratios.append(difflib.SequenceMatcher(None, x.text, b["text"]).ratio())
        compared = same + len(ratios)
        summary["identical"] = f"{same}/{compared}（{same / max(compared, 1):.1%}）"
        if ratios:
            ratios.sort()
            summary["mismatch_similarity_p50"] = round(ratios[len(ratios) // 2], 3)
            summary["mismatch_similarity_min"] = round(ratios[0], 3)

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    json.dump({"summary": summary, "results": _rows(reqs, results)}, open(args.out, "w"), ensure_ascii=False)


def _rows(reqs: list[dict], results: list) -> list[dict]:
    return [
        {k: r[k] for k in ("doc", "page", "kind", "idx")}
        | ({"text": x.text, "finish": x.finish_reason, "tokens": x.completion_tokens, "seconds": round(x.seconds, 3)}
           if isinstance(x, _Result) else {"error": repr(x)[:300]})
        for r, x in zip(reqs, results, strict=True)
    ]


if __name__ == "__main__":
    main()
