"""本运行时和 vLLM 输出不一样的块：把 vLLM 的输出当成既定答案喂给本运行时（teacher forcing），找到第一个
本运行时会选别的 token 的位置，看两个候选的 logit 差多少。差得很小 = 数值上的接近平局；差得大 = 本运行时
在这一步就明确偏向别的内容。

请求集、基线的格式见 ``scripts/parity.py``；``--ours`` 是 parity.py 的输出，``--vllm-runs`` 是 vLLM 另外几轮的结果。

用法：python scripts/divergence.py --model … --reqs reqs.pkl --baseline vllm.json --ours rt.json \\
        --vllm-runs vllm2.json,vllm3.json,… --out divergence.json
"""
from __future__ import annotations

import argparse, json, os, pickle
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import cv2, numpy as np, torch
from PIL import Image

from oxr_runtime.engine import Engine, EngineSettings
from oxr_runtime.language import PrefillBatch, slot_of
from oxr_runtime.kv_cache import blocks_for
from oxr_runtime.preprocess import to_patches


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--model", "--reqs", "--baseline", "--ours", "--vllm-runs", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()

    reqs = [r for r in pickle.load(open(a.reqs, "rb")) if r["kind"] != "whole_page"]
    base = [r for r in json.load(open(a.baseline))["results"] if r["kind"] != "whole_page"]
    ours = json.load(open(a.ours))["results"]
    others = [[r for r in json.load(open(p))["results"] if r["kind"] != "whole_page"] for p in a.vllm_runs.split(",")]

    picked = [i for i, (o, b) in enumerate(zip(ours, base))
              if o.get("text") != b["text"] and o.get("text") not in {x[i]["text"] for x in others}]

    eng = Engine(a.model, EngineSettings(kv_cache_gb=8, cuda_graphs=False))
    m, tok = eng.model, eng.prompts.tokenizer
    out = []
    with torch.inference_mode():
        for i in picked:
            r = reqs[i]
            img = Image.fromarray(cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB))
            patches = to_patches(img)
            prompt = eng.prompts.build(r["prompt"], patches.num_tokens)
            target = tok.encode(base[i]["text"], add_special_tokens=False).ids
            ids = prompt + target
            blocks = eng.cache.allocate(blocks_for(len(ids)))
            try:
                dev = eng.device
                visual = m.vision(torch.from_numpy(patches.pixels).to(dev), [(patches.grid_h, patches.grid_w)])
                t = torch.tensor(ids, device=dev)
                batch = PrefillBatch(
                    m.embed(t, visual), torch.arange(len(ids), device=dev),
                    torch.tensor([0, len(ids)], device=dev, dtype=torch.int32), len(ids),
                    torch.tensor([slot_of(blocks, p) for p in range(len(ids))], device=dev, dtype=torch.long),
                )
                hidden = m.language.prefill(batch, eng.cache)
                logits = m.language.logits(hidden[len(prompt) - 1 : len(ids) - 1]).float()
            finally:
                eng.cache.release(blocks)
            pick = logits.argmax(-1).tolist()
            k = next((j for j, (p, g) in enumerate(zip(pick, target)) if p != g), None)
            row = {"i": i, "doc": r["doc"], "page": r["page"], "kind": r["kind"], "idx": r["idx"],
                   "target_tokens": len(target)}
            if k is None:
                row["result"] = "本运行时贪心也会给出 vLLM 的输出（逐 token 一致）"
            else:
                lg = logits[k]
                top = torch.topk(lg, 2)
                row |= {"diverge_at": k, "ours_token": tok.decode([pick[k]]), "vllm_token": tok.decode([target[k]]),
                        "gap": round(float(lg[pick[k]] - lg[target[k]]), 4),
                        "top2_gap": round(float(top.values[0] - top.values[1]), 4),
                        "context": tok.decode(target[max(0, k - 12):k])}
            out.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
