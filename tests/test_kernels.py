"""手写 Triton kernel（:mod:`oxr_runtime.kernels`）和同一套算式的 PyTorch 写法对照。

CPU 上没有 GPU：用 Triton 的解释器模式（``TRITON_INTERPRET=1``，numpy 逐条执行 kernel）跑 kernel 本身。这个
环境变量要在进程一开始就设（kernel 里调用的另一个 ``@triton.jit`` 函数，中途改环境变量再重新加载模块时解释器
认不出来），所以对照放在子进程里跑。

解释器把 float32 转 bf16 做成直接截断（GPU 上是就近舍入、平局取偶；Triton 3.4 实测，4096 个数和截断的结果全部
相同），误差上限是就近舍入的两倍。所以这里只能查逻辑（写哪一格、取哪一半、用哪个权重、旋转的符号）；有 GPU 时
另一个用例在 GPU 上逐位对照。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

_CHECK = textwrap.dedent('''
    import torch
    from oxr_runtime import kernels

    H, KV, D, n = 4, 2, 128, 5
    torch.manual_seed(0)
    qkv = torch.randn(n, (H + 2 * KV) * D).bfloat16()
    qw, kw = (1 + 0.1 * torch.randn(D)).bfloat16(), (1 + 0.1 * torch.randn(D)).bfloat16()
    angle = 10 * torch.rand(n, D // 2)
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    slots = torch.tensor([7, 0, 11, 3, 5])
    args = (qkv, qw, kw, cos, sin, 1e-6)

    def exact(x, w):  # float64、中间不舍入的真值
        x = x.double()
        y = w.double() * x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        c, s = cos.double()[:, None, :], sin.double()[:, None, :]
        return torch.cat((y[..., : D // 2] * c - y[..., D // 2 :] * s, y[..., : D // 2] * s + y[..., D // 2 :] * c), -1)

    heads = qkv.view(n, H + 2 * KV, D)
    truth = (exact(heads[:, :H], qw), exact(heads[:, H : H + KV], kw), heads[:, H + KV :].double())
    for need_kv in (True, False):
        caches = [torch.full((3, 4, KV, D), 9.0).bfloat16() for _ in range(4)]
        got = kernels._launch(*args, caches[0], caches[1], slots, H, KV, D, need_kv=need_kv)
        want = kernels.qk_norm_rope_store_reference(*args, caches[2], caches[3], slots, H, KV, D, need_kv=need_kv)
        outputs = [(got[0], want[0], truth[0])]
        if need_kv:
            outputs += [(got[1], want[1], truth[1]), (got[2], want[2], truth[2])]
        else:
            assert got[1] is None and got[2] is None
        flat = (-1, KV, D)
        outputs += [(caches[0].view(flat)[slots], caches[2].view(flat)[slots], truth[1]),
                    (caches[1].view(flat)[slots], caches[3].view(flat)[slots], truth[2])]
        for g, w, t in outputs:
            kernel_err = (g.double() - t).abs().max().item()
            reference_err = (w.double() - t).abs().max().item()
            assert kernel_err <= 3 * reference_err + 1e-3, (need_kv, kernel_err, reference_err)
        untouched = torch.ones(12, dtype=torch.bool)
        untouched[slots] = False
        for g in caches[:2]:
            assert (g.view(12, KV, D)[untouched] == 9.0).all(), "没写到的格子被改了"
    print("OK")
''')


def test_qk_norm_rope_store_matches_reference() -> None:
    """场景：5 个 token、4 个 q 头、2 个 KV 头、头维度 128（kernel 的平方和加法顺序按每半 64 个数写死），KV 缓存 3 页、
    每页 4 格，token 写到打乱的格子里。
    条件：bf16，cos/sin 取自真实的角度；Triton kernel（解释器模式）和 PyTorch 写法各算一遍，再算一份 float64、
    中间不舍入的真值。prefill（要连续的 k、v）和 decode（不要）各一遍。
    期待：q、k、v 和缓存里写到的格子，kernel 离真值的误差不超过 PyTorch 写法的 3 倍（解释器转 bf16 是截断，
    误差上限翻倍；逻辑错了会差出几十倍）；没写到的格子原样不动。"""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONPATH=str(root))
    proc = subprocess.run([sys.executable, "-c", _CHECK], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), proc.stdout + proc.stderr


def test_qk_norm_rope_store_bitwise_on_gpu() -> None:
    """场景：4,096 个 token、16 个 q 头、8 个 KV 头、头维度 128（同语言模型），各个头的数值量级不同，token 写到打乱的格子里。
    条件：GPU 上 Triton kernel 和 PyTorch 写法（平方和按同样的顺序两两相加）各算一遍；prefill（要连续的 k、v）和
    decode（不要）各一遍。
    期待：q、k、v 和写进缓存的值逐位相同。平方和的加法顺序一变，实测每一百万个输出就有几个末位不同。"""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("要 GPU")
    from oxr_runtime import kernels

    H, KV, D, n = 16, 8, 128, 4096
    torch.manual_seed(0)
    scale = torch.exp(torch.randn(1, H + 2 * KV, 1, device="cuda"))
    qkv = (torch.randn(n, H + 2 * KV, D, device="cuda") * scale).bfloat16().reshape(n, -1).contiguous()
    qw = (1 + 0.3 * torch.randn(D, device="cuda")).bfloat16()
    kw = (1 + 0.3 * torch.randn(D, device="cuda")).bfloat16()
    angle = 1000 * torch.rand(n, D // 2, device="cuda")
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    slots = torch.randperm(17 * 256, device="cuda")[:n]
    for need_kv in (True, False):
        results = []
        for fn in (kernels._launch, kernels.qk_norm_rope_store_reference):
            caches = [torch.zeros(17, 256, KV, D, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            out = fn(qkv, qw, kw, cos, sin, 1e-6, *caches, slots, H, KV, D, need_kv=need_kv)
            results.append([x for x in out if x is not None] + caches)
        for got, want in zip(*results):
            assert torch.equal(got, want), (need_kv, (got != want).sum().item())
