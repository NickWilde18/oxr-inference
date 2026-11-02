"""推理引擎：持续接新请求、批量 prefill、所有在跑的序列一起逐步 decode（连续批处理）。贪心解码（OXR 官方客户端
固定 temperature=0）。

两个线程：

- **prefill 线程**：收新请求，KV 缓存放得下就发出一批 prefill（先把这批请求的图一起过视觉编码器），发到 prefill
  的 CUDA 流上就返回；发下一批之前先等上一批在 GPU 上算完，同一时刻 GPU 上最多一批 prefill；
- **逐字生成线程**：上一批 prefill 算完（流上的事件完成）就把这批序列并进在跑的批，所有在跑的序列一步步各生成
  一个 token（另一条流，优先级更高）。序列结束、被挤出时把页还回去，通知 prefill 线程。

分成两个线程，是因为发一批 prefill 要从 Python 发出几百个 kernel（几十毫秒）：一个线程轮流做两件事时，这期间
逐字生成停着（实测在线按页跑时，单线程的引擎 42% 的时间在发 prefill，逐字生成每步只要 3.6 ms，每个 token 却要等
11.3 ms）。两个线程时 Python 解释器每隔一小段在两者之间切换，逐字生成不会被一次发起挡住几十毫秒。

GPU 上两边本来就是并行的：prefill 是吃算力的大矩阵乘，逐字生成的每一步都是一串很小的 kernel、吃不满 GPU，两者
叠在一起，逐字生成的时间大部分被 prefill 盖住。

两个线程共用的只有：KV 缓存的页（:class:`~oxr_runtime.kv_cache.KVCache` 自己加锁）、被挤出的请求（逐字生成线程
放进 ``_preempted``，prefill 线程并回队首）、发出了还没并进来的 prefill 批（``_landing``，带锁的计数
``_unlanded``）。排队列表只归 prefill 线程，在跑列表只归逐字生成线程。

两处和 vLLM 不同的行为：

- **复读提前停**：输出末尾有一段 ``repetition_span`` 个 token 是同一段内容循环（周期不超过
  ``repetition_max_period``），就停在第一遍循环之后，``finish_reason`` 记 ``repetition``。实测整页识别
  85 页里有 3 页陷进复读、一直写到 8192 上限，把整批的耗时拉长一倍；``repetition_span=0`` 关掉这一项，
  和 vLLM 逐位比对时要关掉。
- **投机解码**（不用草稿模型）：在跑的序列不超过 ``speculative_max_batch`` 条时（一批请求快收尾、只剩几条长
  输出的时候），每条序列从自己已经写出的内容里找重复片段猜接下来最多 ``speculative_tokens`` 个 token，一次前向
  验证，猜中的前缀直接收下（见 :mod:`oxr_runtime.speculative`）。表格这类结构重复的长输出一次能前进两三个
  token；验证那一步用的 kernel 和逐字生成不同，近似平局的地方结果可能不同。``speculative_tokens=0`` 关掉。
- **重复惩罚**只按 vLLM 的算法实现（出现过的 token：logit 为正除以惩罚系数、为负乘以它，提示词里的 token 也算），
  OXR 的流程只在一块识别坏了、重试时才调高它。
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from oxr_runtime.config import ModelConfig
from oxr_runtime.decode_graph import DecodeGraphs, ExtendGraphs
from oxr_runtime.kv_cache import BLOCK_SIZE, KVCache, blocks_for
from oxr_runtime.language import DecodeBatch, ExtendBatch, PrefillBatch, slot_of
from oxr_runtime.model import OXRModel
from oxr_runtime import preprocess
from oxr_runtime.prompt import PromptBuilder
from oxr_runtime.repetition import repetition_cut
from oxr_runtime.speculative import NgramProposer, accept
from oxr_runtime.weights import load_weights

log = logging.getLogger(__name__)

_PROFILE = os.environ.get("OXR_PROFILE") == "1"
"""开了就按阶段计时（每段结束时同步 GPU，会拖慢一点），结果在 :attr:`Engine.profile`。"""


class _Profile:
    """各阶段的累计耗时（秒）和次数，只在 ``OXR_PROFILE=1`` 时记。"""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.prefill_batches: list[tuple[int, int]] = []
        """每次 prefill 的（请求数，输入 token 数），用来看批凑得多大。"""

    def add(self, name: str, seconds: float, count: int = 1) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + count

    def mark(self, start: float, name: str, count: int = 1) -> float:
        """记下 ``start`` 到现在这一段（先等当前流上的 GPU 工作做完），返回现在的时间，给下一段当起点。"""
        if not _PROFILE:
            return start
        torch.cuda.current_stream().synchronize()
        now = time.perf_counter()
        self.add(name, now - start, count)
        return now

    def batch(self, requests: int, tokens: int) -> None:
        if _PROFILE:
            self.prefill_batches.append((requests, tokens))

    def summary(self) -> dict[str, dict[str, float]]:
        out = {k: {"s": round(v, 2), "n": self.counts[k]} for k, v in self.seconds.items()}
        if self.prefill_batches:
            requests, tokens = (np.array(column) for column in zip(*self.prefill_batches, strict=True))
            out["prefill_batches"] = {
                "n": len(self.prefill_batches),
                **{f"requests_p{q}": float(np.percentile(requests, q)) for q in (10, 50, 90)},
                **{f"tokens_p{q}": float(np.percentile(tokens, q)) for q in (10, 50, 90)},
            }
        return out


@dataclass(frozen=True)
class EngineSettings:
    kv_cache_gb: float = 16.0
    """KV 缓存占多少显存。每 256 个 token 一页、一页 28 MiB，16 GB 约 58 万个 token。"""
    max_running: int = 256
    """同时在跑的序列上限。"""
    max_prefill_tokens: int = 32768
    """一次 prefill 最多处理多少个输入 token（含视觉 token）。"""
    max_positions: int = 40960
    repetition_span: int = 1024
    repetition_max_period: int = 128
    repetition_check_every: int = 64
    cuda_graphs: bool = True
    """逐字生成那一步用 CUDA Graph 回放（见 :mod:`oxr_runtime.decode_graph`）。关掉时每一步照常逐个下发。"""
    speculative_tokens: int = 8
    """投机解码每条序列最多猜几个 token（见模块说明），0 关掉。"""
    speculative_max_batch: int = 8
    """在跑的序列不超过这么多条时才投机：批大时逐字生成的一步本来就吃满了 GPU，多验证几个 token 不划算。"""
    max_waiting: int = 2048
    """还没进 prefill 的请求最多排多少个，满了 :meth:`Engine.submit` 抛 :class:`Busy`。排队的请求只占内存（缩放好的
    图在锁页内存里，一个最多约 2.7 MB），进 prefill 时才上传显存。"""


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: list[int]
    finish_reason: str
    """``stop``（遇到结束符）、``length``（到 max_tokens）、``repetition``（复读提前停，见模块说明）。"""
    prompt_tokens: int
    seconds: float


class Busy(RuntimeError):
    """排队的请求已经有 ``max_waiting`` 个，这个请求没收。"""


class Cancelled(Exception):
    """请求在完成之前被取消了（调用方超时或断开）。"""


@dataclass
class _Request:
    prompt_ids: list[int]
    grid: tuple[int, int]
    """这张图的 patch 网格（行、列）。"""
    image: torch.Tensor
    """缩放好的图（uint8 ``[高, 宽, 3]``，锁页内存）。进 prefill 时才上传显存、切 patch。"""
    max_tokens: int
    repetition_penalty: float
    done: Callable[[Completion | BaseException], None]
    submitted: float = field(default_factory=time.monotonic)
    cancelled: bool = False


class Ticket:
    """:meth:`Engine.submit` 返回的凭据，用来取消这个请求。"""

    def __init__(self, request: _Request):
        self._request = request

    def cancel(self) -> None:
        """取消请求：还在排队的不再进来，在跑的下一步就停、页还回去，``done`` 收到 :class:`Cancelled`。
        已经完成的不受影响。可以从任何线程调用。"""
        self._request.cancelled = True


@dataclass
class _Sequence:
    request: _Request
    blocks: list[int]
    length: int
    """已经写进缓存的 token 数。"""
    output: list[int] = field(default_factory=list)
    seen: torch.Tensor | None = None
    """重复惩罚用：出现过的 token。只在惩罚系数不是 1 时有。"""
    proposer: NgramProposer | None = None
    """投机解码用的 n 元组索引，第一次投机时才建。"""
    indexed: int = 0
    """``proposer`` 已经收录到 ``output`` 的第几个 token。"""


@dataclass
class _InFlight:
    """已经发出、还没算完的一批 prefill。"""

    sequences: list[_Sequence]
    logits: torch.Tensor
    tokens: torch.Tensor
    """按 ``logits`` 贪心选好的第一个 token。"""
    done: torch.cuda.Event


class Engine:
    def __init__(self, model_dir: str | Path, settings: EngineSettings | None = None, device: str = "cuda"):
        self.settings = settings or EngineSettings()
        self.device = torch.device(device)
        # torch 默认按看得见的核数开 CPU 计算线程（比如 96 核的机器），而容器的 CPU 配额只有 16 核。引擎在 CPU 上
        # 只拼些小张量，一旦某个张量过了并行的门槛，几十个线程抢 16 核的配额，整个进程被限流：实测批 256 的一步
        # 拼页表从 1 ms 变成 38 ms。引擎线程和预处理线程各自单线程算就够了。
        torch.set_num_threads(1)
        self.config = ModelConfig.load(model_dir)
        self.prompts = PromptBuilder(model_dir)
        self._eos = set(self.config.eos_token_ids)

        dtype = torch.bfloat16
        with torch.device(self.device):
            model = OXRModel(self.config, self.settings.max_positions).to(dtype)
        load_weights(model, model_dir, self.device)
        model.language.cast_rope(dtype)
        model.language.fuse_projections()
        model.vision.fuse_projections()
        self.model = model.eval().requires_grad_(False)

        num_blocks = int(self.settings.kv_cache_gb * 2**30 // KVCache.bytes_per_block(self.config.language, dtype))
        self.cache = KVCache(self.config.language, num_blocks, dtype=dtype, device=self.device)
        log.info("KV 缓存 %d 页（%d 个 token）", num_blocks, num_blocks * BLOCK_SIZE)

        # 逐字生成的流优先级更高：两边的 kernel 同时排队时先调度逐字生成的，长输出的尾巴不被 prefill 拖住。
        # 逐字生成和投机验证的图都录在这条流上，图才带着它的优先级（见 decode_graph 的模块说明）。
        self._decode_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self._prefill_stream = torch.cuda.Stream(device=self.device)

        self._graphs: DecodeGraphs | None = None
        if self.settings.cuda_graphs:
            started = time.perf_counter()
            with torch.inference_mode():
                self._graphs = DecodeGraphs(
                    self.model.language, self.cache,
                    max_batch=self.settings.max_running, max_positions=self.settings.max_positions,
                    stream=self._decode_stream,
                )
            log.info("CUDA Graph 录好 %d 档，用了 %.1f 秒", len(self._graphs.buckets), time.perf_counter() - started)

        self._extend_graphs: ExtendGraphs | None = None
        if self.settings.cuda_graphs and self.settings.speculative_tokens > 0:
            with torch.inference_mode():
                self._extend_graphs = ExtendGraphs(
                    self.model.language, self.cache, max_batch=self.settings.speculative_max_batch,
                    tokens_per_seq=1 + self.settings.speculative_tokens,
                    max_positions=self.settings.max_positions, scratch_block=self._graphs.scratch_block,
                    stream=self._decode_stream,
                )

        self._landing: collections.deque[_InFlight] = collections.deque()
        """prefill 线程发出了、逐字生成线程还没并进来的批，按发出的顺序。"""
        self._unlanded = 0
        """``_landing`` 里的序列数（准入时和在跑的一起算上限），改它要拿 ``_landing_lock``。"""
        self._landing_lock = threading.Lock()
        self._preempted: collections.deque[_Request] = collections.deque()
        """逐字生成线程挤出去、要放回队首重算的请求，按挤出的先后。"""
        self._live = 0
        """在跑的序列数，逐字生成线程每轮结束时更新（一步当中在跑列表会暂时清空，prefill 线程准入时读这个数）。"""
        self._launched = threading.Event()
        """prefill 线程发出一批时置位，叫醒没活干的逐字生成线程。"""
        self._released = threading.Event()
        """逐字生成线程还页时置位，叫醒因为页不够卡住的 prefill 线程。"""

        self._normalize_table = torch.from_numpy(preprocess.normalize_table()).to(self.device, torch.bfloat16)

        self.profile = _Profile()
        self.preemptions = 0
        """KV 缓存不够、被挤出去重算的次数（累计）。"""
        self.speculative_steps = 0
        """做了几次投机验证（累计）。"""
        self.speculative_taken = 0
        """投机验证一共收下了多少个 token（累计，含每次验证本来就会给出的那一个）。"""
        self._inbox: queue.Queue[_Request] = queue.Queue()
        self._waiting: list[_Request] = []
        self._running: list[_Sequence] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        started = time.perf_counter()
        self._warm_up()
        log.info("prefill 预热用了 %.1f 秒", time.perf_counter() - started)

    def _resize(self, image: Image.Image) -> np.ndarray:
        """CPU 上的那一半预处理：按 :func:`~oxr_runtime.preprocess.smart_resize` 定的尺寸缩放（PIL，和参考实现逐位
        一致）。查表归一化、切 patch 在进 prefill 时于 GPU 上做（:func:`~oxr_runtime.preprocess.patchify`，结果和全在
        CPU 上算逐位相同）。"""
        c = self.config.vision
        return preprocess.resize(image, patch_size=c.patch_size, hidden_stride=c.hidden_stride)

    def _warm_up(self) -> None:
        """启动时用几批大小不同的假图各走一遍 prefill。

        ``OXR_COMPILE=1`` 时编译选项是自动动态模式：一个函数第一次按具体形状编，见到第二种形状才编出 token 数可变
        的版本；flash-attn 4 第一次调用时现编 kernel；Triton kernel 第一次用到 ``need_kv=True`` 时也要现编。不预热
        的话，这些编译（实测第一遍比之后慢约 7 秒）落在头几批真实请求上。图的内容无所谓，只要形状不同。

        最后一批是最坏情况，见 :meth:`reserve_peak`。"""
        self._prefill_sizes((((448, 448),), ((448, 640), (640, 448)), ((448, 448), (896, 448), (448, 896)),
                             self._worst_case()))

    def reserve_peak(self) -> None:
        """按 ``max_prefill_tokens`` 塞满最大尺寸的图走一遍 prefill，让显存分配器把 prefill 的激活峰值占好，之后的批
        不再向 GPU 要新的显存（显存不够就在启动时失败，而不是跑到一半）。构造时已经做过一次；同一进程里之后再录
        CUDA Graph（录图前 PyTorch 会清一次缓存，把占好的还回去）时，录完要再调一次。"""
        self._prefill_sizes((self._worst_case(),))

    def _worst_case(self) -> tuple[tuple[int, int], ...]:
        c = self.config.vision
        side = int((preprocess.MAX_PIXELS ** 0.5) // (c.patch_size * c.hidden_stride)) * c.patch_size * c.hidden_stride
        largest = (side * side) // (c.patch_size ** 2 * c.merge_unit)
        return ((side, side),) * max(1, self.settings.max_prefill_tokens // (largest + 64))

    def _prefill_sizes(self, batches: tuple[tuple[tuple[int, int], ...], ...]) -> None:
        """每批若干张白图（宽、高），各走一遍 prefill 并等它算完。"""
        with torch.inference_mode(), torch.cuda.stream(self._prefill_stream):
            for sizes in batches:
                admitted = []
                for width, height in sizes:
                    request = self._request(self._resize(Image.new("RGB", (width, height), "white")), "", 1, 1.0,
                                            lambda _: None)
                    blocks = self.cache.allocate(blocks_for(len(request.prompt_ids) + 1))
                    if blocks is None:
                        raise RuntimeError("KV 缓存连预热的几页都分不出来")
                    admitted.append((request, blocks))
                try:
                    self._prefill_batch(admitted)
                    torch.cuda.current_stream().synchronize()
                finally:
                    for _, blocks in admitted:
                        self.cache.release(blocks)

    def _request(self, resized: np.ndarray, prompt: str, max_tokens: int, repetition_penalty: float,
                 done: Callable[[Completion | BaseException], None]) -> _Request:
        c = self.config.vision
        grid = (resized.shape[0] // c.patch_size, resized.shape[1] // c.patch_size)
        ids = self.prompts.build(prompt, grid[0] * grid[1] // c.merge_unit)
        image = torch.from_numpy(np.ascontiguousarray(resized)).pin_memory()
        return _Request(ids, grid, image, max_tokens, repetition_penalty, done)

    # ── 对外 ────────────────────────────────────────────────────────

    def submit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_tokens: int,
        repetition_penalty: float = 1.0,
        done: Callable[[Completion | BaseException], None],
    ) -> Ticket:
        """提交一个识别请求：缩放在调用方的线程里做，其余同 :meth:`submit_resized`。"""
        return self.submit_resized(self._resize(image), prompt, max_tokens=max_tokens,
                                   repetition_penalty=repetition_penalty, done=done)

    def submit_resized(
        self,
        resized: np.ndarray,
        prompt: str,
        *,
        max_tokens: int,
        repetition_penalty: float = 1.0,
        done: Callable[[Completion | BaseException], None],
    ) -> Ticket:
        """提交一个已经缩放好的图（:func:`~oxr_runtime.preprocess.resize` 的结果，uint8 ``[高, 宽, 3]``）。``done``
        在引擎线程里被调用，每个请求恰好一次。排队的请求已满时抛 :class:`Busy`，``done`` 不会被调用。"""
        if self._inbox.qsize() + len(self._waiting) >= self.settings.max_waiting:
            raise Busy(f"排队的请求已有 {self.settings.max_waiting} 个")
        request = self._request(resized, prompt, max_tokens, repetition_penalty, done)
        self._inbox.put(request)
        return Ticket(request)

    def start(self) -> None:
        self._threads = [threading.Thread(target=self._prefill_loop, name="oxr-prefill", daemon=True),
                         threading.Thread(target=self._decode_loop, name="oxr-decode", daemon=True)]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._launched.set()
        self._released.set()
        for thread in self._threads:
            thread.join()

    # ── prefill 线程 ────────────────────────────────────────────────

    def _prefill_loop(self) -> None:
        # inference_mode、当前流都是线程局部的，要在这个线程里开。
        with torch.inference_mode(), torch.cuda.stream(self._prefill_stream):
            launched: torch.cuda.Event | None = None
            while not self._stop.is_set():
                self._collect(block=not self._waiting and not self._preempted)
                while self._preempted:
                    self._waiting.insert(0, self._preempted.popleft())
                for request in [r for r in self._waiting if r.cancelled]:
                    self._waiting.remove(request)
                    request.done(Cancelled())
                if not self._waiting:
                    continue
                if launched is not None:
                    launched.synchronize()  # 上一批在 GPU 上算完再发下一批；等的时候不占 GIL
                    launched = None
                self._released.clear()
                batch = self._prefill()
                if batch is None:  # 页不够、在跑的满了，或者这一批出错了：等逐字生成线程还页
                    self._released.wait(0.05)
                    continue
                with self._landing_lock:
                    self._unlanded += len(batch.sequences)
                self._landing.append(batch)
                self._launched.set()
                launched = batch.done

    def _collect(self, block: bool) -> None:
        """把新提交的请求收进排队列表；``block`` 时最多等 50 ms 等第一个。"""
        try:
            while True:
                self._waiting.append(self._inbox.get(block=block, timeout=0.05 if block else None))
                block = False
        except queue.Empty:
            pass

    def _prefill(self) -> _InFlight | None:
        """排队的请求里能进来的发出一批，返回这一批；一个都进不来（页不够、在跑的满了）或者发出时出错（这一批的请求
        报错、页还回去，不影响在跑的序列）返回 None。"""
        s = self.settings
        admitted: list[tuple[_Request, list[int]]] = []
        tokens = 0
        # 在跑的列表归逐字生成线程，这里读它每轮更新的个数（旧一点无妨）；发出了还没并进来的也要算上。
        busy = self._live + self._unlanded

        while self._waiting and busy + len(admitted) < s.max_running:
            request = self._waiting[0]
            length = len(request.prompt_ids)
            if admitted and tokens + length > s.max_prefill_tokens:
                break
            # 给在跑的序列每个留一页的余量：它们下一步可能就要新的一页。不留的话，新请求刚进来就会被挤出去。
            needed = blocks_for(length + 1)
            if busy and self.cache.free_blocks - needed < busy + len(admitted):
                break
            blocks = self.cache.allocate(needed)
            if blocks is None:
                break
            admitted.append((self._waiting.pop(0), blocks))
            tokens += length

        if not admitted:
            return None

        try:
            with torch.cuda.stream(self._prefill_stream):
                sequences, logits = self._prefill_batch(admitted)
                tokens = logits.argmax(dim=-1)
                done = torch.cuda.Event()
                done.record()
        except Exception as exc:
            log.exception("prefill 出错")
            for request, blocks in admitted:
                self.cache.release(blocks)
                request.done(exc)
            return None

        return _InFlight(sequences, logits, tokens, done)

    # ── 逐字生成线程 ────────────────────────────────────────────────

    def _decode_loop(self) -> None:
        # 除了接 prefill 时等它的事件，这个线程的 GPU 工作都在逐字生成的流上。
        with torch.inference_mode(), torch.cuda.stream(self._decode_stream):
            while not self._stop.is_set():
                for seq in [seq for seq in self._running if seq.request.cancelled]:
                    self._running.remove(seq)
                    self._release(seq.blocks)
                    seq.request.done(Cancelled())
                # 没有在跑的序列时没别的事可做，等下一批 prefill 算完；否则只收已经算完的。
                self._land_prefill(wait=not self._running)
                self._live = len(self._running)
                if not self._running:
                    self._launched.wait(0.05)
                    self._launched.clear()
                    continue
                try:
                    self._decode()
                except Exception as exc:  # decode 出错：在跑的请求全部报错，引擎继续跑
                    log.exception("decode 出错")
                    self._fail_all(exc)
                self._live = len(self._running)

    def _land_prefill(self, wait: bool) -> None:
        """发出的 prefill 批里，算完了的按顺序并进在跑的批；``wait`` 时等最前面那一批算完。"""
        while self._landing:
            batch = self._landing[0]
            if not (wait or batch.done.query()):
                return
            wait = False
            self._landing.popleft()
            with self._landing_lock:
                self._unlanded -= len(batch.sequences)
            # 之后读这批序列的 logits、KV 的都在逐字生成的流上，先让这条流排在 prefill 之后。
            torch.cuda.current_stream().wait_event(batch.done)

            for seq in batch.sequences:
                request = seq.request
                if request.repetition_penalty != 1.0:
                    seq.seen = torch.zeros(self.config.language.vocab_size, dtype=torch.bool, device=self.device)
                    prompt = torch.tensor([i for i in request.prompt_ids if i >= 0], device=self.device)
                    seq.seen[prompt] = True

            self._emit(batch.sequences, batch.logits, batch.tokens)

    def _fail_all(self, exc: BaseException) -> None:
        for seq in self._running:
            self._release(seq.blocks)
            seq.request.done(exc)
        self._running.clear()

    def _release(self, blocks: list[int]) -> None:
        """还页，叫醒等页的 prefill 线程。"""
        self.cache.release(blocks)
        self._released.set()

    def _prefill_batch(self, admitted: list[tuple[_Request, list[int]]]) -> tuple[list[_Sequence], torch.Tensor]:
        """在当前流上发出这一批的 prefill，不等它算完。从 CPU 拷到显存的输入都在发出第一个 kernel 之前拷完：
        从普通内存往显存拷，CUDA 要先等这条流上已经发出的 kernel 做完，放在中间会让引擎线程一直等到算完。"""
        dev = self.device
        c = self.config.vision
        lengths = [len(r.prompt_ids) for r, _ in admitted]
        local = [np.arange(n) for n in lengths]
        # 图在锁页内存里，异步拷贝不用等这条流上已经发出的 kernel；拷完就不再需要，分配器会等拷贝做完才复用那块内存。
        images = [r.image.to(dev, non_blocking=True) for r, _ in admitted]
        for r, _ in admitted:
            r.image = None  # type: ignore[assignment]
        ids = torch.from_numpy(np.concatenate([np.asarray(r.prompt_ids) for r, _ in admitted])).to(dev)
        positions = torch.from_numpy(np.concatenate(local)).to(dev)
        slots = torch.from_numpy(np.concatenate([
            np.asarray(blocks)[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for (_, blocks), p in zip(admitted, local)
        ])).to(dev)
        cu = torch.tensor([0, *np.cumsum(lengths)], device=dev, dtype=torch.int32)

        t = time.perf_counter()
        pixels = torch.cat([preprocess.patchify(image, self._normalize_table, patch_size=c.patch_size,
                                                hidden_stride=c.hidden_stride) for image in images])
        visual = self.model.vision(pixels, [r.grid for r, _ in admitted])
        t = self.profile.mark(t, "prefill_vision", len(admitted))

        hidden = self.model.language.prefill(
            PrefillBatch(self.model.embed(ids, visual), positions, cu, max(lengths), slots), self.cache
        )
        last = hidden[cu[1:].long() - 1]
        t = self.profile.mark(t, "prefill_language", sum(lengths))
        self.profile.batch(len(admitted), sum(lengths))

        sequences = [_Sequence(request, blocks, len(request.prompt_ids)) for request, blocks in admitted]
        return sequences, self.model.language.logits(last)

    def _decode(self) -> None:
        if not self._running:
            return

        dev = self.device
        index = 0
        while index < len(self._running):
            seq = self._running[index]
            if seq.length + 1 > len(seq.blocks) * BLOCK_SIZE and not self._grow(seq):
                # 缓存里只剩它一个还放不下：这一个序列就超过了整个缓存，只能按长度用完收尾。
                self._running.remove(seq)
                self._finish(seq, "length")
                continue
            index += 1

        running = self._running
        if not running:
            return
        if self._speculate(running):
            return

        t = time.perf_counter()
        ids = [seq.output[-1] for seq in running]
        positions = [seq.length for seq in running]
        slots = [slot_of(seq.blocks, seq.length) for seq in running]
        seq_lens = [seq.length + 1 for seq in running]
        tables = [seq.blocks for seq in running]

        logits, tokens = None, None
        if self._graphs is not None:
            replayed = self._graphs.run(ids, positions, slots, seq_lens, tables)
            if replayed is not None:
                logits, tokens = replayed
                t = self.profile.mark(t, "decode_graph", len(running))

        if logits is None:
            width = max(len(blocks) for blocks in tables)
            batch = DecodeBatch(
                self.model.language.embed_tokens(torch.tensor(ids, device=dev)),
                torch.tensor(positions, device=dev),
                torch.tensor(slots, device=dev, dtype=torch.long),
                torch.tensor(seq_lens, device=dev, dtype=torch.int32),
                torch.tensor([b + [0] * (width - len(b)) for b in tables], device=dev, dtype=torch.int32),
            )
            logits = self.model.language.logits(self.model.language.decode(batch, self.cache))
            t = self.profile.mark(t, "decode_forward", len(running))
        for seq in running:
            seq.length += 1

        self._running = []
        self._emit(running, logits, tokens)
        self.profile.mark(t, "decode_emit")

    def _grow(self, seq: _Sequence) -> bool:
        """给 ``seq`` 再要一页。不够时把最后进来的那个序列挤出去（页还回去，请求交给 prefill 线程放回队首，之后从头重算；
        贪心解码，重算的结果相同），直到要到为止。挤到只剩它自己还要不到时返回 False。"""
        while True:
            blocks = self.cache.allocate(1)
            if blocks is not None:
                seq.blocks.extend(blocks)
                return True

            victim = self._running[-1]
            if victim is seq:
                return False

            self._running.pop()
            self._release(victim.blocks)
            victim.blocks = []
            self._preempted.append(victim.request)
            self.preemptions += 1

    def _emit(self, sequences: list[_Sequence], logits: torch.Tensor, tokens: torch.Tensor | None = None) -> None:
        """每个序列按 logits 选下一个 token，没结束的放回在跑列表。``tokens`` 是已经按 logits 贪心选好的（CUDA
        Graph 里算的）；这一批有序列要做重复惩罚时不用它，改完 logits 再选。"""
        penalized = False
        for i, seq in enumerate(sequences):
            if seq.seen is not None:
                row = logits[i]
                penalty = seq.request.repetition_penalty
                hit = seq.seen
                row[hit] = torch.where(row[hit] > 0, row[hit] / penalty, row[hit] * penalty)
                penalized = True

        next_ids = (logits.argmax(dim=-1) if tokens is None or penalized else tokens).tolist()

        for seq, token in zip(sequences, next_ids, strict=True):
            if self._take(seq, token):
                self._running.append(seq)

    def _take(self, seq: _Sequence, token: int) -> bool:
        """给 ``seq`` 收一个新 token。还要接着生成返回 True；结束了（结束符、到上限、复读）返回 False，页已经还回去。"""
        if token in self._eos:
            self._finish(seq, "stop")
            return False

        seq.output.append(token)
        if seq.seen is not None:
            seq.seen[token] = True

        if len(seq.output) >= seq.request.max_tokens:
            self._finish(seq, "length")
            return False
        if (cut := self._repetition_cut(seq.output)) is not None:
            del seq.output[cut:]
            self._finish(seq, "repetition")
            return False
        return True

    def _speculate(self, running: list[_Sequence]) -> bool:
        """在跑的序列少时做一步投机解码（见模块说明）。做了返回 True；不该做或做不了（没猜出东西、页不够、有序列要
        重复惩罚）返回 False，调用方照常逐字生成这一步。"""
        s = self.settings
        if s.speculative_tokens <= 0 or len(running) > s.speculative_max_batch:
            return False
        if any(seq.seen is not None for seq in running):
            return False

        drafts = []
        for seq in running:
            if seq.proposer is None:
                seq.proposer = NgramProposer(seq.output)
            else:
                seq.proposer.extend(seq.output, seq.indexed)
            seq.indexed = len(seq.output)
            # 猜的 token 收下以后不能超过 max_tokens（最后那个由模型给），也不能超过位置上限。
            room = min(seq.request.max_tokens - len(seq.output) - 1, s.max_positions - seq.length - 1)
            drafts.append(seq.proposer.propose(seq.output, min(s.speculative_tokens, max(room, 0))))
        if not any(drafts):
            return False

        # 录了图的话每个序列固定 1 + speculative_tokens 个 token（图的形状固定）；没录图时按这一批最长的猜测补齐。
        width = self._extend_graphs.tokens_per_seq if self._extend_graphs else 1 + max(len(d) for d in drafts)
        if any(seq.length + width > s.max_positions for seq in running):  # 补位的 token 也占位置，不能超过位置表
            return False
        for seq in running:
            missing = blocks_for(seq.length + width) - len(seq.blocks)
            if missing > 0:
                more = self.cache.allocate(missing)
                if more is None:  # 页不够：这一步不投机（已经分到的页留给这个序列，之后接着用）
                    return False
                seq.blocks.extend(more)

        t = time.perf_counter()
        dev = self.device
        ids, positions, slots = [], [], []
        for seq, draft in zip(running, drafts, strict=True):
            # 上一个 token（还没写进缓存）加上猜的，补到同样长：补的 token 在序列末尾之后，前面的 token 看不到它们，
            # 它们写进缓存的值之后也会被覆盖。
            for j, token in enumerate([seq.output[-1], *draft] + [0] * (width - 1 - len(draft))):
                ids.append(token)
                positions.append(seq.length + j)
                slots.append(slot_of(seq.blocks, seq.length + j))
        seq_lens = [seq.length + width for seq in running]
        tokens = None
        if self._extend_graphs is not None:
            tokens = self._extend_graphs.run(ids, positions, slots, seq_lens, [seq.blocks for seq in running])
        if tokens is None:
            table_width = max(len(seq.blocks) for seq in running)
            batch = ExtendBatch(
                self.model.language.embed_tokens(torch.tensor(ids, device=dev)),
                torch.tensor(positions, device=dev),
                torch.tensor(slots, device=dev, dtype=torch.long),
                torch.tensor(seq_lens, device=dev, dtype=torch.int32),
                torch.tensor([seq.blocks + [0] * (table_width - len(seq.blocks)) for seq in running], device=dev,
                             dtype=torch.int32),
                width,
            )
            hidden = self.model.language.extend(batch, self.cache)
            tokens = self.model.language.logits(hidden).argmax(dim=-1).view(len(running), width)
        predicted = tokens.tolist()

        self._running = []
        self.speculative_steps += 1
        for seq, draft, row in zip(running, drafts, predicted, strict=True):
            taken = accept(draft, row[: len(draft) + 1])
            # 写进缓存的是「上一个 token + 收下的猜测」，模型最后给的那个 token 下一步才写。
            seq.length += len(taken)
            self.speculative_taken += len(taken)
            if all(self._take(seq, token) for token in taken):
                self._running.append(seq)
        self.profile.mark(t, "speculate", len(running))
        return True

    def _repetition_cut(self, output: list[int]) -> int | None:
        s = self.settings
        return repetition_cut(
            output, span=s.repetition_span, max_period=s.repetition_max_period, every=s.repetition_check_every
        )

    def _finish(self, seq: _Sequence, reason: str) -> None:
        self._release(seq.blocks)
        seq.blocks = []
        request = seq.request
        completion = Completion(
            text=self.prompts.decode(seq.output),
            token_ids=seq.output,
            finish_reason=reason,
            prompt_tokens=len(request.prompt_ids),
            seconds=time.monotonic() - request.submitted,
        )
        seq.request = None  # type: ignore[assignment]
        request.done(completion)
