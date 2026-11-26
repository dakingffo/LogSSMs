from __future__ import annotations

from collections import Counter, deque
from contextlib import ExitStack, contextmanager, nullcontext
from heapq import heappop, heappush
import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from logssms.cache import CacheResources
from logssms.cache.config import LogStateCacheConfig, MarconiCacheConfig
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.manager import LogStateCacheManager
from logssms.cache.marconi.block_pool import HybridBlockPool
from logssms.cache.marconi.manager import (
    MarconiCacheManager,
    MarconiCachePlan,
    marconi_layout,
)
from logssms.cache.marconi.prefix_cache import MarconiPrefixCache
from logssms.core import ChunkedRequest, Request, SamplingParams
from logssms.distributed.info import DistributedInfo
from logssms.engine import Engine, ForwardOutput
from logssms.engine.mixed_graph import MixedGraphRunner
from logssms.models.config import RotaryConfig
from logssms.models.qwen3_next import Qwen3_5Config
from logssms.scheduler.decode import DecodeManager
from logssms.scheduler.prefill import PrefillManager
from logssms.scheduler.profile import SchedulerProfiler
from logssms.scheduler.scheduler import Scheduler
from logssms.scheduler.table import TableManager

from datasets.parser import BlockPrompt


class VirtualClock:
    def __init__(self, step_ms):
        if not math.isfinite(step_ms):
            raise ValueError("Virtual clock step must be finite")
        self.step = round(step_ms * 1_000_000)
        if self.step < 1:
            raise ValueError("Virtual clock step must be at least one nanosecond")
        self.now = 0
        self._timestamp = -1

    def monotonic_ns(self):
        self._timestamp = max(self.now, self._timestamp + 1)
        return self._timestamp


class CompletedEvent:
    def synchronize(self):
        pass

    def query(self):
        return True


class NumericalKernels:
    @staticmethod
    def copy_states(pool, sources, outputs):
        assert len(sources) == len(outputs)
        for slot in (*sources, *outputs):
            pool.allocator.slot(int(slot))

    @staticmethod
    def restore(pool, chains):
        for base, logs, output in chains:
            if base is not None:
                assert pool.allocator.slot(base).kind == "checkpoint"
            assert pool.allocator.slot(output).kind == "work"
            for slot, begin, end in logs:
                assert pool.allocator.slot(slot).kind == "log"
                assert 0 <= begin < end <= pool.page_size

    @staticmethod
    def copy_conv(pool, sources, source_granules, outputs, output_granules):
        for slot, granule in zip((*sources.tolist(), *outputs.tolist()),
                                 (*source_granules.tolist(), *output_granules.tolist())):
            pool.validate_conv(slot, granule)


class MetadataPool(MHALogStatePool):
    """Production allocation/descriptors with numerical storage on the meta device."""

    def __init__(self, **kwargs):
        super().__init__(device=torch.device("meta"), **kwargs)
        self._device = torch.device("cpu")
        self.kernels = NumericalKernels()
        for name in ("payload_frame_table", "conv_frame_table", "decay_frame_table"):
            setattr(self, name, torch.full_like(getattr(self, name), -1, device="cpu"))
        self.min_free_frames = self.allocator.free_frames

    def reserve(self, **kwargs):
        result = super().reserve(**kwargs)
        self.min_free_frames = min(self.min_free_frames, self.allocator.free_frames)
        return result

    def zero_state(self, slot):
        assert self.allocator.slot(slot).kind == "work"

    def zero_work_conv(self, slot):
        self.zero_state(slot)

    def zero_work_slots(self, slots):
        for slot in slots:
            self.zero_state(slot)

    def validate_conv(self, slot, granule):
        record = self.allocator.slot(int(slot))
        assert 0 <= granule < self.layout.num_granules
        assert record.conv

    def copy_conv_snapshot(self, source_slot, source_granule, output_slot, output_granule=0):
        self.validate_conv(source_slot, source_granule)
        self.validate_conv(output_slot, output_granule)


class MarconiMetadataPool(HybridBlockPool):
    """Production block/group allocation with tensor storage on the meta device."""

    def __init__(self, layout, num_blocks: int, dtype: torch.dtype):
        super().__init__(layout, num_blocks, torch.device("meta"), dtype)
        self._device = torch.device("cpu")
        self.group_table = torch.zeros(num_blocks, layout.state_groups, device="cpu", dtype=torch.int64)

    def copy_state(self, source, target):
        assert source in self.state_blocks and target in self.state_blocks

    def zero_state(self, slot):
        assert slot in self.state_blocks


class _SyncFuture:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


class _SyncExecutor:
    def submit(self, fn, *args, **kwargs):
        return _SyncFuture(fn(*args, **kwargs))

    def shutdown(self, wait=False, cancel_futures=False):
        pass


_KNOWN_MODEL_DIMS = {
    "Qwen3.5-0.8B": (8, 1024, 3584),
    "Qwen3.5-2B": (8, 2048, 6144),
    "Qwen3.5-4B": (16, 2560, 9216),
}


def _marconi_model_config(case, max_seq_len):
    model = case["model"]
    attn_layers, lin_layers = model["attention_layers"], model["linear_layers"]
    total_layers = attn_layers + lin_layers
    interval = max(1, total_layers // max(1, attn_layers))
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(total_layers)
    ]
    defaults = _KNOWN_MODEL_DIMS.get(model.get("name"), (16, 2560, 9216))
    num_qo_heads = model.get("attention_heads", defaults[0])
    hidden_size = model.get("hidden_size", defaults[1])
    intermediate_size = model.get("intermediate_size", defaults[2])
    return Qwen3_5Config(
        num_layers=total_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=model["kv_heads"],
        head_dim=model["head_dim"],
        hidden_size=hidden_size,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(64, max_seq_len, 10000000.0, None),
        tie_word_embeddings=True,
        architectures=["Qwen3_5ForConditionalGeneration"],
        layer_types=layer_types,
        linear_num_key_heads=model["linear_key_heads"],
        linear_num_value_heads=model["linear_value_heads"],
        linear_key_head_dim=model["linear_key_dim"],
        linear_value_head_dim=model["linear_value_dim"],
        linear_conv_kernel_dim=model["conv_width"],
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        mlp_only_layers=[],
        decoder_sparse_step=1,
        mtp_num_hidden_layers=0,
    )


def _make_marconi_manager(case, max_seq_len, *, block_size=128, bootstrap_multiplier=5):
    config = SimpleNamespace(
        model_config=_marconi_model_config(case, max_seq_len),
        tp_info=DistributedInfo(0, case["tp_size"]),
        dtype=torch.bfloat16,
        cache_config=MarconiCacheConfig(block_size=block_size, bootstrap_multiplier=bootstrap_multiplier),
        max_running_req=case["max_running_requests"],
        max_seq_len=max_seq_len,
    )
    layout = marconi_layout(config)
    block_bytes = layout.block_bytes(config.dtype.itemsize)
    num_blocks = case["cache_bytes_per_rank"] // block_bytes
    max_len = (max_seq_len + 63) // 64 * 64
    plan = MarconiCachePlan(
        num_blocks, num_blocks * layout.block_size, max_seq_len, max_len,
        layout, num_blocks, 0,
    )
    pool = MarconiMetadataPool(layout, num_blocks, config.dtype)
    dummy_page, = pool.allocator.allocate(1)
    dummy_state = pool.allocate_state()
    pool.zero_state(dummy_state)
    rows = case["max_running_requests"] + 1
    pages = torch.zeros((rows, max_len), dtype=torch.int32)
    pages[-1] = dummy_page * layout.block_size + torch.arange(max_len, dtype=torch.int32) % layout.block_size
    states = torch.full((rows, max_len), dummy_state, dtype=torch.int32)
    resources = CacheResources(pool, pages, pool, states, None)
    return MarconiCacheManager(resources, config, plan)


def make_manager(case, mode, max_seq_len, window, half_life, *, enable_hit_materialization=False,
                 block_size=128, bootstrap_multiplier=5):
    if mode == "marconi":
        assert case["tp_size"] == case["gpu_count"]
        return _make_marconi_manager(
            case, max_seq_len,
            block_size=case.get("marconi_block_size", block_size),
            bootstrap_multiplier=case.get("marconi_bootstrap_multiplier", bootstrap_multiplier),
        )
    model, tp, g = case["model"], case["tp_size"], case.get("granularity", LogStateCacheConfig.granularity)
    assert tp == case["gpu_count"] == 2
    frame_bytes = (2 * model["attention_layers"] * g * (model["kv_heads"] // tp)
                   * model["head_dim"] * torch.bfloat16.itemsize)
    pool = MetadataPool(
        num_frames=case["cache_bytes_per_rank"] // frame_bytes,
        page_size=g, granularity=g,
        num_attention_layers=model["attention_layers"],
        num_kv_heads=model["kv_heads"] // tp, head_dim=model["head_dim"],
        num_linear_layers=model["linear_layers"],
        num_key_heads=model["linear_key_heads"] // tp,
        num_value_heads=model["linear_value_heads"] // tp,
        head_k_dim=model["linear_key_dim"], head_v_dim=model["linear_value_dim"],
        conv_dim=(2 * model["linear_key_heads"] * model["linear_key_dim"]
                  + model["linear_value_heads"] * model["linear_value_dim"]) // tp,
        state_len=model["conv_width"] - 1, dtype=torch.bfloat16,
    )
    rows = case["max_running_requests"] + 1
    resources = CacheResources(
        kv_pool=pool, linear_state=pool,
        page_table=torch.zeros((rows, max_seq_len), dtype=torch.int32),
        state_table=torch.full((rows, max_seq_len), -1, dtype=torch.int32),
        work_state_table=torch.full((rows, 1), -1, dtype=torch.int32),
    )
    with patch.dict("os.environ", {"LOGSSMS_PROFILE_SCHEDULER": "1"}):
        manager = LogStateCacheManager(
            num_pages=pool.allocator.num_frames - 1, num_states=pool.allocator.num_frames,
            num_work_states=rows, page_size=g, resources=resources, granularity=g,
            enable_merge_adjustment=mode == "online",
            enable_hit_materialization=enable_hit_materialization,
            min_sampling_len=window, hit_half_life=half_life,
        )
    # Only the CPU planner's instrumentation is enabled; no CUDA timing events.
    manager._profile_enabled = False
    manager._match_profile_enabled = False
    return manager


def cpu_batch_mapping(indices, page_table, *, decode=False, dummy_row=0):
    """CPU equivalent of the CUDA-only index expansion kernel."""
    size = indices.num_tokens + indices.prefill_padding
    positions = torch.zeros(size, dtype=torch.int32)
    rows = torch.full((size,), dummy_row, dtype=torch.int64)
    columns = torch.zeros(size, dtype=torch.int64)
    locations = torch.full((size,), -1, dtype=torch.int32)
    for i, row in enumerate(indices.table_ids.tolist()):
        begin, end = indices.cu_seqlens_q_cpu[i:i + 2].tolist()
        offset = indices.prefill_padding if i >= indices.padding_after else 0
        output = slice(begin + offset, end + offset)
        start = int(indices.cached_lens[i])
        pos = torch.arange(start, start + end - begin)
        positions[output], rows[output], columns[output] = pos, row, pos
        locations[output] = page_table[row, pos]
    if decode:
        pages = []
        for row, length in zip(indices.table_ids.tolist(), indices.seq_lens_cpu.tolist()):
            pages.append(page_table[row, :length:indices.page_size] // indices.page_size)
        indices.page_indices = torch.cat(pages)
    return positions, (rows, columns), locations


@contextmanager
def numerical_boundaries(mode="default"):
    def pin(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return tensor

    with ExitStack() as stack:
        stack.enter_context(patch.dict("os.environ", {"LOGSSMS_PROFILE_SCHEDULER": "0"}))
        stack.enter_context(patch.object(torch.Tensor, "pin_memory", pin))
        stack.enter_context(patch("logssms.kernel.metadata.make_batch_mapping", cpu_batch_mapping))
        stack.enter_context(patch("logssms.kernel.linear_attention.logstate.copy_states",
                                  NumericalKernels.copy_states))
        stack.enter_context(patch(
            "logssms.layers.linear_attention_backend.mixed.prepare_mixed_linear_metadata",
            lambda *args: None,
        ))
        if mode == "marconi":
            orig_blocks = MarconiPrefixCache._blocks

            def cached_blocks(self, node):
                cached = getattr(node, "_cached_blocks", None)
                if cached is None:
                    cached = orig_blocks(self, node)
                    node._cached_blocks = cached
                return cached

            stack.enter_context(patch(
                "logssms.cache.marconi.manager.ProcessPoolExecutor",
                lambda *args, **kwargs: _SyncExecutor(),
            ))
            stack.enter_context(patch("torch.distributed.broadcast", lambda *args, **kwargs: None))
            stack.enter_context(patch.object(MarconiPrefixCache, "_blocks", cached_blocks))
        yield


class RecordedEngine(Engine):
    def __init__(self, scheduler, manager, records, max_seq_len):
        self.scheduler, self.records = scheduler, records
        self.config = scheduler.config
        self.device = torch.device("cpu")
        self.page_table = manager.page_table
        self.cache_manager = manager
        self.ctx = SimpleNamespace(kv_cache=getattr(manager, "logstate_pool", getattr(manager, "pool", None)))
        self.stream = SimpleNamespace(wait_stream=lambda _: None)
        self.profiler = SimpleNamespace(enabled=False)
        self.piecewise_graph_runner = None
        self.decode_graph_runner = SimpleNamespace(
            max_graph_bs=0,
            can_use_cuda_graph=lambda _: False,
            pad_batch=lambda batch: setattr(batch, "padded_reqs", batch.reqs),
        )
        self.attn_backend = SimpleNamespace(
            uses_paged_indices=True, prepare_metadata=lambda _: None,
            prepare_mixed_metadata=lambda _: None,
        )
        self.linear_attn_backend = SimpleNamespace(prepare_metadata=lambda _: None)
        self.sampler = SimpleNamespace(prepare=lambda _: None, prepare_params=lambda _: None)
        self.max_seq_len = max_seq_len
        self.dummy_req = Request(
            torch.zeros(1, dtype=torch.int32), self.config.max_running_req, 0, 1, -1,
            SamplingParams(max_tokens=1), None, 2,
        )
        self.mixed_graph_runner = MixedGraphRunner(self)

    def prepare_batch(self, batch, cache_manager):
        self.scheduler.observe_admissions(batch)
        if batch.prefill_count:
            self.scheduler.prefill_steps.append(dict(
                phase=batch.phase,
                prefill_requests=batch.prefill_count,
                decode_requests=len(batch.decode_reqs),
                tokens=sum(req.extend_len for req in batch.prefill_reqs),
                requests=[dict(uid=req.uid, cached_len=req.cached_len,
                               device_len=req.device_len, tokens=req.extend_len)
                          for req in batch.prefill_reqs],
            ))
        return super().prepare_batch(batch, cache_manager)

    def forward_batch(self, batch, args):
        self.cache_manager.after_forward(batch)
        tokens = []
        for req in batch.reqs:
            if isinstance(req, ChunkedRequest):
                tokens.append(0)
                continue
            record = self.records[req.uid]
            index = req.device_len - len(record["prompt_token_ids"])
            assert index >= 0
            output = record["generated_token_ids"]
            # Overlap can submit one decode after a request's actual final token.
            tokens.append(output[min(index, len(output) - 1)])
        tensor = torch.tensor(tokens, dtype=torch.int32)
        return ForwardOutput(tensor, tensor, CompletedEvent())


class ReplayScheduler(Scheduler):
    """Run the production overlap loop with deterministic token playback and I/O."""

    def __init__(self, manager, records, case, batch_size, max_seq_len,
                 observe_admissions, scheduling_policy="mixed", prefill_budget=8192,
                 *, tree_concurrency=None, clock=None):
        self.records = records
        self.remaining = list(range(len(records)))
        self.finished, self.active_trees = set(), set()
        self.client_finished = set()
        self._outgoing_completions, self._incoming_completions = [], []
        self.tree_concurrency = tree_concurrency
        self.open_trees = set()
        self.tree_remaining = Counter(record["tree_id"] for record in records)
        self.peak_tree_concurrency = 0
        self.received, self.returned_tokens = set(), {}
        self.arrival_batch_size = batch_size
        self.observe_admissions = observe_admissions
        self.prefill_steps = []
        granularity = getattr(manager, "granularity", manager.page_size)
        cache_cfg = getattr(getattr(manager, "config", None), "cache_config", None)
        prefill_alignment = getattr(cache_cfg, "prefill_alignment", granularity)
        self.config = SimpleNamespace(
            max_running_req=case["max_running_requests"], page_size=granularity,
            decode_batch_budget=case["max_running_requests"],
            scheduling_policy=scheduling_policy, mixed_prefill_tokens=128,
            cache_config=SimpleNamespace(prefill_alignment=prefill_alignment),
            model_config=SimpleNamespace(is_hybrid=True),
        )
        self.cache_manager = manager
        self.table_manager = TableManager(self.config.max_running_req, manager.page_table)
        self.token_pool = self.table_manager.token_pool
        self.device = torch.device("cpu")
        self.output_cap = case["output_cap"]
        self.eos_token_id = case["eos_token_id"]
        self.ar_manager = DecodeManager(
            self.config, self.device, manager, self.table_manager, self.eos_token_id,
        )
        self.prefill_manager = PrefillManager(self.token_pool)
        self.prefill_budget = prefill_budget
        self.engine = RecordedEngine(self, manager, records, max_seq_len)
        self.stream = object()
        self.engine_stream_ctx = nullcontext()
        self.profiler = SchedulerProfiler(self.device, manager)
        self._last_stall_report = 0.0
        self._deferred_req_frees = []
        self.by_name = {record["uid"]: i for i, record in enumerate(records)}
        self.clock = clock
        if clock is not None:
            sessions = {}
            for uid, record in enumerate(records):
                sessions.setdefault(record["tree_id"], []).append(uid)
            self._unopened = deque(sessions.items())
            self._arrivals = []
            self.arrival_times = {}
            self._open_sessions()

    def _open_sessions(self):
        while self._unopened and (
            self.tree_concurrency is None or len(self.open_trees) < self.tree_concurrency
        ):
            tree_id, uids = self._unopened.popleft()
            self.open_trees.add(tree_id)
            for uid in uids:
                when = self.clock.now + round(self.records[uid]["t"] * 1_000_000_000)
                heappush(self._arrivals, (when, uid))
        self.peak_tree_concurrency = max(self.peak_tree_concurrency, len(self.open_trees))

    def advance_to_next_arrival(self) -> bool:
        if self.clock is None or not self._arrivals:
            return False
        self.clock.now = max(self.clock.now, self._arrivals[0][0])
        return True

    def _receive_timed(self, blocking):
        if blocking and not self.client_has_pending_completions:
            self.advance_to_next_arrival()
        for _ in range(self.arrival_batch_size):
            if not self._arrivals or self._arrivals[0][0] > self.clock.now:
                break
            _, uid = heappop(self._arrivals)
            self.arrival_times[uid] = self.clock.now / 1_000_000_000
            yield self._make_message(uid)

    def _make_message(self, uid):
        from logssms.message import UserMsg

        record = self.records[uid]
        ids = record["prompt_token_ids"]
        self.received.add(uid)
        return UserMsg(
            uid, ids.tensor() if isinstance(ids, BlockPrompt) else torch.tensor(ids, dtype=torch.int32),
            SamplingParams(max_tokens=record.get("output_cap", self.output_cap),
                           ignore_eos=record.get("ignore_eos", False)),
            profile_tag=record["uid"], return_token_ids=True,
            cache_session_id=str(record["tree_id"]),
        )

    @property
    def client_has_pending_completions(self):
        return bool(self._outgoing_completions or self._incoming_completions)

    def receive_msg(self, *, blocking):
        # Model one receive interval for detokenization, streaming, and
        # client submission before the response can trigger a new request.
        visible, self._incoming_completions = (
            self._incoming_completions, self._outgoing_completions,
        )
        self._outgoing_completions = []
        for uid in visible:
            tree_id = self.records[uid]["tree_id"]
            self.client_finished.add(uid)
            if self.clock is None:
                self.active_trees.remove(tree_id)
            self.tree_remaining[tree_id] -= 1
            if self.tree_remaining[tree_id] == 0 and tree_id in self.open_trees:
                self.open_trees.remove(tree_id)

        if self.clock is not None:
            self._open_sessions()
            yield from self._receive_timed(blocking)
            return

        sent = 0
        for uid in self.remaining.copy():
            record = self.records[uid]
            if record["tree_id"] in self.active_trees:
                continue
            if record["parent"] is not None and self.by_name[record["parent"]] not in self.client_finished:
                continue
            if len(self.active_trees) >= self.config.max_running_req or sent == self.arrival_batch_size:
                break
            tree_id = record["tree_id"]
            if self.tree_concurrency is not None and tree_id not in self.open_trees:
                if len(self.open_trees) >= self.tree_concurrency:
                    continue
                self.open_trees.add(tree_id)
            self.active_trees.add(tree_id)
            self.peak_tree_concurrency = max(
                self.peak_tree_concurrency,
                len(self.open_trees) if self.tree_concurrency is not None else len(self.active_trees),
            )
            self.remaining.remove(uid)
            sent += 1
            yield self._make_message(uid)

    def send_result(self, replies):
        for reply in replies:
            record = self.records[reply.uid]
            output = self.returned_tokens.setdefault(reply.uid, [])
            assert reply.next_token == record["generated_token_ids"][len(output)]
            output.append(reply.next_token)
            if reply.finished:
                assert output == record["generated_token_ids"]
                self.finished.add(reply.uid)
                self._outgoing_completions.append(reply.uid)
