from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from logssms.cache import CacheResources
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.manager import LogStateCacheManager
from logssms.core import ChunkedRequest, Request, SamplingParams
from logssms.engine import Engine, ForwardOutput
from logssms.engine.mixed_graph import MixedGraphRunner
from logssms.scheduler.decode import DecodeManager
from logssms.scheduler.prefill import PrefillManager
from logssms.scheduler.profile import SchedulerProfiler
from logssms.scheduler.scheduler import Scheduler
from logssms.scheduler.table import TableManager
from logssms.utils.mixed import prefill_buckets, prefill_partitions


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


def make_manager(case, mode, max_seq_len, window, half_life):
    model, tp, g = case["model"], case["tp_size"], 32
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
            enable_memorized_evict=True, enable_merge_adjustment=mode == "online",
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
def numerical_boundaries():
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
        yield


class RecordedEngine(Engine):
    def __init__(self, scheduler, manager, records, max_seq_len):
        self.scheduler, self.records = scheduler, records
        self.config = scheduler.config
        self.device = torch.device("cpu")
        self.page_table = manager.page_table
        self.cache_manager = manager
        self.ctx = SimpleNamespace(kv_cache=manager.logstate_pool)
        self.stream = SimpleNamespace(wait_stream=lambda _: None)
        self.profiler = SimpleNamespace(enabled=False)
        self.piecewise_graph_runner = None
        self.decode_graph_runner = SimpleNamespace(
            max_graph_bs=self.config.max_running_req,
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
        runner = MixedGraphRunner.__new__(MixedGraphRunner)
        runner.graphs, runner.sizes = {}, []
        runner.partitions = {len(p): p for p in prefill_partitions(self.config.mixed_prefill_tokens, 32)}
        runner.buckets = prefill_buckets(self.config.mixed_prefill_tokens, 32)
        runner.buckets_by_count = {
            count: tuple(p for p in runner.buckets if len(p) == count)
            for count in runner.partitions
        }
        self.mixed_graph_runner = runner

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
                 observe_admissions, scheduling_policy="mixed", prefill_budget=8192):
        self.records = records
        self.remaining = list(range(len(records)))
        self.finished, self.active_trees = set(), set()
        self.received, self.returned_tokens = set(), {}
        self.arrival_batch_size = batch_size
        self.observe_admissions = observe_admissions
        self.prefill_steps = []
        self.config = SimpleNamespace(
            max_running_req=case["max_running_requests"], page_size=32,
            decode_batch_budget=case["max_running_requests"],
            scheduling_policy=scheduling_policy, mixed_prefill_tokens=128,
            model_config=SimpleNamespace(is_hybrid=True),
        )
        self.cache_manager = manager
        self.table_manager = TableManager(self.config.max_running_req, manager.page_table)
        self.token_pool = self.table_manager.token_pool
        self.device = torch.device("cpu")
        self.eos_token_id = -1
        self.ar_manager = DecodeManager(self.config, self.device, manager, self.table_manager, -1)
        self.prefill_manager = PrefillManager(self.token_pool, 32, manager)
        self.prefill_budget = prefill_budget
        self.engine = RecordedEngine(self, manager, records, max_seq_len)
        self.stream = object()
        self.engine_stream_ctx = nullcontext()
        self.profiler = SchedulerProfiler(self.device, manager)
        self._last_stall_report = 0.0
        self._deferred_req_frees = []
        self.by_name = {record["uid"]: i for i, record in enumerate(records)}

    def receive_msg(self, *, blocking):
        from logssms.message import UserMsg

        sent = 0
        for uid in self.remaining.copy():
            record = self.records[uid]
            if record["tree_id"] in self.active_trees:
                continue
            if record["parent"] is not None and self.by_name[record["parent"]] not in self.finished:
                continue
            if len(self.active_trees) >= self.config.max_running_req or sent == self.arrival_batch_size:
                break
            self.active_trees.add(record["tree_id"])
            self.received.add(uid)
            self.remaining.remove(uid)
            sent += 1
            yield UserMsg(
                uid, torch.tensor(record["prompt_token_ids"], dtype=torch.int32),
                SamplingParams(max_tokens=len(record["generated_token_ids"]), ignore_eos=True),
                profile_tag=record["uid"], return_token_ids=True,
            )

    def send_result(self, replies):
        for reply in replies:
            record = self.records[reply.uid]
            output = self.returned_tokens.setdefault(reply.uid, [])
            assert reply.next_token == record["generated_token_ids"][len(output)]
            output.append(reply.next_token)
            if reply.finished:
                assert output == record["generated_token_ids"]
                self.finished.add(reply.uid)
                self.active_trees.remove(record["tree_id"])
