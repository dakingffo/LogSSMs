from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, Sequence

import torch
from logssms.cache.base import BaseCacheManager, CacheResources, PrefillAdmission, PrefillAllocation
from logssms.cache.factory import SUPPORTED_CACHE_FACTORIES, BaseCacheFactory, CachePlan
from logssms.cache.logstate import prefix_cache as _prefix_cache_mod
from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.cache.logstate.layout import LOG_STATE_GRANULARITY, LOG_STATE_REPLAY_WINDOW, LogStateSlotLayout
from logssms.cache.logstate.metadata import LogStateCacheMetadata
from logssms.cache.logstate.native import load_logstate_core
from logssms.cache.logstate.prefix_cache import (
    LogStateCacheHandle, LogStateEntry, LogStateKind, LogStatePrefixProxy, LogStateRecovery,
)
from logssms.cache.prefix_cache import BaseCacheHandle, MatchResult
from logssms.core import Request
from logssms.utils import align_ceil, align_down, div_ceil, init_logger
from logssms.utils.torch_utils import make_device_vectors

if TYPE_CHECKING:
    from logssms.core import Batch
    from logssms.engine import EngineConfig
    from logssms.models.config import AffineLogCacheSpec
    from logssms.scheduler.prefill import PendingRequest
    from logssms.speculator import SpeculatorReserve

logger = init_logger(__name__)


class LogStateCapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class RestoreTask:
    output_slot: int
    base_slot: int | None
    logs: tuple[LogStateEntry, ...]
    source_conv_slot: int
    source_conv_granule: int
    output_conv_granule: int


class LogStateConstruction:
    """Restore execution through the pool API, independent of cache policy."""

    def __init__(self, pool: MHALogStatePool) -> None:
        self.pool = pool

    def restore(self, restores: Sequence[RestoreTask]) -> None:
        if not restores:
            return
        pool = self.pool
        if (wait_fn := getattr(pool, "wait_slots_on_current_stream", None)) is not None:
            deps = [t.source_conv_slot for t in restores]
            for t in restores:
                if t.base_slot is not None:
                    deps.append(t.base_slot)
                deps.extend(e.slot for e in t.logs)
            wait_fn(deps)
        direct = [t for t in restores if not t.logs]
        if direct:
            assert all(t.base_slot is not None for t in direct)
            pool.copy_states([t.base_slot for t in direct], [t.output_slot for t in direct])
        chains = [(t.base_slot, [(e.slot, 0, e.end - e.begin) for e in t.logs], t.output_slot)
                  for t in restores if t.logs]
        conv_vectors = make_device_vectors([
            [t.source_conv_slot for t in restores], [t.source_conv_granule for t in restores],
            [t.output_slot for t in restores], [t.output_conv_granule for t in restores],
        ], pool.device)
        if chains:
            pool.kernels.restore(pool, chains)
        pool.kernels.copy_conv(pool, *conv_vectors)


@dataclass(frozen=True)
class LogStateCachePlan(CachePlan):
    linear_spec: AffineLogCacheSpec
    num_states: int
    num_work_states: int
    state_units: int


class LogStateCacheManager(BaseCacheManager):
    def __init__(
        self, num_pages: int, num_states: int, num_work_states: int, page_size: int,
        resources: CacheResources, granularity: int = LOG_STATE_GRANULARITY,
        enable_merge_adjustment: bool = True, min_sampling_len: int = 8192,
        max_density_exponent: float = 16.0, hit_half_life: int = 1024,
        enable_hit_materialization: bool = False,
    ) -> None:
        pool, state_table, work_state_table = resources.linear_state, resources.state_table, resources.work_state_table
        assert isinstance(pool, MHALogStatePool), "LogStateCacheManager requires MHALogStatePool."
        assert state_table is not None and work_state_table is not None and granularity > 0 and page_size == granularity

        self.num_pages, self.num_states, self.num_work_states = num_pages, num_states, num_work_states
        self.granularity = granularity
        self.enable_merge_adjustment = enable_merge_adjustment
        self.enable_hit_materialization = enable_hit_materialization
        self.logstate_pool, self._allocator = pool, pool.allocator
        self._construction = LogStateConstruction(pool)
        self._core = dict(load_logstate_core().make_core(
            pool.allocator.core_id, granularity, page_size, enable_merge_adjustment,
            min_sampling_len, float(max_density_exponent), hit_half_life, enable_hit_materialization,
        ))
        self._clock_overridden = False
        self._group_ids: dict[object, int] = {}
        self._decode_state: dict[int, list[int]] = {}
        self._lazy_free_pages: list[int] | None = None
        self._lazy_free_states: list[int] | None = None

        proxy = LogStatePrefixProxy(self._core, resources.page_table.device, self._sync_clock)
        super().__init__(page_size=page_size, resources=resources, prefix_cache=proxy)

        reserved = pool.reserve(kv=1, logs=1, works=2)
        assert reserved is not None, "Not enough frames to reserve LogState dummy slots."
        self._dummy_kv_frame, self._dummy_log_slot = reserved[0][0], reserved[1][0]
        self._dummy_work_slot, self._pressure_work_slot = reserved[2]
        self.page_table[-1].fill_(self._dummy_kv_frame * page_size)
        state_table[-1].fill_(self._dummy_log_slot)
        work_state_table[-1, 0] = self._dummy_work_slot
        pool.zero_state(self._dummy_work_slot)
        pool.zero_work_conv(self._dummy_work_slot)
        self._core["set_reserved_dummies"](
            self._dummy_kv_frame, self._dummy_log_slot, self._dummy_work_slot, self._pressure_work_slot,
        )

    def _sync_clock(self) -> None:
        t = _prefix_cache_mod.time
        if t.monotonic_ns is not _prefix_cache_mod._STDLIB_MONOTONIC_NS:
            self._clock_overridden = True
            if (now := getattr(t, "now", None)) is not None:
                self._core["advance_virtual_clock"](int(now))
            else:
                self._core["set_fixed_clock"](int(t.monotonic_ns()))
        elif self._clock_overridden:
            self._core["clear_fixed_clock"]()
            self._core["set_virtual_clock"](False, 0)
            self._clock_overridden = False

    def _group_id_for(self, req: PendingRequest) -> int:
        key = req.cache_session_id
        if key is None and req.profile_tag and "/turn-" in req.profile_tag:
            key = req.profile_tag.partition("/turn-")[0]
        if key is None:
            return -1
        if (gid := self._group_ids.get(key)) is None:
            gid = len(self._group_ids) + 1
            self._group_ids[key] = gid
        return gid

    @property
    def admission_capacity(self) -> int:
        self.logstate_pool.poll_reclamation()
        return int(self._core["admission_capacity"]())

    @property
    def available_size(self) -> int:
        self.logstate_pool.poll_reclamation()
        return int(self._core["available_size"]())

    def plan_prefill_chunk(self, remain_len: int, token_budget: int) -> int:
        chunk = min(remain_len, token_budget)
        return align_down(chunk, self.granularity) if chunk < remain_len else chunk

    def begin_prefill_admission(
        self, active_decode: tuple[tuple[Request, int], ...], inflight_prefill: tuple[Request, ...],
    ) -> PrefillAdmission:
        reserved = int(self._core["begin_prefill_admission"](
            [r.table_idx for r, _ in active_decode], [int(r.can_decode) for r, _ in active_decode],
            [r.table_idx for r in inflight_prefill], [r.max_device_len - r.output_len for r in inflight_prefill],
            [r.max_device_len for r in inflight_prefill], [r.remain_len for r in inflight_prefill],
        ))
        return PrefillAdmission(active_decode, inflight_prefill, reserved)

    def match_req(self, req: PendingRequest) -> MatchResult:
        match = super().match_req(req)
        h = match.cuda_handle
        assert isinstance(h, LogStateCacheHandle)
        object.__setattr__(h, "query_uid", req.uid)
        object.__setattr__(h, "request_end", req.input_len + req.output_len)
        object.__setattr__(h, "analysis_group", self._group_id_for(req))
        return match

    def _request_prefill_peak(self, req: PendingRequest, handle: LogStateCacheHandle) -> int:
        prev = req.chunked_req
        return int(self._core["request_prefill_peak"](
            -1 if prev is None else prev.table_idx, handle.cached_len, req.input_len, req.output_len,
        ))

    def _acquire_prefill(
        self, req: PendingRequest, handle: BaseCacheHandle, chunk_size: int, admission: PrefillAdmission,
    ) -> str | None:
        assert isinstance(handle, LogStateCacheHandle)
        if req.chunked_req is not None:
            return None
        self._sync_clock()
        self.logstate_pool.poll_reclamation()
        peak = self._request_prefill_peak(req, handle)
        tr = handle.match_trace
        res = self._core["acquire_prefill"](
            req.uid, 0 if tr is None else tr.node_id, 0 if tr is None else tr.matched_len,
            True if tr is None else tr.is_exact, False if tr is None else tr.has_children,
            handle.node_id, handle.cached_len,
            -1 if handle.recovery.base_checkpoint_slot is None else handle.recovery.base_checkpoint_slot,
            handle.recovery_slots_t, admission.reserved + peak, True,
            [r.table_idx for r, _ in admission.active_decode if r.can_decode],
            [r.confirmed_decode_prefix_len for r, _ in admission.active_decode if r.can_decode],
            handle.analysis_group, handle.source_owner,
        )
        if res["admitted"]:
            object.__setattr__(handle, "node_id", int(res["node_id"]))
            object.__setattr__(handle, "analysis_owner", int(res["analysis_owner"]))
            return None
        return f"cache admission rejected: peak={peak}, reserved={admission.reserved}, matched={handle.cached_len}"

    def _account_prefill(self, req: PendingRequest, allocation: PrefillAllocation, admission: PrefillAdmission) -> None:
        assert isinstance(allocation.handle, LogStateCacheHandle)
        if req.chunked_req is None:
            admission.reserved += self._request_prefill_peak(req, allocation.handle) - (
                self._allocator.payload_frames + self._allocator.conv_frames
            )
        elif not any(cur.uid == req.uid for cur in admission.inflight_prefill):
            admission.reserved += self._request_prefill_peak(req, allocation.handle)

    def init_with_handle(self, table_idx: int, handle: BaseCacheHandle) -> None:
        assert isinstance(handle, LogStateCacheHandle)
        work_slot = self._allocate_work()
        assert self.work_state_table is not None and self.state_table is not None
        self.work_state_table[table_idx, 0] = work_slot
        if handle.cached_len > 0:
            self.page_table[table_idx, :handle.cached_len].copy_(handle.kv_indices.to(self.device, non_blocking=True))
        self.state_table[table_idx].fill_(-1)
        rec, tr = handle.recovery, handle.match_trace
        self._core["init_request"](
            table_idx, -1 if handle.query_uid is None else handle.query_uid, work_slot,
            handle.cached_len, handle.request_end, handle.node_id, handle.analysis_owner,
            list(handle.protected_endpoints), 0 if tr is None else tr.node_id,
            0 if tr is None else tr.matched_len, handle.trace_input_len, handle.trace_revision,
            -1 if rec.base_checkpoint_slot is None else rec.base_checkpoint_slot,
            -1 if rec.conv_slot is None else rec.conv_slot, -1 if rec.conv_granule is None else rec.conv_granule,
        )
        self._decode_state[table_idx] = [
            handle.cached_len, handle.cached_len, handle.cached_len,
            1 if handle.cached_len == 0 else 0, handle.request_end,
        ]

    def _allocate_work(self) -> int:
        pool = self.logstate_pool
        reserved = pool.reserve(works=1)
        while reserved is None:
            ev = self._core["plan_and_apply_eviction"](0, 0, 1, [], [], [])
            if not ev["ok"]:
                raise LogStateCapacityError(f"Insufficient evictable LogState capacity for work slot: free={self._allocator.free_frames}.")
            self._execute_eviction_gpu_cmds(ev)
            reserved = pool.reserve(works=1)
        return int(reserved[2][0])

    def _execute_eviction_gpu_cmds(self, ev) -> None:
        pool = self.logstate_pool
        if retired := torch.from_dlpack(ev["retired_slots"]).tolist():
            pool.retire_slots(retired)
        if ev["has_materialization"]:
            m = ev["mat_cmd"]
            base, absorb = int(m["base"]), bool(m["absorb"])
            slots = tuple(torch.from_dlpack(m["slots"]).tolist())
            logs = [(s, 0, l) for s, l in zip(torch.from_dlpack(m["log_slots"]).tolist(),
                                              torch.from_dlpack(m["log_lens"]).tolist(), strict=True)]
            pool.construct_checkpoint(None if base < 0 else base, logs, self._pressure_work_slot, slots, absorb=absorb)
            self._core["finish_pressure_materialization"](
                int(m["owner_nid"]), int(m["dest"]), int(m["begin"]), int(m["end"]), int(m["owner_ts"]),
            )
            pool.retire_slots(slots if absorb else slots[:-1], on_maint=True)
        if ev["has_active"]:
            a = ev["active_cmd"]
            sources = tuple(torch.from_dlpack(a["sources"]).tolist())
            log_slots = torch.from_dlpack(a["log_slots"]).tolist()
            if log_slots:
                base = int(a["base"])
                logs = [(s, 0, l) for s, l in zip(log_slots, torch.from_dlpack(a["log_lens"]).tolist(), strict=True)]
                pool.construct_checkpoint(None if base < 0 else base, logs, self._pressure_work_slot, (*sources, int(a["last_slot"])))
                assert self.state_table is not None
                self.state_table[int(a["table_idx"]), int(a["first_begin"]):int(a["last_end"])].fill_(-1)
            pool.retire_slots(sources, on_maint=bool(log_slots))

    def allocate(self, reqs: list[Request], *, phase: str | None = None) -> None:
        if not reqs:
            return
        ds = self._decode_state
        if phase == "decode" and all(
            (st := ds.get(r.table_idx)) is not None and r.device_len <= st[1] and r.max_device_len == st[4] and st[3] == 0
            for r in reqs
        ):
            return
        self._sync_clock()
        pool, rows = self.logstate_pool, [r.table_idx for r in reqs]
        if phase == "prefill":
            self._core["prepare_prefill_protections"](rows, [r.uid for r in reqs], [r.input_ids.contiguous() for r in reqs])

        dev_lens, max_lens, cached_lens = [r.device_len for r in reqs], [r.max_device_len for r in reqs], [r.cached_len for r in reqs]
        pt_width = self.page_table.shape[1]
        pool.poll_reclamation()
        res = self._core["try_allocate_batch"](rows, dev_lens, max_lens, cached_lens, pt_width)
        if fresh_works := torch.from_dlpack(res["fresh_work_slots"]).tolist():
            pool.zero_work_slots(fresh_works)
        for r in reqs:
            if (st := ds.get(r.table_idx)) is not None:
                st[3], st[4] = 0, r.max_device_len

        needed = int(res["needed"])
        if needed == 0:
            return
        dec_rows = rows if phase == "decode" else []
        dec_conf = [r.confirmed_decode_prefix_len for r in reqs] if phase == "decode" else []
        dec_can = [int(r.can_decode and r.extend_len == 1) for r in reqs] if phase == "decode" else []

        while not res["ok"]:
            if pool.wait_one_reclamation():
                res = self._core["try_allocate_batch"](rows, dev_lens, max_lens, cached_lens, pt_width)
                if res["ok"]:
                    break
            ev = self._core["plan_and_apply_eviction"](needed, needed, 0, dec_rows, dec_conf, dec_can)
            if not ev["ok"]:
                raise LogStateCapacityError(
                    f"Insufficient evictable LogState capacity: need={needed}, free={self._allocator.free_frames}."
                )
            self._execute_eviction_gpu_cmds(ev)
            pool.poll_reclamation()
            res = self._core["try_allocate_batch"](rows, dev_lens, max_lens, cached_lens, pt_width)

        pool.write_descriptors_packed(*(torch.from_dlpack(res[k]) for k in ("slot_indices", "payload", "conv", "decay")))
        from logssms.kernel.metadata import write_cache_mapping
        regions, ps = torch.from_dlpack(res["regions"]).tolist(), self.page_size
        for reg in regions:
            if (st := ds.get(reg[0])) is not None:
                st[0], st[1] = reg[1], ((reg[1] // ps) + 1) * ps
        write_cache_mapping(regions, self.page_table, self.state_table, ps)

    def prepare_batch(self, batch: Batch) -> None:
        batch.cache_metadata = LogStateCacheMetadata()
        ds = self._decode_state
        if not batch.is_prefill:
            if batch.is_decode:
                rw, fast_ok, counts = LOG_STATE_REPLAY_WINDOW, True, []
                for req in batch.reqs:
                    st, cl = ds.get(req.table_idx), req.cached_len
                    if st is None or not (st[0] <= cl < st[1]) or not (0 <= (cnt := cl - st[2]) < rw):
                        fast_ok = False
                        break
                    counts.append(cnt)
                if fast_ok:
                    for req in batch.reqs:
                        st, cl = ds[req.table_idx], req.cached_len
                        if (cl - st[0] + 1) % rw == 0:
                            st[2] = cl + 1
                    if (pad := batch.padded_size - len(counts)) > 0:
                        counts.extend([0] * pad)
                    batch.cache_metadata.replay_counts = counts
                else:
                    batch.cache_metadata.replay_counts = [
                        int(x) for x in self._core["prepare_decode_replay_counts"](
                            [r.table_idx for r in batch.reqs], [r.cached_len for r in batch.reqs], batch.padded_size, rw,
                        )
                    ]
                    for r in batch.reqs:
                        ds.pop(r.table_idx, None)
            return

        self._sync_clock()
        for req in batch.reqs:
            if (st := ds.get(req.table_idx)) is not None:
                st[2] = req.device_len
        rows = [r.table_idx for r in batch.reqs]
        res = self._core["prepare_prefill_batch"](
            rows, [r.cached_len for r in batch.reqs], [r.device_len for r in batch.reqs],
            [r.uid for r in batch.reqs], [r.input_ids.contiguous() for r in batch.reqs],
        )
        if res["has_first_offsets"]:
            batch.cache_metadata.first_offsets = [int(x) for x in res["first_offsets"]]

        pool, empty_i32 = self.logstate_pool, _prefix_cache_mod._EMPTY_I32
        for req in batch.reqs:
            h = req.cache_handle
            if isinstance(h, LogStateCacheHandle) and h.recovery_slots_t.numel():
                self._core["pin_recovery"](h.recovery_slots_t, True)
                object.__setattr__(h, "recovery_slots_t", empty_i32)

        mat_groups = [[int(s) for s in grp] for grp in res["materialize_groups"]]
        if mat_groups:
            pre_tasks = self._core["get_restore_tasks"](rows, False)
            works = list(dict.fromkeys(torch.from_dlpack(pre_tasks["work_slots"]).tolist()))
            for idx in range(0, len(mat_groups), max(1, len(works))) if works else ():
                wave = mat_groups[idx:idx + len(works)]
                chains = []
                for wi, grp in enumerate(wave):
                    ri = self._core["recovery_for_slot_end"](grp[-1])
                    base = int(ri["base"])
                    logs = [(s, 0, l) for s, l in zip(torch.from_dlpack(ri["log_slots"]).tolist(),
                                                      torch.from_dlpack(ri["log_lens"]).tolist(), strict=True)]
                    chains.append((None if base < 0 else base, logs, works[wi]))
                pool.kernels.restore(pool, chains)
                if accepted := [tuple(int(s) for s in g) for g in self._core["commit_hit_materializations"](wave)]:
                    acc_set = set(accepted)
                    pool.materialize_groups(accepted, [works[i] for i, g in enumerate(wave) if tuple(g) in acc_set])
                    pool.retire_slots([s for g in accepted for s in g[:-1]])

        ti = self._core["get_restore_tasks"](rows, True)
        trows, ws, bs, cs, cgs, fls, fll, lo, po = (
            torch.from_dlpack(ti[k]).tolist()
            for k in ("task_rows", "work_slots", "base_slots", "conv_slots", "conv_granules",
                      "log_slots", "log_lens", "log_offsets", "pin_offsets")
        )
        flat_pins = torch.from_dlpack(ti["pin_slots"])
        restores, row_to_pins = [], {}
        for i in range(len(ws)):
            logs = tuple(LogStateEntry(fls[j], LogStateKind.LOG, 0, fll[j]) for j in range(lo[i], lo[i + 1]))
            restores.append(RestoreTask(ws[i], None if bs[i] < 0 else bs[i], logs, cs[i], cgs[i], 0))
            if po[i + 1] > po[i]:
                row_to_pins[trows[i]] = flat_pins[po[i]:po[i + 1]].contiguous()
        self._construction.restore(restores)
        for req in batch.reqs:
            if (pins := row_to_pins.get(req.table_idx)) is not None and isinstance(req.cache_handle, LogStateCacheHandle):
                object.__setattr__(req.cache_handle, "recovery_slots_t", pins)

    def after_forward(self, batch: Batch) -> None:
        self.logstate_pool.poll_reclamation()

    def cache_req(self, req: Request, *, finished: bool) -> None:
        self._sync_clock()
        old = req.cache_handle
        assert isinstance(old, LogStateCacheHandle) and self.state_table is not None
        res = self._core["cache_req"](
            req.table_idx, req.uid, req.input_ids[:req.cached_len].contiguous(), req.cached_len,
            finished, old.node_id, old.cached_len, old.recovery_slots_t, old.analysis_owner,
            list(old.protected_endpoints), req.can_decode,
        )
        ins_len = int(res["inserted_len"])
        if ins_len > 0:
            self.state_table[req.table_idx, :ins_len].fill_(-1)
            new_kv = torch.from_dlpack(res["new_kv_indices"])
            self.page_table[req.table_idx, :ins_len].copy_(new_kv.to(self.device, non_blocking=True))
        else:
            new_kv = _prefix_cache_mod._EMPTY_I32
        if (conv_src := int(res["conv_copy_src"])) >= 0:
            self.logstate_pool.copy_conv_snapshot(conv_src, int(res["conv_copy_gran"]), int(res["conv_copy_dst"]))

        rel_kv = torch.from_dlpack(res["released_kv_frames"]).tolist()
        rel_slots = torch.from_dlpack(res["released_slots"]).tolist()
        if finished:
            self._decode_state.pop(req.table_idx, None)
            self.state_table[req.table_idx].fill_(-1)
            assert self.work_state_table is not None
            self.work_state_table[req.table_idx, 0] = -1
            self.record_cached_session(req, ins_len)
        else:
            bckpt, cs, cg = int(res["new_base_ckpt"]), int(res["new_conv_slot"]), int(res["new_conv_granule"])
            logs = tuple(
                LogStateEntry(s, LogStateKind.LOG, b, e) for s, b, e in zip(
                    torch.from_dlpack(res["new_log_slots"]).tolist(),
                    torch.from_dlpack(res["new_log_begins"]).tolist(),
                    torch.from_dlpack(res["new_log_ends"]).tolist(), strict=True,
                )
            )
            req.cache_handle = LogStateCacheHandle(
                cached_len=ins_len, node_id=int(res["new_node_id"]),
                recovery=LogStateRecovery(ins_len, None if bckpt < 0 else bckpt, logs, None if cs < 0 else cs, None if cg < 0 else cg),
                kv_indices=new_kv, analysis_owner=int(res["new_owner"]),
                protected_endpoints=tuple(torch.from_dlpack(res["new_prot_endpoints"]).tolist()),
            )
        if rel_kv or rel_slots:
            self._return_slots(rel_kv, rel_slots)

    def free_to_cached(self, req: Request) -> None:
        self._decode_state.pop(req.table_idx, None)
        res = self._core["free_to_cached"](req.table_idx, req.cached_len)
        assert self.state_table is not None
        ranges = torch.from_dlpack(res["clear_ranges"]).tolist()
        for i in range(0, len(ranges), 2):
            self.state_table[req.table_idx, ranges[i]:ranges[i + 1]].fill_(-1)
        rel_kv, rel_slots = torch.from_dlpack(res["released_kv_frames"]).tolist(), torch.from_dlpack(res["released_slots"]).tolist()
        if rel_kv or rel_slots:
            self._return_slots(rel_kv, rel_slots)
        req.device_len = min(req.cached_len + 1, req.max_device_len)

    def commit_speculative(self, req: Request, C: int, num_sampled: int) -> None:
        raise NotImplementedError("LogState speculative commit requires the LogState verify backend.")

    def check_integrity(self) -> None:
        self.logstate_pool.poll_reclamation()
        self._core["check_integrity"](list(self.logstate_pool.retiring_slots))

    @contextmanager
    def lazy_free_region(self):
        assert self._lazy_free_pages is None and self._lazy_free_states is None
        self._lazy_free_pages, self._lazy_free_states = [], []
        try:
            yield
        finally:
            pages, states = self._lazy_free_pages, self._lazy_free_states
            self._lazy_free_pages = self._lazy_free_states = None
            if pages or states:
                self._return_slots(pages, states)

    def _return_slots(self, kv_frames: Sequence[int], states: Sequence[int]) -> None:
        if self._lazy_free_pages is None:
            if kv_frames:
                self.logstate_pool.release_kv(list(kv_frames))
            if states:
                self.logstate_pool.release_slots(list(states))
        else:
            assert self._lazy_free_states is not None
            self._lazy_free_pages.extend(kv_frames)
            self._lazy_free_states.extend(states)

    def eviction_stats(self) -> dict[str, int]:
        st = self._core["eviction_stats"]()
        return {k: int(st[k]) for k in (
            "eviction_rounds", "pressure_dropped_checkpoints", "pressure_pruned_checkpoints",
            "evicted_kv_tokens", "released_state_slots_from_eviction",
        )}

    def layout_snapshot(self) -> dict:
        snap = self._core["tree_layout_snapshot"]()
        return {
            "checkpoint": int(snap["checkpoint"]), "log": int(snap["log"]),
            "checkpoint_spans": dict(Counter(str(x) for x in torch.from_dlpack(snap["checkpoint_spans"]).tolist())),
            "log_run_spans": dict(Counter(str(x) for x in torch.from_dlpack(snap["log_run_spans"]).tolist())),
        }

    def final_layout_snapshot(self) -> list[dict]:
        s = self._core["tree_nodes_summary"]()
        nodes, entries, keys = s["nodes"], s["entries"], s["prefix_keys"]
        return [{
            "prefix_sha256": hashlib.sha256(torch.from_dlpack(keys[i]).numpy().tobytes()).hexdigest(),
            "begin": int(nodes[i][3]), "end": int(nodes[i][4]), "protected": bool(int(nodes[i][5])),
            "entries": [
                {"begin": int(entries[i][j + 2]), "end": int(entries[i][j + 3]),
                 "kind": "checkpoint" if int(entries[i][j + 1]) == 1 else "log"}
                for j in range(0, len(entries[i]), 5)
            ],
        } for i in range(len(nodes))]


class LogStateCacheFactory(BaseCacheFactory):
    def plan(self, config: EngineConfig, available_memory: int, speculator_reserve: SpeculatorReserve | None) -> LogStateCachePlan:
        from logssms.cache.config import LogStateCacheConfig
        from logssms.cache.factory import make_linear_cache_spec
        from logssms.distributed import try_get_tp_info
        from logssms.models.config import AffineLogCacheSpec
        from logssms.utils import div_even

        assert isinstance(config.cache_config, LogStateCacheConfig) and config.model_config.is_hybrid
        assert speculator_reserve is None, "LogState speculative verify is not implemented yet."
        tp_info = try_get_tp_info()
        assert tp_info is not None
        cache_per_page = (
            2 * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, tp_info.size, allow_replicate=True)
            * config.page_size * config.dtype.itemsize * config.model_config.num_attention_layers
        )
        spec = make_linear_cache_spec(config.model_config)
        if not isinstance(spec, AffineLogCacheSpec):
            raise NotImplementedError(f"LogState storage for {spec.recurrent_type} is not implemented.")
        num_v_heads, head_k_dim, head_v_dim = spec.state_shape
        conv_dim, state_len = spec.conv_shape
        layout = LogStateSlotLayout(
            page_size=config.page_size, granularity=config.cache_config.granularity,
            num_linear_layers=spec.num_layers, num_key_heads=spec.num_groups, num_value_heads=num_v_heads,
            head_k_dim=head_k_dim, head_v_dim=head_v_dim, conv_dim=conv_dim, state_len=state_len, dtype=config.dtype,
        )
        num_work_states = config.max_running_req + 1
        reserved_work = num_work_states + 1
        payload_frames = div_ceil(spec.num_layers * num_v_heads * layout.head_payload_bytes(cache_per_page), cache_per_page)
        conv_frames = div_ceil(layout.conv_state_bytes // layout.num_granules, cache_per_page)
        assert layout.log_decay_bytes <= cache_per_page
        state_units = div_ceil(layout.log_payload_bytes, cache_per_page) + layout.num_granules * conv_frames + 1
        work_units = payload_frames + conv_frames
        num_pages = available_memory // cache_per_page - 1
        if config.num_page_override is not None:
            num_pages = min(num_pages, config.num_page_override)
        num_states = max(0, (num_pages - reserved_work * work_units - state_units) // state_units)
        assert num_pages > 1 and num_states > 1, "Not enough memory for LogState cache allocation."
        num_tokens = num_pages * config.page_size
        max_seq_len = min(
            config.max_seq_len, num_tokens,
            max(0, num_pages - reserved_work * work_units - state_units) // (1 + state_units) * config.page_size,
        )
        return LogStateCachePlan(
            linear_spec=spec, num_pages=num_pages, num_tokens=num_tokens, max_seq_len=max_seq_len,
            aligned_max_seq_len=align_ceil(max_seq_len, 32), num_states=num_states,
            num_work_states=num_work_states, state_units=state_units,
        )

    def allocate(self, config: EngineConfig, plan: CachePlan, device: torch.device, dtype: torch.dtype) -> CacheResources:
        from logssms.cache.config import LogStateCacheConfig
        from logssms.distributed import try_get_tp_info
        from logssms.utils import div_even

        assert isinstance(plan, LogStateCachePlan) and isinstance(config.cache_config, LogStateCacheConfig)
        model, spec, tp_info = config.model_config, plan.linear_spec, try_get_tp_info()
        assert tp_info is not None
        num_v_heads, head_k_dim, head_v_dim = spec.state_shape
        conv_dim, state_len = spec.conv_shape
        shared_pool = MHALogStatePool(
            num_frames=plan.num_pages + 1, page_size=config.page_size,
            num_attention_layers=model.num_attention_layers,
            num_kv_heads=div_even(model.num_kv_heads, tp_info.size, allow_replicate=True),
            head_dim=model.head_dim, num_linear_layers=spec.num_layers, num_key_heads=spec.num_groups,
            num_value_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
            conv_dim=conv_dim, state_len=state_len, device=device, dtype=dtype,
            granularity=config.cache_config.granularity, recurrent_type=spec.recurrent_type,
            deterministic_reclamation=tp_info.size > 1,
        )
        page_table = torch.zeros((config.max_running_req + 1, plan.aligned_max_seq_len), dtype=torch.int32, device=device)
        page_table[-1].fill_(plan.num_tokens)
        state_table = torch.full((config.max_running_req + 1, plan.aligned_max_seq_len), -1, dtype=torch.int32, device=device)
        work_state_table = torch.full((config.max_running_req + 1, 1), -1, dtype=torch.int32, device=device)
        logger.info(
            f"Allocating {plan.num_pages + 1} shared cache frames, "
            f"up to {plan.num_work_states} work states = {shared_pool.storage.numel() / (1 << 30):.2f} GiB"
        )
        return CacheResources(shared_pool, page_table, shared_pool, state_table, work_state_table)

    def make_manager(self, config: EngineConfig, plan: CachePlan, resources: CacheResources) -> LogStateCacheManager:
        from logssms.cache.config import LogStateCacheConfig
        assert isinstance(plan, LogStateCachePlan) and isinstance(config.cache_config, LogStateCacheConfig)
        cc = config.cache_config
        return LogStateCacheManager(
            plan.num_pages, plan.num_states, plan.num_work_states, config.page_size, resources,
            cc.granularity, cc.enable_merge_adjustment, cc.min_sampling_len,
            cc.max_density_exponent, cc.hit_half_life, cc.enable_hit_materialization,
        )


SUPPORTED_CACHE_FACTORIES.register("logstate")(LogStateCacheFactory())

__all__ = [
    "LogStateCacheManager", "LogStateCacheFactory", "LogStateCachePlan",
    "LogStateCapacityError", "LogStateConstruction", "RestoreTask",
]
