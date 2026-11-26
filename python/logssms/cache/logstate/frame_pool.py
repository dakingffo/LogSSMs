from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

import torch

from logssms.cache.logstate.allocator import LogStateAllocator, RetiredResources
from logssms.cache.logstate.layout import LogStateSlotLayout
from logssms.kernel.linear_attention import SUPPORTED_LOG_STATE_KERNELS
from logssms.utils.torch_utils import make_device_vectors


class MHALogStatePool:
    def __init__(
        self, *, num_frames: int, page_size: int, granularity: int,
        num_attention_layers: int, num_kv_heads: int, head_dim: int,
        num_linear_layers: int, num_key_heads: int, num_value_heads: int,
        head_k_dim: int, head_v_dim: int, conv_dim: int, state_len: int,
        device: torch.device, dtype: torch.dtype, recurrent_type: str = "gdn",
        deterministic_reclamation: bool = False,
    ) -> None:
        layout = LogStateSlotLayout(
            page_size=page_size, granularity=granularity,
            num_linear_layers=num_linear_layers, num_key_heads=num_key_heads,
            num_value_heads=num_value_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
            conv_dim=conv_dim, state_len=state_len, dtype=dtype,
        )
        frame_bytes = 2 * num_attention_layers * page_size * num_kv_heads * head_dim * dtype.itemsize
        head_payload_bytes = layout.head_payload_bytes(frame_bytes)
        allocator = LogStateAllocator(
            num_frames=num_frames, frame_bytes=frame_bytes,
            payload_bytes=num_linear_layers * num_value_heads * head_payload_bytes,
            conv_snapshot_bytes=layout.conv_state_bytes // layout.num_granules,
            decay_bytes=layout.log_decay_bytes, num_granules=layout.num_granules,
            log_payload_bytes=layout.log_payload_bytes,
        )
        self.layout, self.kernels, self.allocator = layout, SUPPORTED_LOG_STATE_KERNELS[recurrent_type], allocator
        self.shared_keys = num_key_heads < num_value_heads
        self._dtype, self.page_size, self.kv_page_size = dtype, page_size, granularity
        self.frame_bytes, self.head_payload_bytes = frame_bytes, head_payload_bytes
        self._k_heads, self._head_dim, self._num_value_heads = num_kv_heads, head_dim, num_value_heads

        self.storage = torch.empty((num_frames, frame_bytes), dtype=torch.uint8, device=device)
        self._device = self.storage.device
        self.payload_frame_table = torch.full((num_frames, allocator.descriptor_frames), -1, dtype=torch.int32, device=device)
        self.conv_frame_table = torch.full((num_frames, layout.num_granules, allocator.conv_frames), -1, dtype=torch.int32, device=device)
        self.decay_frame_table = torch.full((num_frames, 2), -1, dtype=torch.int32, device=device)

        layer_bytes = 2 * granularity * num_kv_heads * head_dim * dtype.itemsize
        k_bytes = granularity * num_kv_heads * head_dim * dtype.itemsize
        assert frame_bytes // layout.num_granules == layer_bytes * num_attention_layers
        self._k_buffer = [self._kv_view(l * layer_bytes) for l in range(num_attention_layers)]
        self._v_buffer = [self._kv_view(l * layer_bytes + k_bytes) for l in range(num_attention_layers)]

        self._retired_releases: list[tuple[torch.cuda.Event, tuple[int, ...]]] = []
        self._retiring_slots: set[int] = set()
        self._retired_rebindings: list[tuple[torch.cuda.Event, tuple[RetiredResources, ...]]] = []
        self.deterministic_reclamation = deterministic_reclamation

        is_cuda = self._device.type == "cuda"
        self._maint_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self._device) if is_cuda else None
        self._event_pool: list[torch.cuda.Event] = []
        self._aux_events: list[torch.cuda.Event] = []
        self._slot_ready_event: dict[int, torch.cuda.Event] = {}
        self._desc_ring: list[list[object]] = [[None, None, torch.cuda.Event(), False] for _ in range(8)] if is_cuda else []
        self._rel_ring: list[list[object]] = [[None, None, torch.cuda.Event(), False] for _ in range(8)] if is_cuda else []
        self._desc_ring_idx = self._rel_ring_idx = 0

    @property
    def device(self) -> torch.device: return self._device
    @property
    def dtype(self) -> torch.dtype: return self._dtype
    @property
    def num_layers(self) -> int: return len(self._k_buffer)
    @property
    def retiring_slots(self) -> frozenset[int]: return frozenset(self._retiring_slots)
    def k_cache(self, layer: int) -> torch.Tensor: return self._k_buffer[layer]
    def v_cache(self, layer: int) -> torch.Tensor: return self._v_buffer[layer]

    def _acquire_event(self) -> torch.cuda.Event:
        return self._event_pool.pop() if (self._maint_stream is not None and self._event_pool) else torch.cuda.Event()

    def _recycle_event(self, ev: torch.cuda.Event) -> None:
        if self._maint_stream is not None and len(self._event_pool) < 64:
            self._event_pool.append(ev)

    def wait_slots_on_current_stream(self, slots: Sequence[int]) -> None:
        ready = self._slot_ready_event
        if not ready or self._maint_stream is None:
            return
        cur = torch.cuda.current_stream(self._device)
        for s in slots:
            ev = ready.get(s)
            if ev is None:
                continue
            if ev.query():
                ready.pop(s, None)
                self._recycle_event(ev)
            else:
                cur.wait_event(ev)

    def _kv_view(self, byte_offset: int) -> torch.Tensor:
        sz = self.dtype.itemsize
        return torch.as_strided(
            self.storage.view(self.dtype),
            size=(self.allocator.num_frames * self.layout.num_granules, self.kv_page_size, self._k_heads, self._head_dim),
            stride=(self.frame_bytes // self.layout.num_granules // sz, self._k_heads * self._head_dim, self._head_dim, 1),
            storage_offset=byte_offset // sz,
        )

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        pages = torch.div(out_loc, self.kv_page_size, rounding_mode="floor").long()
        offsets = (out_loc % self.kv_page_size).long()
        shape = (out_loc.numel(), self._k_heads, self._head_dim)
        self._k_buffer[layer_id][pages, offsets] = k.reshape(shape)
        self._v_buffer[layer_id][pages, offsets] = v.reshape(shape)

    def conv_view(self, slot: int, granule: int) -> torch.Tensor:
        if self._slot_ready_event:
            self.wait_slots_on_current_stream((slot,))
        rec = self.allocator.slot(slot)
        if rec.kind == "work":
            assert granule == 0
            idx = 0
        elif rec.kind == "checkpoint":
            assert granule == self.layout.num_granules - 1
            idx = 0
        else:
            assert 0 <= granule < self.layout.num_granules
            idx = granule
        frames = rec.conv[idx]
        if len(frames) != 1:
            raise ValueError("The convolution snapshot spans multiple frames")
        size = self.layout.conv_state_bytes // self.layout.num_granules
        return self.storage[frames[0], :size].view(self.dtype).view(
            self.layout.num_linear_layers, self.layout.conv_dim, self.layout.state_len,
        )

    def _state_indices(self, slots: torch.Tensor, layer: int) -> torch.Tensor:
        slots = slots.to(torch.int64)
        heads = torch.arange(self.layout.num_value_heads, dtype=torch.int64, device=self.device)
        offs = torch.arange(self.layout.head_k_dim * self.layout.head_v_dim, dtype=torch.int64, device=self.device)
        fe = self.frame_bytes // self.dtype.itemsize
        if self.head_payload_bytes <= self.frame_bytes:
            logical = (layer * self.layout.num_value_heads + heads) * self.head_payload_bytes
            frames = self.payload_frame_table[slots[:, None], (logical // self.frame_bytes)[None, :]].to(torch.int64)
            return (frames * fe + (logical % self.frame_bytes)[None, :] // self.dtype.itemsize)[:, :, None] + offs
        logical = (layer * self.layout.num_value_heads + heads[:, None]) * self.head_payload_bytes + offs[None, :] * self.dtype.itemsize
        frames = self.payload_frame_table[slots[:, None, None], logical[None, :, :] // self.frame_bytes].long()
        return frames * fe + logical[None, :, :] % self.frame_bytes // self.dtype.itemsize

    def read_states(self, slots: torch.Tensor, layer: int) -> torch.Tensor:
        if self._slot_ready_event:
            self.wait_slots_on_current_stream(slots.tolist())
        return self.storage.view(self.dtype).flatten()[self._state_indices(slots, layer)].view(
            len(slots), self.layout.num_value_heads, self.layout.head_k_dim, self.layout.head_v_dim,
        )

    def write_states(self, slots: torch.Tensor, layer: int, values: torch.Tensor) -> None:
        assert values.shape == (len(slots), self.layout.num_value_heads, self.layout.head_k_dim, self.layout.head_v_dim)
        if self.device.type == "cuda":
            from logssms.kernel.linear_attention.masked_store import store_states
            store_states(self, slots.to(self.device, non_blocking=True), layer, values)
            return
        active = slots >= 0
        self.storage.view(self.dtype).flatten()[self._state_indices(slots[active], layer)] = values[active].to(self.dtype).flatten(2)

    def zero_state(self, slot: int) -> None:
        slots = torch.tensor([slot], dtype=torch.int64, device=self.device)
        vals = torch.zeros((1, self.layout.num_value_heads, self.layout.head_k_dim, self.layout.head_v_dim), dtype=self.dtype, device=self.device)
        for layer in range(self.layout.num_linear_layers):
            self.write_states(slots, layer, vals)

    def _conv_indices(self, slots: torch.Tensor, granules: torch.Tensor, layer: int) -> torch.Tensor:
        offs = torch.arange(self.layout.conv_dim * self.layout.state_len, dtype=torch.int64, device=self.device)
        fe = self.frame_bytes // self.dtype.itemsize
        if self.allocator.conv_frames == 1:
            frames = self.conv_frame_table[slots.to(torch.int64), granules.to(torch.int64), 0].to(torch.int64)
            return (frames * fe + layer * self.layout.conv_dim * self.layout.state_len)[:, None] + offs[None, :]
        logical = layer * self.layout.conv_dim * self.layout.state_len + offs
        frames = self.conv_frame_table[slots.long()[:, None], granules.long()[:, None], logical[None, :] // fe].long()
        return frames * fe + logical[None, :] % fe

    def read_conv(self, slots: torch.Tensor, granules: torch.Tensor, layer: int) -> torch.Tensor:
        if self._slot_ready_event:
            self.wait_slots_on_current_stream(slots.tolist())
        return self.storage.view(self.dtype).flatten()[self._conv_indices(slots, granules, layer)].view(
            len(slots), self.layout.conv_dim, self.layout.state_len,
        )

    def write_conv(self, slots: torch.Tensor, granules: torch.Tensor, layer: int, values: torch.Tensor) -> None:
        assert values.shape == (len(slots), self.layout.conv_dim, self.layout.state_len)
        if self.device.type == "cuda":
            from logssms.kernel.linear_attention.masked_store import store_conv
            store_conv(self, slots.to(self.device, non_blocking=True), granules.to(self.device, non_blocking=True), layer, values)
            return
        active = slots >= 0
        self.storage.view(self.dtype).flatten()[self._conv_indices(slots[active], granules[active], layer)] = values[active].to(self.dtype).flatten(1)

    def copy_conv_snapshot(self, src_slot: int, src_granule: int, dst_slot: int, dst_granule: int = 0) -> None:
        if self._slot_ready_event:
            self.wait_slots_on_current_stream((src_slot,))
        if self.allocator.conv_frames == 1:
            self.conv_view(dst_slot, dst_granule).copy_(self.conv_view(src_slot, src_granule))
            return
        if self.device.type == "cuda":
            from logssms.kernel.linear_attention.conv_copy import copy_conv_snapshots
            copy_conv_snapshots(self, *make_device_vectors([[src_slot], [src_granule], [dst_slot], [dst_granule]], self.device))
            return
        for layer in range(self.layout.num_linear_layers):
            src = self.read_conv(torch.tensor([src_slot]), torch.tensor([src_granule]), layer)
            self.write_conv(torch.tensor([dst_slot]), torch.tensor([dst_granule]), layer, src)

    def zero_work_conv(self, slot: int) -> None:
        slots, grans = torch.tensor([slot], dtype=torch.int64, device=self.device), torch.zeros(1, dtype=torch.int64, device=self.device)
        vals = torch.zeros((1, self.layout.conv_dim, self.layout.state_len), dtype=self.dtype, device=self.device)
        for layer in range(self.layout.num_linear_layers):
            self.write_conv(slots, grans, layer, vals)

    def zero_work_slots(self, slots: Sequence[int]) -> None:
        frames: list[int] = []
        for slot in slots:
            rec = self.allocator.slot(slot)
            assert rec.kind == "work" and rec.decay is None
            frames.extend(rec.payload)
            for c in rec.conv:
                frames.extend(c)
        if frames:
            self.storage.index_fill_(0, torch.tensor(frames, dtype=torch.int64, device=self.device), 0)

    def _write_descriptors(self, slots: Sequence[int]) -> None:
        if slots:
            self.write_descriptors_packed(*self.allocator.pack_descriptors(list(slots)))

    def write_descriptors_packed(
        self, indices: torch.Tensor, payload: torch.Tensor, conv: torch.Tensor, decay: torch.Tensor,
    ) -> None:
        if not indices.numel():
            return
        if self.device.type == "cuda":
            from logssms.kernel.metadata import write_slot_descriptors
            if self._maint_stream is not None:
                n, p_n, c_n, d_n = indices.numel(), payload.numel(), conv.numel(), decay.numel()
                total = n + p_n + c_n + d_n
                buf = self._desc_ring[self._desc_ring_idx]
                self._desc_ring_idx = (self._desc_ring_idx + 1) & 7
                pin, dev, ev, recorded = buf
                if recorded and not ev.query():
                    ev.synchronize()
                if pin is None or pin.numel() < total:
                    cap = max(total, 4096)
                    with torch.inference_mode(False):
                        pin = torch.empty(cap, dtype=torch.int32, pin_memory=True)
                        dev = torch.empty(cap, dtype=torch.int32, device=self._device)
                    buf[0], buf[1] = pin, dev
                o1, o2, o3 = n, n + p_n, n + p_n + c_n
                pin[:o1].copy_(indices); pin[o1:o2].copy_(payload); pin[o2:o3].copy_(conv); pin[o3:total].copy_(decay)
                dev[:total].copy_(pin[:total], non_blocking=True)
                write_slot_descriptors(
                    dev[:o1], dev[o1:o2], dev[o2:o3], dev[o3:total],
                    self.payload_frame_table, self.conv_frame_table, self.decay_frame_table,
                )
                ev.record(torch.cuda.current_stream(self._device))
                buf[3] = True
                return
            write_slot_descriptors(
                indices.to(self.device, non_blocking=True), payload.to(self.device, non_blocking=True),
                conv.to(self.device, non_blocking=True), decay.to(self.device, non_blocking=True),
                self.payload_frame_table, self.conv_frame_table, self.decay_frame_table,
            )
            return
        n, idx_l = indices.numel(), indices.to(device=self.device, dtype=torch.int64)
        self.payload_frame_table.index_copy_(0, idx_l, payload.to(self.device).reshape(n, -1))
        self.conv_frame_table.index_copy_(0, idx_l, conv.to(self.device).reshape(n, self.layout.num_granules, self.allocator.conv_frames))
        self.decay_frame_table.index_copy_(0, idx_l, decay.to(self.device).reshape(n, 2))

    def wait_one_reclamation(self) -> bool:
        if self._retired_rebindings:
            ev = self._retired_rebindings[0][0]
        elif self._retired_releases:
            ev = self._retired_releases[0][0]
        else:
            return False
        ev.synchronize()
        self.poll_reclamation(completed_event=ev if self.deterministic_reclamation else None)
        return True

    def reserve(
        self, *, kv: int = 0, logs: int = 0, works: int = 0, timings: dict[str, float] | None = None,
    ) -> tuple[list[int], list[int], list[int]] | None:
        t0 = perf_counter_ns() if timings is not None else 0
        self.poll_reclamation(timings)
        if timings is not None:
            t1 = perf_counter_ns(); timings["poll_reclamation_ms"] = (t1 - t0) / 1e6; t0 = t1
        allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
        if timings is not None:
            t1 = perf_counter_ns(); timings["allocator_reserve_ms"] = (t1 - t0) / 1e6; t0 = t1
        while allocated is None and (self._retired_releases or self._retired_rebindings):
            if timings is not None:
                timings.setdefault("waited_merge_slots", 0)
            if self._retired_rebindings:
                ev = self._retired_rebindings[0][0]
            else:
                ev, slots = self._retired_releases[0]
                if timings is not None:
                    timings["waited_merge_slots"] += len(slots)
            ev.synchronize()
            if timings is not None:
                t1 = perf_counter_ns(); timings["wait_materialization_ms"] = timings.get("wait_materialization_ms", 0.0) + (t1 - t0) / 1e6; t0 = t1
            self.poll_reclamation(timings, completed_event=ev if self.deterministic_reclamation else None)
            if timings is not None:
                t1 = perf_counter_ns(); timings["retry_poll_reclamation_ms"] = timings.get("retry_poll_reclamation_ms", 0.0) + (t1 - t0) / 1e6; t0 = t1
            allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
            if timings is not None:
                t1 = perf_counter_ns(); timings["retry_allocator_reserve_ms"] = timings.get("retry_allocator_reserve_ms", 0.0) + (t1 - t0) / 1e6; t0 = t1
        if allocated is None:
            if timings is not None:
                timings["failed"] = 1
            return None
        self._write_descriptors(allocated[1] + allocated[2])
        if timings is not None:
            timings["descriptor_write_ms"] = (perf_counter_ns() - t0) / 1e6
            timings["descriptor_count"] = len(allocated[1]) + len(allocated[2])
        return allocated

    def copy_states(self, sources: Sequence[int], destinations: Sequence[int]) -> None:
        from logssms.kernel.linear_attention.logstate import copy_states
        if self._slot_ready_event and (self._maint_stream is None or torch.cuda.current_stream(self._device) != self._maint_stream):
            self.wait_slots_on_current_stream(sources)
        copy_states(self, sources, destinations)

    def construct_checkpoint(
        self, base: int | None, logs: Sequence[tuple[int, int, int]], work: int,
        slots: tuple[int, ...], *, absorb: bool = False,
    ) -> None:
        if self._maint_stream is not None:
            ms = self._maint_stream
            dep_ev = self._acquire_event()
            dep_ev.record(torch.cuda.current_stream(self._device))
            ms.wait_event(dep_ev)
            self._aux_events.append(dep_ev)
            with torch.cuda.stream(ms):
                self.kernels.restore(self, [(base, logs, work)])
                if absorb:
                    assert base is not None
                    self.copy_states([work], [base])
                    self.copy_conv_snapshot(slots[-1], 0, base)
                    dest = base
                else:
                    self._materialize_groups_on_stream([slots], [work], ms)
                    dest = slots[-1]
                done_ev = self._acquire_event()
                done_ev.record(ms)
                if (old_ev := self._slot_ready_event.get(dest)) is not None:
                    self._aux_events.append(old_ev)
                self._slot_ready_event[dest] = done_ev
            return
        self.kernels.restore(self, [(base, logs, work)])
        if absorb:
            assert base is not None
            self.copy_states([work], [base])
            self.copy_conv_snapshot(slots[-1], 0, base)
        else:
            self.materialize_groups([slots], [work])

    def _materialize_groups_on_stream(
        self, groups: Sequence[tuple[int, ...]], works: Sequence[int], stream: torch.cuda.Stream,
    ) -> None:
        from logssms.kernel.linear_attention.logstate import copy_states
        rets = [self.allocator.materialize_group(s) for s in groups]
        dests = [s[-1] for s in groups]
        self._write_descriptors(dests)
        copy_states(self, works, dests)
        ev = self._acquire_event()
        ev.record(stream)
        self._retired_rebindings.append((ev, tuple(rets)))

    def materialize_groups(self, groups: Sequence[tuple[int, ...]], works: Sequence[int]) -> None:
        from logssms.kernel.linear_attention.logstate import copy_states
        assert len(groups) == len(works)
        if not groups:
            return
        rets = [self.allocator.materialize_group(s) for s in groups]
        dests = [s[-1] for s in groups]
        self._write_descriptors(dests)
        copy_states(self, works, dests)
        if self.device.type == "cuda":
            ev = self._acquire_event()
            ev.record(torch.cuda.current_stream(self.device))
            self._retired_rebindings.append((ev, tuple(rets)))
        else:
            for ret in rets:
                self.allocator.release_retirement(ret)

    def retire_slots(self, slots: Sequence[int], *, on_maint: bool = False) -> None:
        if not slots:
            return
        assert len(slots) == len(set(slots)) and not self._retiring_slots.intersection(slots)
        if self.device.type != "cuda":
            self.release_slots(slots)
            return
        ms = self._maint_stream
        if ms is not None:
            has_maint = on_maint
            for s in slots:
                if (old_ev := self._slot_ready_event.pop(s, None)) is not None:
                    has_maint = True
                    self._aux_events.append(old_ev)
            if has_maint:
                dep_ev = self._acquire_event()
                dep_ev.record(torch.cuda.current_stream(self._device))
                ms.wait_event(dep_ev)
                self._aux_events.append(dep_ev)
                ev = self._acquire_event()
                ev.record(ms)
            else:
                ev = self._acquire_event()
                ev.record(torch.cuda.current_stream(self._device))
        else:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
        self._retiring_slots.update(slots)
        for s in slots:
            self.allocator.set_reclaimable(s, True)
        self._retired_releases.append((ev, tuple(slots)))

    def poll_reclamation(
        self, timings: dict[str, float] | None = None, *, completed_event: torch.cuda.Event | None = None,
    ) -> None:
        if completed_event is None and (
            self.deterministic_reclamation
            or (not self._retired_rebindings and not self._retired_releases and not self._aux_events)
        ):
            return
        if completed_event is not None:
            assert any(q and q[0][0] is completed_event for q in (self._retired_rebindings, self._retired_releases)), (
                "Reclamation must complete the selected queue head"
            )
        if completed_event is None and self._aux_events:
            still = []
            for ev in self._aux_events:
                if ev.query():
                    self._recycle_event(ev)
                else:
                    still.append(ev)
            self._aux_events = still

        ready = lambda ev: completed_event is not None or ev.query()
        select = lambda q: (q, []) if completed_event is None else (([q.pop(0)], q) if q and q[0][0] is completed_event else ((), q))

        b_ev, rem_b = select(self._retired_rebindings)
        for ev, recs in b_ev:
            if ready(ev):
                for rec in recs:
                    self.allocator.release_retirement(rec)
                self._recycle_event(ev)
            else:
                rem_b.append((ev, recs))
        self._retired_rebindings = rem_b

        r_ev, rem_r = select(self._retired_releases)
        for ev, slots in r_ev:
            if ready(ev):
                self._release_slots(slots, timings=timings)
                self._retiring_slots.difference_update(slots)
                self._recycle_event(ev)
                if timings is not None:
                    timings["merge_reclamation_count"] = timings.get("merge_reclamation_count", 0) + len(slots)
            else:
                rem_r.append((ev, slots))
        self._retired_releases = rem_r

    def release_kv(self, frames: list[int] | tuple[int, ...]) -> None:
        self.allocator.release_kv(frames)

    def release_slots(self, slots: Sequence[int], timings: dict[str, float] | None = None) -> None:
        if not slots:
            return
        assert len(slots) == len(set(slots)) and not self._retiring_slots.intersection(slots), "Slot already queued for reclamation"
        t0 = perf_counter_ns() if timings is not None else 0
        self.poll_reclamation(timings)
        if timings is not None:
            timings["poll_reclamation_ms"] = timings.get("poll_reclamation_ms", 0.0) + (perf_counter_ns() - t0) / 1e6
        self._release_slots(slots, timings=timings)

    def _release_slots(self, slots: Sequence[int], timings: dict[str, float] | None = None) -> None:
        if self._slot_ready_event:
            for s in slots:
                if (ev := self._slot_ready_event.pop(s, None)) is not None:
                    if not ev.query():
                        ev.synchronize()
                    self._recycle_event(ev)
        t0 = perf_counter_ns() if timings is not None else 0
        self.allocator.release_slots(list(slots))
        if timings is not None:
            t1 = perf_counter_ns(); timings["allocator_release_ms"] = timings.get("allocator_release_ms", 0.0) + (t1 - t0) / 1e6; t0 = t1
        if self._maint_stream is not None:
            from logssms.kernel.metadata import write_slot_descriptors
            n = len(slots)
            buf = self._rel_ring[self._rel_ring_idx]
            self._rel_ring_idx = (self._rel_ring_idx + 1) & 7
            pin, dev, ev, recorded = buf
            if recorded and not ev.query():
                ev.synchronize()
            if pin is None or pin.numel() < n:
                cap = max(n, 1024)
                with torch.inference_mode(False):
                    pin = torch.empty(cap, dtype=torch.int64, pin_memory=True)
                    dev = torch.empty(cap, dtype=torch.int64, device=self._device)
                buf[0], buf[1] = pin, dev
            pin[:n].copy_(torch.tensor(slots, dtype=torch.int64))
            dev[:n].copy_(pin[:n], non_blocking=True)
            write_slot_descriptors(dev[:n], None, None, None, self.payload_frame_table, self.conv_frame_table, self.decay_frame_table)
            ev.record(torch.cuda.current_stream(self._device))
            buf[3] = True
        elif self.device.type == "cuda":
            from logssms.kernel.metadata import write_slot_descriptors
            write_slot_descriptors(make_device_vectors([slots], self.device, dtype=torch.int64)[0], None, None, None,
                                   self.payload_frame_table, self.conv_frame_table, self.decay_frame_table)
        else:
            idx, = make_device_vectors([slots], self.device, dtype=torch.int64)
            self.payload_frame_table.index_fill_(0, idx, -1)
            self.conv_frame_table.index_fill_(0, idx, -1)
            self.decay_frame_table.index_fill_(0, idx, -1)
        if timings is not None:
            timings["descriptor_clear_ms"] = timings.get("descriptor_clear_ms", 0.0) + (perf_counter_ns() - t0) / 1e6
