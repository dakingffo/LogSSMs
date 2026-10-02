from __future__ import annotations

from time import perf_counter_ns
from collections.abc import Sequence

import torch

from logssms.kernel.linear_attention import SUPPORTED_LOG_STATE_KERNELS
from logssms.utils.torch_utils import make_device_vectors
from .frame_allocator import LogStateFrameAllocator
from .layout import LogStateSlotLayout
from ..kv_pool.base import BaseKVCachePool


class FramedMHALogStatePool(BaseKVCachePool):
    """Fixed-size physical frames with stable, device-visible slot maps.

    The KV view preserves the existing 16/32/64-token FlashInfer page layout:
    a physical frame contains all KV layers for one 64-token page, subdivided
    into ``page_size / granularity`` independently addressable KV granules.
    Linear payloads need not occupy adjacent frames. Their kernels must resolve
    addresses through the descriptor tables rather than a fixed slot stride.
    """

    def __init__(
        self,
        *,
        num_frames          : int,
        page_size           : int,
        granularity         : int,
        num_attention_layers: int,
        num_kv_heads        : int,
        head_dim            : int,
        num_linear_layers  : int,
        num_key_heads      : int,
        num_value_heads    : int,
        head_k_dim          : int,
        head_v_dim          : int,
        conv_dim            : int,
        state_len           : int,
        device              : torch.device,
        dtype               : torch.dtype,
        recurrent_type      : str = "gdn",
    ) -> None:
        layout = LogStateSlotLayout(
            page_size=page_size,
            granularity=granularity,
            num_linear_layers=num_linear_layers,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            state_len=state_len,
            dtype=dtype,
        )
        frame_bytes = (
            2 * num_attention_layers * page_size * num_kv_heads
            * head_dim * dtype.itemsize
        )
        head_payload_bytes = max(
            layout.head_log_key_bytes + layout.head_log_value_bytes,
            layout.head_recurrent_state_bytes,
        )
        payload_bytes = num_linear_layers * num_value_heads * head_payload_bytes
        conv_snapshot_bytes = layout.conv_state_bytes // layout.num_granules
        allocator = LogStateFrameAllocator(
            num_frames=num_frames,
            frame_bytes=frame_bytes,
            payload_bytes=payload_bytes,
            conv_snapshot_bytes=conv_snapshot_bytes,
            decay_bytes=layout.log_decay_bytes,
            num_granules=layout.num_granules,
        )
        self.layout = layout
        self.kernels = SUPPORTED_LOG_STATE_KERNELS[recurrent_type]
        self.allocator = allocator
        self._dtype = dtype
        self.page_size = page_size
        self.kv_page_size = granularity
        self.frame_bytes = frame_bytes
        self.head_payload_bytes = head_payload_bytes
        self._k_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_value_heads = num_value_heads
        self.storage = torch.empty(
            (num_frames, frame_bytes), dtype=torch.uint8, device=device
        )
        self._device = self.storage.device
        self.payload_frame_table = torch.full(
            (num_frames, allocator.payload_frames), -1,
            dtype=torch.int32, device=device,
        )
        self.conv_frame_table = torch.full(
            (num_frames, layout.num_granules, allocator.conv_frames), -1,
            dtype=torch.int32, device=device,
        )
        self.decay_frame_table = torch.full(
            (num_frames, 2), -1, dtype=torch.int32, device=device,
        )
        kv_granule_bytes = frame_bytes // layout.num_granules
        layer_bytes = (
            2 * granularity * num_kv_heads * head_dim * dtype.itemsize
        )
        k_bytes = granularity * num_kv_heads * head_dim * dtype.itemsize
        assert kv_granule_bytes == layer_bytes * num_attention_layers
        self._k_buffer = [
            self._kv_view(layer * layer_bytes)
            for layer in range(num_attention_layers)
        ]
        self._v_buffer = [
            self._kv_view(layer * layer_bytes + k_bytes)
            for layer in range(num_attention_layers)
        ]
        self._retired_materializations: list[tuple[torch.cuda.Event, tuple[int, ...]]] = []
        self._retired_releases: list[tuple[torch.cuda.Event, tuple[int, ...]]] = []
        self._retiring_slots: set[int] = set()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _kv_view(self, byte_offset: int) -> torch.Tensor:
        itemsize = self.dtype.itemsize
        return torch.as_strided(
            self.storage.view(self.dtype),
            size=(
                self.allocator.num_frames * self.layout.num_granules,
                self.kv_page_size,
                self._k_heads,
                self._head_dim,
            ),
            stride=(
                self.frame_bytes // self.layout.num_granules // itemsize,
                self._k_heads * self._head_dim,
                self._head_dim,
                1,
            ),
            storage_offset=byte_offset // itemsize,
        )

    def k_cache(self, layer: int) -> torch.Tensor:
        return self._k_buffer[layer]

    def v_cache(self, layer: int) -> torch.Tensor:
        return self._v_buffer[layer]

    @property
    def num_layers(self) -> int:
        return len(self._k_buffer)

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor,
        out_loc: torch.Tensor, layer_id: int,
    ) -> None:
        pages = torch.div(out_loc, self.kv_page_size, rounding_mode="floor").long()
        offsets = (out_loc % self.kv_page_size).long()
        shape = (out_loc.numel(), self._k_heads, self._head_dim)
        self._k_buffer[layer_id][pages, offsets] = k.reshape(shape)
        self._v_buffer[layer_id][pages, offsets] = v.reshape(shape)

    def conv_view(self, slot: int, granule: int) -> torch.Tensor:
        record = self.allocator.slot(slot)
        if record.kind == "work":
            assert granule == 0
            index = 0
        elif record.kind == "checkpoint":
            assert granule == self.layout.num_granules - 1
            index = 0
        else:
            assert 0 <= granule < self.layout.num_granules
            index = granule
        frames = record.conv[index]
        if len(frames) != 1:
            raise ValueError("The convolution snapshot spans multiple frames")
        size = self.layout.conv_state_bytes // self.layout.num_granules
        return self.storage[frames[0], :size].view(self.dtype).view(
            self.layout.num_linear_layers,
            self.layout.conv_dim,
            self.layout.state_len,
        )

    def decay_view(self, slot: int) -> torch.Tensor:
        """FP32 logarithmic decay g, where alpha = exp(g)."""
        record = self.allocator.slot(slot)
        assert record.decay is not None
        size = self.layout.log_decay_bytes
        begin = record.decay.index * size
        return self.storage[
            record.decay.frame, begin:begin + size
        ].view(torch.float32).view(
            self.layout.num_linear_layers,
            self.page_size,
            self.layout.num_value_heads,
        )

    def _state_indices(self, slots: torch.Tensor, layer: int) -> torch.Tensor:
        assert self.frame_bytes % self.head_payload_bytes == 0
        slots = slots.to(torch.int64)
        heads = torch.arange(
            self.layout.num_value_heads, dtype=torch.int64, device=self.device
        )
        logical = (layer * self.layout.num_value_heads + heads) * self.head_payload_bytes
        frames = self.payload_frame_table[
            slots[:, None], (logical // self.frame_bytes)[None, :]
        ].to(torch.int64)
        base = (
            frames * (self.frame_bytes // self.dtype.itemsize)
            + (logical % self.frame_bytes)[None, :] // self.dtype.itemsize
        )
        offsets = torch.arange(
            self.layout.head_k_dim * self.layout.head_v_dim,
            dtype=torch.int64, device=self.device,
        )
        return base[:, :, None] + offsets[None, None, :]

    def read_states(self, slots: torch.Tensor, layer: int) -> torch.Tensor:
        return self.storage.view(self.dtype).flatten()[self._state_indices(slots, layer)].view(
            len(slots), self.layout.num_value_heads,
            self.layout.head_k_dim, self.layout.head_v_dim,
        )

    def write_states(
        self, slots: torch.Tensor, layer: int, values: torch.Tensor
    ) -> None:
        assert values.shape == (
            len(slots), self.layout.num_value_heads,
            self.layout.head_k_dim, self.layout.head_v_dim,
        )
        self.storage.view(self.dtype).flatten()[self._state_indices(slots, layer)] = (
            values.to(self.dtype).flatten(2)
        )

    def zero_state(self, slot: int) -> None:
        slots = torch.tensor([slot], dtype=torch.int64, device=self.device)
        values = torch.zeros(
            (1, self.layout.num_value_heads,
             self.layout.head_k_dim, self.layout.head_v_dim),
            dtype=self.dtype, device=self.device,
        )
        for layer in range(self.layout.num_linear_layers):
            self.write_states(slots, layer, values)

    def copy_state(self, source: int, destination: int) -> None:
        src = torch.tensor([source], dtype=torch.int64, device=self.device)
        dst = torch.tensor([destination], dtype=torch.int64, device=self.device)
        for layer in range(self.layout.num_linear_layers):
            self.write_states(dst, layer, self.read_states(src, layer))

    def _conv_indices(
        self, slots: torch.Tensor, granules: torch.Tensor, layer: int
    ) -> torch.Tensor:
        assert self.allocator.conv_frames == 1
        frames = self.conv_frame_table[
            slots.to(torch.int64), granules.to(torch.int64), 0
        ].to(torch.int64)
        base = (
            frames * (self.frame_bytes // self.dtype.itemsize)
            + layer * self.layout.conv_dim * self.layout.state_len
        )
        offsets = torch.arange(
            self.layout.conv_dim * self.layout.state_len,
            dtype=torch.int64, device=self.device,
        )
        return base[:, None] + offsets[None, :]

    def read_conv(
        self, slots: torch.Tensor, granules: torch.Tensor, layer: int
    ) -> torch.Tensor:
        return self.storage.view(self.dtype).flatten()[
            self._conv_indices(slots, granules, layer)
        ].view(len(slots), self.layout.conv_dim, self.layout.state_len)

    def write_conv(
        self, slots: torch.Tensor, granules: torch.Tensor,
        layer: int, values: torch.Tensor,
    ) -> None:
        assert values.shape == (
            len(slots), self.layout.conv_dim, self.layout.state_len,
        )
        self.storage.view(self.dtype).flatten()[
            self._conv_indices(slots, granules, layer)
        ] = values.to(self.dtype).flatten(1)

    def copy_conv_snapshot(
        self,
        source_slot    : int,
        source_granule : int,
        output_slot    : int,
        output_granule : int = 0,
    ) -> None:
        self.conv_view(output_slot, output_granule).copy_(
            self.conv_view(source_slot, source_granule)
        )

    def zero_work_conv(self, slot: int) -> None:
        slots = torch.tensor([slot], dtype=torch.int64, device=self.device)
        granules = torch.zeros(1, dtype=torch.int64, device=self.device)
        values = torch.zeros(
            (1, self.layout.conv_dim, self.layout.state_len),
            dtype=self.dtype, device=self.device,
        )
        for layer in range(self.layout.num_linear_layers):
            self.write_conv(slots, granules, layer, values)

    def zero_work_slots(self, slots: Sequence[int]) -> None:
        """Clear exclusive work frames in one operation; never touch shared decay frames."""
        frames: list[int] = []
        for slot in slots:
            record = self.allocator.slot(slot)
            assert record.kind == "work" and record.decay is None
            frames.extend(record.payload)
            for conv in record.conv:
                frames.extend(conv)
        if frames:
            indices = torch.tensor(frames, dtype=torch.int64, device=self.device)
            self.storage.index_fill_(0, indices, 0)

    def _write_descriptors(self, slots: Sequence[int]) -> None:
        if not slots:
            return
        payload, conv, decay = [], [], []
        conv_width = self.layout.num_granules * self.allocator.conv_frames
        for slot in slots:
            record = self.allocator.slot(slot)
            payload.extend(record.payload)
            row = [-1] * conv_width
            if record.kind == "log":
                granules = range(self.layout.num_granules)
            elif record.kind == "work":
                granules = (0,)
            else:
                granules = (self.layout.num_granules - 1,)
            for granule, frames in zip(granules, record.conv, strict=True):
                begin = granule * self.allocator.conv_frames
                row[begin:begin + self.allocator.conv_frames] = frames
            conv.extend(row)
            decay.extend(
                (-1, -1) if record.decay is None else
                (record.decay.frame, record.decay.index)
            )
        indices, payload, conv, decay = make_device_vectors(
            [slots, payload, conv, decay], self.device
        )
        indices = indices.long()
        self.payload_frame_table.index_copy_(0, indices, payload.reshape(len(slots), -1))
        self.conv_frame_table.index_copy_(0, indices, conv.reshape(
            len(slots), self.layout.num_granules, self.allocator.conv_frames
        ))
        self.decay_frame_table.index_copy_(0, indices, decay.reshape(len(slots), 2))

    def reserve(
        self, *, kv: int = 0, logs: int = 0, works: int = 0,
        timings: dict[str, float] | None = None,
    ) -> tuple[list[int], list[int], list[int]] | None:
        stage = perf_counter_ns() if timings is not None else 0
        self.poll_reclamation(timings)
        if timings is not None:
            now = perf_counter_ns()
            timings["poll_reclamation_ms"] = (now - stage) / 1e6
            stage = now
        allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
        if timings is not None:
            now = perf_counter_ns()
            timings["allocator_reserve_ms"] = (now - stage) / 1e6
            stage = now
        if allocated is None and (self._retired_materializations or self._retired_releases):
            if timings is not None:
                timings["waited_materializations"] = sum(
                    len(slots) for _event, slots in self._retired_materializations
                )
                timings["waited_merge_slots"] = len(self._retiring_slots)
            for event, _slots in self._retired_materializations:
                event.synchronize()
            for event, _slots in self._retired_releases:
                event.synchronize()
            if timings is not None:
                now = perf_counter_ns()
                timings["wait_materialization_ms"] = (now - stage) / 1e6
                stage = now
            self.poll_reclamation(timings)
            if timings is not None:
                now = perf_counter_ns()
                timings["retry_poll_reclamation_ms"] = (now - stage) / 1e6
                stage = now
            allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
            if timings is not None:
                now = perf_counter_ns()
                timings["retry_allocator_reserve_ms"] = (now - stage) / 1e6
                stage = now
        if allocated is None:
            if timings is not None:
                timings["failed"] = 1
            return None
        self._write_descriptors(allocated[1] + allocated[2])
        if timings is not None:
            timings["descriptor_write_ms"] = (
                perf_counter_ns() - stage
            ) / 1e6
            timings["descriptor_count"] = len(allocated[1]) + len(allocated[2])
        return allocated

    def materialize(self, slot: int) -> None:
        self.materialize_slots((slot,))

    def materialize_slots(self, slots: Sequence[int]) -> None:
        """Retire Log-only frames after construction kernels have finished."""
        if not slots:
            return
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._retired_materializations.append((event, tuple(slots)))
        else:
            for slot in slots:
                self.allocator.materialize(slot)
            self._write_descriptors(slots)

    def commit_checkpoint(self, slot: int) -> None:
        self.allocator.commit_checkpoint(slot)
        self._write_descriptors((slot,))

    def retire_slots(self, slots: Sequence[int]) -> None:
        """Remove merged slots from ownership now, reclaim frames after GPU reads.

        Descriptors and allocator records remain live until the event completes.
        Capacity checks must not count these frames as free before reclamation.
        """
        if not slots:
            return
        assert len(slots) == len(set(slots))
        assert not self._retiring_slots.intersection(slots)
        if self.device.type != "cuda":
            self.release_slots(slots)
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._retiring_slots.update(slots)
        self._retired_releases.append((event, tuple(slots)))

    @property
    def retiring_slots(self) -> frozenset[int]:
        return frozenset(self._retiring_slots)

    def poll_reclamation(self, timings: dict[str, float] | None = None) -> None:
        remaining = []
        completed = []
        for event, slots in self._retired_materializations:
            start = perf_counter_ns() if timings is not None else 0
            if event.query():
                if timings is not None:
                    timings["reclamation_query_ms"] = (
                        timings.get("reclamation_query_ms", 0.0)
                        + (perf_counter_ns() - start) / 1e6
                    )
                    start = perf_counter_ns()
                materialized = 0
                for slot in slots:
                    if slot in self._retiring_slots:
                        # The entire slot will be freed; do not rewrite a
                        # checkpoint descriptor only to clear it again.
                        continue
                    self.allocator.materialize(slot)
                    completed.append(slot)
                    materialized += 1
                if timings is not None:
                    timings["reclamation_materialize_ms"] = (
                        timings.get("reclamation_materialize_ms", 0.0)
                        + (perf_counter_ns() - start) / 1e6
                    )
                    timings["reclamation_count"] = (
                        timings.get("reclamation_count", 0) + materialized
                    )
            else:
                if timings is not None:
                    timings["reclamation_query_ms"] = (
                        timings.get("reclamation_query_ms", 0.0)
                        + (perf_counter_ns() - start) / 1e6
                    )
                remaining.append((event, slots))
        self._retired_materializations = remaining
        remaining_releases = []
        for event, slots in self._retired_releases:
            if event.query():
                self._release_slots(slots, timings=timings)
                self._retiring_slots.difference_update(slots)
                if timings is not None:
                    timings["merge_reclamation_count"] = (
                        timings.get("merge_reclamation_count", 0) + len(slots)
                    )
            else:
                remaining_releases.append((event, slots))
        self._retired_releases = remaining_releases
        start = perf_counter_ns() if timings is not None else 0
        self._write_descriptors(completed)
        if timings is not None:
            timings["reclamation_descriptor_ms"] = (
                timings.get("reclamation_descriptor_ms", 0.0)
                + (perf_counter_ns() - start) / 1e6
            )

    def release_kv(self, frames: list[int] | tuple[int, ...]) -> None:
        self.allocator.release_kv(frames)

    def release_slot(
        self, slot: int, timings: dict[str, float] | None = None,
    ) -> None:
        self.release_slots((slot,), timings=timings)

    def release_slots(
        self, slots: Sequence[int], timings: dict[str, float] | None = None,
    ) -> None:
        if not slots:
            return
        assert len(slots) == len(set(slots))
        assert not self._retiring_slots.intersection(slots), "Slot already queued for reclamation"
        stage = perf_counter_ns() if timings is not None else 0
        selected = set(slots)
        for event, pending_slots in self._retired_materializations:
            if selected.intersection(pending_slots):
                event.synchronize()
        if timings is not None:
            now = perf_counter_ns()
            timings["wait_materialization_ms"] = (
                timings.get("wait_materialization_ms", 0.0)
                + (now - stage) / 1e6
            )
            stage = now
        self.poll_reclamation(timings)
        if timings is not None:
            now = perf_counter_ns()
            timings["poll_reclamation_ms"] = (
                timings.get("poll_reclamation_ms", 0.0)
                + (now - stage) / 1e6
            )
            stage = now
        self._release_slots(slots, timings=timings)

    def _release_slots(
        self, slots: Sequence[int], timings: dict[str, float] | None = None,
    ) -> None:
        stage = perf_counter_ns() if timings is not None else 0
        for slot in slots:
            self.allocator.release_slot(slot)
        if timings is not None:
            now = perf_counter_ns()
            timings["allocator_release_ms"] = (
                timings.get("allocator_release_ms", 0.0)
                + (now - stage) / 1e6
            )
            stage = now
        indices, = make_device_vectors([slots], self.device, dtype=torch.int64)
        self.payload_frame_table.index_fill_(0, indices, -1)
        self.conv_frame_table.index_fill_(0, indices, -1)
        self.decay_frame_table.index_fill_(0, indices, -1)
        if timings is not None:
            timings["descriptor_clear_ms"] = (
                timings.get("descriptor_clear_ms", 0.0)
                + (perf_counter_ns() - stage) / 1e6
            )
