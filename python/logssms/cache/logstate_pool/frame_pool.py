from __future__ import annotations

import torch

from .frame_allocator import LinearFrameSlot, LogStateFrameAllocator
from .layout import LogStateSlotLayout


class FramedMHALogStatePool:
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
        self.allocator = allocator
        self.device = device
        self.dtype = dtype
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
        self.device = self.storage.device
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
        self._retired_materializations: list[tuple[torch.cuda.Event, int]] = []

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

    def head_view(
        self, slot: int, layer: int, head: int, *, state: bool
    ) -> torch.Tensor:
        """Return one contiguous head overlay; useful for testing the frame map.

        Production kernels must do the equivalent lookup on the GPU. A head
        crossing a frame boundary needs a tiled kernel instead of a tensor view.
        """
        assert 0 <= layer < self.layout.num_linear_layers
        assert 0 <= head < self._num_value_heads
        logical_offset = (
            layer * self._num_value_heads + head
        ) * self.head_payload_bytes
        frame_index, offset = divmod(logical_offset, self.frame_bytes)
        if offset + self.head_payload_bytes > self.frame_bytes:
            raise ValueError("One head crosses a physical frame boundary")
        frame = self.allocator.slot(slot).payload[frame_index]
        storage = self.storage[frame, offset:offset + self.head_payload_bytes]
        if state:
            return storage[:self.layout.head_recurrent_state_bytes].view(
                self.dtype
            ).view(self.layout.head_k_dim, self.layout.head_v_dim)
        size = self.layout.head_log_key_bytes + self.layout.head_log_value_bytes
        return storage[:size].view(self.dtype)

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

    def zero_work_conv(self, slot: int) -> None:
        slots = torch.tensor([slot], dtype=torch.int64, device=self.device)
        granules = torch.zeros(1, dtype=torch.int64, device=self.device)
        values = torch.zeros(
            (1, self.layout.conv_dim, self.layout.state_len),
            dtype=self.dtype, device=self.device,
        )
        for layer in range(self.layout.num_linear_layers):
            self.write_conv(slots, granules, layer, values)

    def _write_descriptor(self, slot: int, record: LinearFrameSlot) -> None:
        self.payload_frame_table[slot].copy_(
            torch.tensor(record.payload, dtype=torch.int32, device=self.device)
        )
        self.conv_frame_table[slot].fill_(-1)
        if record.kind == "log":
            granules = range(self.layout.num_granules)
        elif record.kind == "work":
            granules = (0,)
        else:
            granules = (self.layout.num_granules - 1,)
        for granule, frames in zip(granules, record.conv, strict=True):
            self.conv_frame_table[slot, granule].copy_(
                torch.tensor(frames, dtype=torch.int32, device=self.device)
            )
        if record.decay is None:
            self.decay_frame_table[slot].fill_(-1)
        else:
            self.decay_frame_table[slot] = torch.tensor(
                (record.decay.frame, record.decay.index),
                dtype=torch.int32, device=self.device,
            )

    def reserve(
        self, *, kv: int = 0, logs: int = 0, works: int = 0
    ) -> tuple[list[int], list[int], list[int]] | None:
        self.poll_reclamation()
        allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
        if allocated is None and self._retired_materializations:
            for event, _slot in self._retired_materializations:
                event.synchronize()
            self.poll_reclamation()
            allocated = self.allocator.reserve(kv=kv, logs=logs, states=works)
        if allocated is None:
            return None
        for slot in allocated[1] + allocated[2]:
            self._write_descriptor(slot, self.allocator.slot(slot))
        return allocated

    def materialize(self, slot: int) -> None:
        """Retire Log-only frames after construction kernels have finished."""
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._retired_materializations.append((event, slot))
        else:
            self.allocator.materialize(slot)
            self._write_descriptor(slot, self.allocator.slot(slot))

    def poll_reclamation(self) -> None:
        remaining = []
        for event, slot in self._retired_materializations:
            if event.query():
                self.allocator.materialize(slot)
                self._write_descriptor(slot, self.allocator.slot(slot))
            else:
                remaining.append((event, slot))
        self._retired_materializations = remaining

    def release_kv(self, frames: list[int] | tuple[int, ...]) -> None:
        self.allocator.release_kv(frames)

    def release_slot(self, slot: int) -> None:
        for event, pending_slot in self._retired_materializations:
            if pending_slot == slot:
                event.synchronize()
        self.poll_reclamation()
        self.allocator.release_slot(slot)
        self.payload_frame_table[slot].fill_(-1)
        self.conv_frame_table[slot].fill_(-1)
        self.decay_frame_table[slot].fill_(-1)
