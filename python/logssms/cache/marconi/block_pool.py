from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import prod

import torch

from .allocator import BlockAllocator
from .layout import HybridBlockLayout


class HybridBlockPool:
    def __init__(self, layout: HybridBlockLayout, num_blocks: int, device, dtype):
        self.layout = layout
        self.allocator = BlockAllocator(num_blocks)
        self._device, self._dtype = torch.device(device), dtype
        self.kv_page_size = layout.block_size
        self.storage = torch.empty(
            layout.layers_per_group, num_blocks, layout.layer_elements,
            device=device, dtype=dtype,
        )
        self.num_blocks = num_blocks
        if num_blocks < layout.state_blocks:
            raise ValueError("Insufficient blocks for a complete State")
        self.state_blocks: dict[int, tuple[int, ...]] = {}
        self.state_refs: dict[int, int] = {}
        self._state_ids = deque(range(num_blocks))
        self.group_table = torch.zeros(num_blocks, layout.state_groups, device=device, dtype=torch.int64)
        lane_stride, block_stride, _ = self.storage.stride()
        state_starts = num_blocks - layout.state_blocks_per_group + 1
        self.recurrent_state = self.storage.as_strided(
            (state_starts, layout.layers_per_group, *layout.state_shape),
            (block_stride, lane_stride, layout.state_shape[1] * layout.state_shape[2],
             layout.state_shape[2], 1),
        )
        self.conv_state = self.storage.as_strided(
            (state_starts, layout.layers_per_group, *layout.conv_shape),
            (block_stride, lane_stride, layout.conv_shape[1], 1),
            storage_offset=prod(layout.state_shape),
        )

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def num_layers(self):
        return self.layout.layers_per_group

    def k_cache(self, index):
        l = self.layout
        return self.storage[index, :, :l.layer_elements // 2].view(
            self.num_blocks, l.block_size, l.kv_heads, l.head_dim)

    def v_cache(self, index):
        l = self.layout
        return self.storage[index, :, l.layer_elements // 2:].view(
            self.num_blocks, l.block_size, l.kv_heads, l.head_dim)

    def store_kv(self, k, v, out_loc, layer_id):
        # Rotary attention writes through the ordinary fused paged store.
        pages, offsets = out_loc // self.kv_page_size, out_loc % self.kv_page_size
        self.k_cache(layer_id)[pages, offsets] = k
        self.v_cache(layer_id)[pages, offsets] = v

    def allocate_state(self) -> int:
        width = self.layout.state_blocks_per_group
        blocks = self.allocator.allocate_groups(self.layout.state_groups, width)
        slot = self._state_ids.popleft()
        self.state_blocks[slot] = blocks
        self.state_refs[slot] = 1
        self.group_table[slot] = torch.tensor(blocks[::width], device=self.device)
        return slot

    def retain_state(self, slot):
        self.state_refs[slot] += 1

    def release_state(self, slot):
        self.state_refs[slot] -= 1
        if self.state_refs[slot] == 0:
            self.allocator.release(self.state_blocks.pop(slot))
            del self.state_refs[slot]
            self._state_ids.append(slot)

    def copy_state(self, source, target):
        width = self.layout.state_blocks_per_group
        for src, dst in zip(self.state_blocks[source][::width], self.state_blocks[target][::width]):
            self.recurrent_state[dst].copy_(self.recurrent_state[src])
            self.conv_state[dst].copy_(self.conv_state[src])

    def zero_state(self, slot):
        for block in self.state_blocks[slot][::self.layout.state_blocks_per_group]:
            self.recurrent_state[block].zero_()
            self.conv_state[block].zero_()

    def map_metadata(self, metadata, layer):
        group, lane = divmod(layer, self.layout.layers_per_group)
        def mapped(value):
            if value is None:
                return None
            result = self.group_table[:, group][value.clamp_min(0).long()]
            return torch.where(value >= 0, result, -1)
        names = ("read_slots", "write_slots", "baseline_slots", "checkpoint_slots",
                 "chunk_write_slots", "conv_token_slots")
        return replace(metadata, **{name: mapped(getattr(metadata, name)) for name in names}), lane
