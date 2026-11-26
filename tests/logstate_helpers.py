from dataclasses import replace

import torch

from logssms.cache.logstate.frame_pool import MHALogStatePool
from logssms.kernel.logstate import EvictionPlanner


def decay_view(pool: MHALogStatePool, slot: int) -> torch.Tensor:
    record = pool.allocator.slot(slot)
    assert record.decay is not None
    size = pool.layout.log_decay_bytes
    begin = record.decay.index * size
    return pool.storage[record.decay.frame, begin:begin + size].view(torch.float32).view(
        pool.layout.num_linear_layers, pool.page_size, pool.layout.num_value_heads,
    )


def matched_kv(tree, node) -> torch.Tensor:
    values = [part.kv for part in tree.path(node)]
    return torch.cat(values) if values else tree.empty_tensor


def is_lock_boundary(tree, leaf, position: int) -> bool:
    for node in tree.path(leaf):
        if node.depth == position:
            return node.ref_count > sum(child.ref_count for child in node.children.values())
    return False


def log_offsets(layout, positions, columns, layer, heads, *, value, head_bytes):
    key_size = layout.granularity * layout.num_key_heads * layout.head_k_dim
    layer_size = key_size + layout.granularity * layout.num_value_heads * layout.head_v_dim
    offset = (positions // layout.granularity * layout.num_linear_layers + layer) * layer_size
    if value:
        return offset + key_size + (heads * layout.granularity + positions % layout.granularity) * layout.head_v_dim + columns
    return offset + (heads * layout.granularity + positions % layout.granularity) * layout.head_k_dim + columns


def eviction_node(tree, node):
    from logssms.cache.logstate.prefix_cache import LogRadixEvictionNode
    return LogRadixEvictionNode(
        node, node.begin, node.depth, node.eviction_slots, tree,
    )


def _head_indices(
    pool: MHALogStatePool, slot: int, layer: int, head: int, *, state: bool
) -> torch.Tensor:
    assert 0 <= layer < pool.layout.num_linear_layers
    assert 0 <= head < pool._num_value_heads
    layout = pool.layout
    if state:
        elements = torch.arange(layout.head_k_dim * layout.head_v_dim, device=pool.device)
        offsets = (layer * layout.num_value_heads + head) * pool.head_payload_bytes
        offsets = offsets + elements * pool.dtype.itemsize
    else:
        positions = torch.arange(pool.page_size, device=pool.device)
        key = log_offsets(layout, positions[:, None], torch.arange(layout.head_k_dim, device=pool.device),
                          layer, head // (layout.num_value_heads // layout.num_key_heads),
                          value=False, head_bytes=pool.head_payload_bytes)
        value = log_offsets(layout, positions[:, None], torch.arange(layout.head_v_dim, device=pool.device),
                            layer, head, value=True, head_bytes=pool.head_payload_bytes)
        offsets = torch.cat((key.flatten(), value.flatten())) * pool.dtype.itemsize
    frames = pool.payload_frame_table[slot, offsets // pool.frame_bytes].long()
    return frames * (pool.frame_bytes // pool.dtype.itemsize) + offsets % pool.frame_bytes // pool.dtype.itemsize


def head_view(
    pool: MHALogStatePool, slot: int, layer: int, head: int, *, state: bool
) -> torch.Tensor:
    """Read a logical State or Log head."""
    result = pool.storage.view(pool.dtype).flatten()[_head_indices(pool, slot, layer, head, state=state)]
    return result.view(pool.layout.head_k_dim, pool.layout.head_v_dim) if state else result


def write_head(
    pool: MHALogStatePool, slot: int, layer: int, head: int, values,
    *, state: bool, index=slice(None),
) -> None:
    indices = _head_indices(pool, slot, layer, head, state=state)[index]
    values = torch.as_tensor(values, dtype=pool.dtype, device=pool.device)
    pool.storage.view(pool.dtype).flatten()[indices] = values.flatten() if values.ndim else values


def eviction_slot_records(slots):
    return [(*row, int(row[2] < 0)) for row in slots]


def plan_eviction(nodes, kv, node_slots, slots, occupied, config, actions, models=None):
    if models is None:
        # Isolate timestamps and span ordering without assigning locality to synthetic rows.
        models = [(index, 0, 32, 2., -1000., 0., 32, 0, 32)
                  for index in range(len(nodes))]
    order = [(0, uuid, index) for index, (uuid, *_) in enumerate(nodes)]
    planner = EvictionPlanner(
        nodes, kv, node_slots, eviction_slot_records(slots), occupied, config, actions, models, order,
    )
    return planner.plan(config, occupied)


def materialize_log(allocator, slot):
    retired = allocator.materialize_group((slot,))
    allocator.release_retirement(retired)
    return retired.frames


def materialize_pool_log(pool, slot):
    materialize_log(pool.allocator, slot)
    pool._write_descriptors([slot])


def seed_checkpoint_slot(pool, slot):
    allocator = pool.allocator
    record = allocator.slot(slot)
    assert record.kind == "work" and record.decay is None
    allocator._slots[slot] = replace(record, kind="checkpoint")
    pool._write_descriptors([slot])
