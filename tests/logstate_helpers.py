import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool


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


def log_offsets(layout, positions, columns, layer, heads, *, value):
    key_size = layout.granularity * layout.num_key_heads * layout.head_k_dim
    layer_size = key_size + layout.granularity * layout.num_value_heads * layout.head_v_dim
    offset = (positions // layout.granularity * layout.num_linear_layers + layer) * layer_size
    if value:
        return offset + key_size + (positions % layout.granularity * layout.num_value_heads + heads) * layout.head_v_dim + columns
    return offset + (positions % layout.granularity * layout.num_key_heads + heads) * layout.head_k_dim + columns


def remember_log_tail(recycle, tree, node) -> None:
    """Remember a pruned path in production's leaf-to-root order."""
    for part in reversed(tree.path(node)):
        recycle.remember(tree, part)


def _head_indices(
    pool: MHALogStatePool, slot: int, layer: int, head: int, *, state: bool
) -> torch.Tensor:
    assert 0 <= layer < pool.layout.num_linear_layers
    assert 0 <= head < pool._num_value_heads
    layout = pool.layout
    if state:
        elements = torch.arange(layout.head_k_dim * layout.head_v_dim, device=pool.device)
    else:
        positions = torch.arange(pool.page_size, device=pool.device)
        if pool.shared_keys:
            key = log_offsets(layout, positions[:, None], torch.arange(layout.head_k_dim, device=pool.device),
                                     layer, head // (layout.num_value_heads // layout.num_key_heads), value=False)
            value = log_offsets(layout, positions[:, None], torch.arange(layout.head_v_dim, device=pool.device),
                                       layer, head, value=True)
            offsets = torch.cat((key.flatten(), value.flatten())) * pool.dtype.itemsize
            frames = pool.payload_frame_table[slot, offsets // pool.frame_bytes].long()
            return frames * (pool.frame_bytes // pool.dtype.itemsize) + offsets % pool.frame_bytes // pool.dtype.itemsize
        head_granule = pool.head_payload_bytes // pool.dtype.itemsize // layout.num_granules
        key = positions[:, None] // layout.granularity * head_granule
        key = key + positions[:, None] % layout.granularity * layout.head_k_dim
        key = key + torch.arange(layout.head_k_dim, device=pool.device)
        value = positions[:, None] // layout.granularity * head_granule
        value = value + layout.granularity * layout.head_k_dim
        value = value + positions[:, None] % layout.granularity * layout.head_v_dim
        value = value + torch.arange(layout.head_v_dim, device=pool.device)
        elements = torch.cat((key.flatten(), value.flatten()))
    offsets = layout.payload_offsets(elements, layer, head, pool.frame_bytes) * pool.dtype.itemsize
    frames = pool.payload_frame_table[slot, offsets // pool.frame_bytes].long()
    return frames * (pool.frame_bytes // pool.dtype.itemsize) + offsets % pool.frame_bytes // pool.dtype.itemsize


def head_view(
    pool: MHALogStatePool, slot: int, layer: int, head: int, *, state: bool
) -> torch.Tensor:
    """Read a logical head; granule-major heads are not contiguous views."""
    result = pool.storage.view(pool.dtype).flatten()[_head_indices(pool, slot, layer, head, state=state)]
    return result.view(pool.layout.head_k_dim, pool.layout.head_v_dim) if state else result


def write_head(
    pool: MHALogStatePool, slot: int, layer: int, head: int, values,
    *, state: bool, index=slice(None),
) -> None:
    indices = _head_indices(pool, slot, layer, head, state=state)[index]
    values = torch.as_tensor(values, dtype=pool.dtype, device=pool.device)
    pool.storage.view(pool.dtype).flatten()[indices] = values.flatten() if values.ndim else values
