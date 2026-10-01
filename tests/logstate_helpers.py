import torch

from logssms.cache.logstate_pool.frame_pool import FramedMHALogStatePool


def head_view(
    pool: FramedMHALogStatePool, slot: int, layer: int, head: int, *, state: bool
) -> torch.Tensor:
    """Return one contiguous head overlay; useful for testing the frame map.

    Production kernels must do the equivalent lookup on the GPU. A head
    crossing a frame boundary needs a tiled kernel instead of a tensor view.
    """
    assert 0 <= layer < pool.layout.num_linear_layers
    assert 0 <= head < pool._num_value_heads
    logical_offset = (
        layer * pool._num_value_heads + head
    ) * pool.head_payload_bytes
    frame_index, offset = divmod(logical_offset, pool.frame_bytes)
    if offset + pool.head_payload_bytes > pool.frame_bytes:
        raise ValueError("One head crosses a physical frame boundary")
    frame = pool.allocator.slot(slot).payload[frame_index]
    storage = pool.storage[frame, offset:offset + pool.head_payload_bytes]
    if state:
        return storage[:pool.layout.head_recurrent_state_bytes].view(
            pool.dtype
        ).view(pool.layout.head_k_dim, pool.layout.head_v_dim)
    size = pool.layout.head_log_key_bytes + pool.layout.head_log_value_bytes
    return storage[:size].view(pool.dtype)
