import triton
import triton.language as tl


@triton.jit
def payload_pointer(
    storage, frames, slot, layer, head, elements,
    LAYERS: tl.constexpr, HEADS: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, GRANULARITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr, HEAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
):
    itemsize: tl.constexpr = 2
    head_granule: tl.constexpr = HEAD_BYTES // itemsize // (PAGE_SIZE // GRANULARITY)
    key_size: tl.constexpr = GRANULARITY * K
    value_size: tl.constexpr = GRANULARITY * V
    padding: tl.constexpr = head_granule - key_size - value_size
    granule = elements // head_granule
    within = elements % head_granule
    key_offset = (within // K * HEADS + head) * K + within % K
    value = within - key_size
    value_offset = key_size * HEADS + (value // V * HEADS + head) * V + value % V
    padding_offset = (key_size + value_size) * HEADS + head * padding + within - key_size - value_size
    offset = (granule * LAYERS + layer) * HEADS * head_granule
    offset += tl.where(within < key_size, key_offset,
                       tl.where(within < key_size + value_size, value_offset, padding_offset))
    byte_offset = offset * itemsize
    frame = tl.load(frames + slot * FRAMES_PER_SLOT + byte_offset // FRAME_BYTES,
                    mask=(slot >= 0) & (elements >= 0) & (elements < HEAD_BYTES // itemsize), other=0)
    return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
            + byte_offset % FRAME_BYTES).to(storage.dtype)


@triton.jit
def log_pointer(
    storage, frames, slot, layer, head, positions, columns,
    LAYERS: tl.constexpr, HEADS: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, GRANULARITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr, HEAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    VALUE: tl.constexpr,
    KEY_HEADS: tl.constexpr = 0,
):
    if KEY_HEADS != 0 and KEY_HEADS != HEADS:
        key_size: tl.constexpr = GRANULARITY * KEY_HEADS * K
        layer_size: tl.constexpr = key_size + GRANULARITY * HEADS * V
        compact_dim: tl.constexpr = V if VALUE else K
        heads: tl.constexpr = HEADS if VALUE else KEY_HEADS
        index = head if VALUE else head // (HEADS // KEY_HEADS)
        offset = (positions // GRANULARITY * LAYERS + layer) * layer_size
        offset += (key_size if VALUE else 0) + (positions % GRANULARITY * heads + index) * compact_dim
        if ((FRAME_BYTES % (compact_dim * 2) == 0 and layer_size % compact_dim == 0)
            and (not VALUE or key_size % compact_dim == 0)):


            row_bytes = offset * 2
            frame = tl.load(frames + slot * FRAMES_PER_SLOT + row_bytes // FRAME_BYTES,
                            mask=(slot >= 0) & (positions >= 0) & (positions < PAGE_SIZE), other=0)
            return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
                    + row_bytes % FRAME_BYTES).to(storage.dtype) + columns
        offset += columns
        byte_offset = offset * 2
        frame = tl.load(frames + slot * FRAMES_PER_SLOT + byte_offset // FRAME_BYTES,
                        mask=(slot >= 0) & (positions >= 0) & (positions < PAGE_SIZE), other=0)
        return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
                + byte_offset % FRAME_BYTES).to(storage.dtype)
    itemsize: tl.constexpr = 2
    head_granule: tl.constexpr = HEAD_BYTES // itemsize // (PAGE_SIZE // GRANULARITY)
    dim: tl.constexpr = V if VALUE else K
    elements = positions // GRANULARITY * head_granule
    elements += (GRANULARITY * K if VALUE else 0) + positions % GRANULARITY * dim + columns
    return payload_pointer(
        storage, frames, slot, layer, head, elements,
        LAYERS, HEADS, K, V, GRANULARITY, PAGE_SIZE, HEAD_BYTES, FRAME_BYTES, FRAMES_PER_SLOT,
    )
