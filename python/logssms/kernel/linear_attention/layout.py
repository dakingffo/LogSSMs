import triton
import triton.language as tl


@triton.jit
def state_pointer(
    storage, frames, slot, layer, head, elements,
    HEADS: tl.constexpr, HEAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
):
    byte_offset = (layer * HEADS + head) * HEAD_BYTES
    if HEAD_BYTES <= FRAME_BYTES:
        frame = tl.load(frames + slot * FRAMES_PER_SLOT + byte_offset // FRAME_BYTES,
                        mask=slot >= 0, other=0)
        return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
                + byte_offset % FRAME_BYTES).to(storage.dtype) + elements
    byte_offset += elements * storage.dtype.element_ty.primitive_bitwidth // 8
    frame = tl.load(frames + slot * FRAMES_PER_SLOT + byte_offset // FRAME_BYTES,
                    mask=slot >= 0, other=0)
    return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
            + byte_offset % FRAME_BYTES).to(storage.dtype)


@triton.jit
def _log_pointer(
    storage, frames, slot, layer, head, positions, columns,
    LAYERS: tl.constexpr, HEADS: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, GRANULARITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr, HEAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    VALUE: tl.constexpr,
    KEY_HEADS: tl.constexpr = 0,
):
    key_heads: tl.constexpr = KEY_HEADS if KEY_HEADS != 0 else HEADS
    key_size: tl.constexpr = GRANULARITY * key_heads * K
    layer_size: tl.constexpr = key_size + GRANULARITY * HEADS * V
    dim: tl.constexpr = V if VALUE else K
    index = head if VALUE else head // (HEADS // key_heads)
    offset = (positions // GRANULARITY * LAYERS + layer) * layer_size
    offset += (key_size if VALUE else 0) + (index * GRANULARITY + positions % GRANULARITY) * dim
    if ((FRAME_BYTES % (dim * 2) == 0 and layer_size % dim == 0)
        and (not VALUE or key_size % dim == 0)):
        row_bytes = offset * 2
        frame = tl.load(frames + slot * FRAMES_PER_SLOT + row_bytes // FRAME_BYTES,
                        mask=(slot >= 0) & (positions >= 0) & (positions < PAGE_SIZE), other=0)
        return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
                + row_bytes % FRAME_BYTES).to(storage.dtype) + columns
    byte_offset = (offset + columns) * 2
    frame = tl.load(frames + slot * FRAMES_PER_SLOT + byte_offset // FRAME_BYTES,
                    mask=(slot >= 0) & (positions >= 0) & (positions < PAGE_SIZE), other=0)
    return (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
            + byte_offset % FRAME_BYTES).to(storage.dtype)


@triton.jit
def log_pointer(
    storage, frames, slot, layer, head, positions, columns,
    LAYERS: tl.constexpr, HEADS: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, GRANULARITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr, HEAD_BYTES: tl.constexpr,
    FRAME_BYTES: tl.constexpr, FRAMES_PER_SLOT: tl.constexpr,
    VALUE: tl.constexpr, KEY_HEADS: tl.constexpr = 0,
    start=None, end=None,
):
    key_heads: tl.constexpr = KEY_HEADS if KEY_HEADS != 0 else HEADS
    dim: tl.constexpr = V if VALUE else K
    key_size: tl.constexpr = GRANULARITY * key_heads * K
    layer_size: tl.constexpr = key_size + GRANULARITY * HEADS * V
    aligned: tl.constexpr = ((FRAME_BYTES % (GRANULARITY * dim * 2) == 0)
                            & (layer_size % (GRANULARITY * dim) == 0)
                            & ((not VALUE) | (key_size % (GRANULARITY * dim) == 0)))
    if start is not None and end is not None:
        if aligned:
            if (start // GRANULARITY == (end - 1) // GRANULARITY) | (start == end):
                index = head if VALUE else head // (HEADS // key_heads)
                head_offset = (start // GRANULARITY * LAYERS + layer) * layer_size
                head_offset += (key_size if VALUE else 0) + index * GRANULARITY * dim
                head_bytes = head_offset * 2
                frame = tl.load(frames + slot * FRAMES_PER_SLOT + head_bytes // FRAME_BYTES,
                                mask=(slot >= 0) & (start >= 0) & (start < PAGE_SIZE), other=0)
                base = (storage.to(tl.pointer_type(tl.uint8)) + frame.to(tl.int64) * FRAME_BYTES
                        + head_bytes % FRAME_BYTES).to(storage.dtype)
                pointer = base + positions % GRANULARITY * dim + columns
            else:
                pointer = _log_pointer(
                    storage, frames, slot, layer, head, positions, columns,
                    LAYERS, HEADS, K, V, GRANULARITY, PAGE_SIZE, HEAD_BYTES,
                    FRAME_BYTES, FRAMES_PER_SLOT, VALUE, KEY_HEADS,
                )
            return pointer
    return _log_pointer(
        storage, frames, slot, layer, head, positions, columns,
        LAYERS, HEADS, K, V, GRANULARITY, PAGE_SIZE, HEAD_BYTES,
        FRAME_BYTES, FRAMES_PER_SLOT, VALUE, KEY_HEADS,
    )


@triton.jit
def conv_pointer(storage, frames, slot, granule, elements,
                 GRANULES: tl.constexpr, FRAMES: tl.constexpr,
                 FRAME_ELEMS: tl.constexpr, valid=True):
    if FRAMES == 1:
        frame = tl.load(frames + slot * GRANULES + granule, slot >= 0, other=0)
        return storage + frame.to(tl.int64) * FRAME_ELEMS + elements
    frame = tl.load(frames + (slot * GRANULES + granule) * FRAMES + elements // FRAME_ELEMS,
                    (slot >= 0) & valid, other=0)
    return storage + frame.to(tl.int64) * FRAME_ELEMS + elements % FRAME_ELEMS
