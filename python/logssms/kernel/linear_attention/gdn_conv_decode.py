import torch
import triton
import triton.language as tl


@triton.jit
def _frame_element_offset(frame, frame_elems: tl.constexpr):
    return frame.to(tl.int64) * frame_elems


@triton.jit
def _gdn_conv_decode(
    X, W, Q, K, V, StandardState, Storage, FrameTable,
    ReadSlots, ReadGranules, WriteSlots, WriteGranules, WorkSlots,
    C: tl.constexpr, KEY_DIM: tl.constexpr, STATE_LEN: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr, X_COL_STRIDE: tl.constexpr,
    NUM_LAYERS: tl.constexpr, LAYER: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, NUM_GRANULES: tl.constexpr,
    LOGSTATE: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    ch = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    s = tl.arange(0, BLOCK_S)
    channel_ok = ch < C
    tap_ok = s < STATE_LEN

    read_slot = tl.load(ReadSlots + b)
    write_slot = tl.load(WriteSlots + b)
    if LOGSTATE:
        read_granule = tl.load(ReadGranules + b)
        write_granule = tl.load(WriteGranules + b)
        work_slot = tl.load(WorkSlots + b)
        read_frame = tl.load(
            FrameTable + read_slot * NUM_GRANULES + read_granule
        )
        write_frame = tl.load(
            FrameTable + write_slot * NUM_GRANULES + write_granule
        )
        work_frame = tl.load(FrameTable + work_slot * NUM_GRANULES)
        state_ptr = tl.cast(Storage, tl.pointer_type(X.dtype.element_ty))
        old_base = _frame_element_offset(read_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
        new_base = _frame_element_offset(write_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
        work_base = _frame_element_offset(work_frame, FRAME_ELEMS) + LAYER * C * STATE_LEN
    else:
        state_ptr = StandardState
        old_base = (read_slot * NUM_LAYERS + LAYER) * C * STATE_LEN
        new_base = (write_slot * NUM_LAYERS + LAYER) * C * STATE_LEN
        work_base = 0

    old = tl.load(
        state_ptr + old_base + ch[:, None] * STATE_LEN + s[None, :],
        channel_ok[:, None] & tap_ok[None, :], other=0,
    )
    x = tl.load(
        X + b * X_ROW_STRIDE + ch * X_COL_STRIDE, channel_ok, other=0
    ).to(W.dtype.element_ty)
    weights = tl.load(
        W + ch[:, None] * (STATE_LEN + 1) + s[None, :],
        channel_ok[:, None] & tap_ok[None, :], other=0,
    )
    last_weight = tl.load(
        W + ch * (STATE_LEN + 1) + STATE_LEN, channel_ok, other=0,
    )
    result = tl.sum(old.to(tl.float32) * weights.to(tl.float32), 1)
    result += x.to(tl.float32) * last_weight.to(tl.float32)

    result = result.to(W.dtype.element_ty).to(tl.float32)
    result = (result * tl.sigmoid(result)).to(X.dtype.element_ty)

    shifted = tl.load(
        state_ptr + old_base + ch[:, None] * STATE_LEN + (s[None, :] + 1),
        channel_ok[:, None] & (s[None, :] + 1 < STATE_LEN), other=0,
    )
    shifted = tl.where(s[None, :] == STATE_LEN - 1, x[:, None], shifted)
    tl.store(
        state_ptr + new_base + ch[:, None] * STATE_LEN + s[None, :],
        shifted, channel_ok[:, None] & tap_ok[None, :],
    )
    if LOGSTATE:
        tl.store(
            state_ptr + work_base + ch[:, None] * STATE_LEN + s[None, :],
            shifted, channel_ok[:, None] & tap_ok[None, :],
        )
    tl.store(Q + b * KEY_DIM + ch, result, channel_ok & (ch < KEY_DIM))
    tl.store(K + b * KEY_DIM + ch - KEY_DIM, result,
             channel_ok & (ch >= KEY_DIM) & (ch < 2 * KEY_DIM))
    tl.store(V + b * (C - 2 * KEY_DIM) + ch - 2 * KEY_DIM, result,
             channel_ok & (ch >= 2 * KEY_DIM))


def conv_decode_qkv(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    config,
    *,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
    standard_state: torch.Tensor | None = None,
    pool=None,
    read_granules: torch.Tensor | None = None,
    write_granules: torch.Tensor | None = None,
    work_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mixed_qkv.is_cuda and mixed_qkv.ndim == 2
    bs, channels = mixed_qkv.shape
    assert channels == config.conv_dim
    q = torch.empty((bs, 1, config.num_k_heads, config.head_k_dim),
                    dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k = torch.empty_like(q)
    v = torch.empty((bs, 1, config.num_v_heads, config.head_v_dim),
                    dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    logstate = pool is not None
    if logstate:
        assert read_granules is not None and write_granules is not None
        assert work_slots is not None and pool.allocator.conv_frames == 1
        state = pool.storage
        table = pool.conv_frame_table
        frame_elems = pool.frame_bytes // mixed_qkv.element_size()
        num_granules = pool.layout.num_granules
        num_layers = pool.layout.num_linear_layers
    else:
        assert standard_state is not None
        state = standard_state
        table = standard_state
        read_granules = write_granules = work_slots = read_slots
        frame_elems = num_granules = 1
        num_layers = standard_state.shape[1]
    _gdn_conv_decode[(bs, triton.cdiv(channels, 32))](
        mixed_qkv, weight, q, k, v, state, state, table,
        read_slots, read_granules, write_slots, write_granules, work_slots,
        channels, config.key_dim, config.state_len,
        mixed_qkv.stride(0), mixed_qkv.stride(1),
        num_layers, config.layer_idx, frame_elems, num_granules,
        logstate, 32, triton.next_power_of_2(config.state_len),
        num_warps=4,
    )
    return q, k, v
