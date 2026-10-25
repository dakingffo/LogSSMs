import triton
import triton.language as tl


@triton.jit
def reconstruct_tile(
    state_head, key_rows, scaled_values, valid, k_offset, v_offset, total_decay,
    HEAD_K_DIM: tl.constexpr, HEAD_V_DIM: tl.constexpr,
):
    state = tl.load(
        state_head + k_offset[:, None] * HEAD_V_DIM + v_offset[None, :],
        mask=(k_offset[:, None] < HEAD_K_DIM) & (v_offset[None, :] < HEAD_V_DIM), other=0.0,
    ).to(tl.float32)
    keys = tl.load(
        key_rows[None, :] + k_offset[:, None],
        mask=(k_offset[:, None] < HEAD_K_DIM) & valid[None, :], other=0.0,
    )
    return state * total_decay + tl.dot(keys, scaled_values)
