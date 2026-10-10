from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["TOKENS", "ROWS", "Q_BATCH_STRIDE", "K_BATCH_STRIDE",
                             "V_BATCH_STRIDE", "G_BATCH_STRIDE", "B_BATCH_STRIDE"])
def _prepare_gdn_kernel(
    query, key, value, gate, beta, A_log, dt_bias,
    q_out, k_out, v_out, gate_out, beta_out,
    Q_BATCH_STRIDE, Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr, Q_DIM_STRIDE: tl.constexpr,
    K_BATCH_STRIDE, K_TOKEN_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr, K_DIM_STRIDE: tl.constexpr,
    V_BATCH_STRIDE, V_TOKEN_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr, V_DIM_STRIDE: tl.constexpr,
    G_BATCH_STRIDE, G_TOKEN_STRIDE: tl.constexpr, G_HEAD_STRIDE: tl.constexpr,
    B_BATCH_STRIDE, B_TOKEN_STRIDE: tl.constexpr, B_HEAD_STRIDE: tl.constexpr,
    A_STRIDE: tl.constexpr, DT_STRIDE: tl.constexpr,
    TOKENS, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr, VALUE_DIM: tl.constexpr, ROWS,
    PACKED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK_DIM)
    head = rows % VALUE_HEADS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    if PACKED:
        token = rows // VALUE_HEADS
        batch = tl.full((BLOCK_ROWS,), 0, tl.int64)
    else:
        token = (rows // VALUE_HEADS) % TOKENS
        batch = (rows // (VALUE_HEADS * TOKENS)).to(tl.int64)
    active = rows < ROWS
    key_mask = active[:, None] & (columns[None, :] < KEY_DIM)
    q = tl.load(
        query + batch[:, None] * Q_BATCH_STRIDE + token[:, None] * Q_TOKEN_STRIDE
        + key_head[:, None] * Q_HEAD_STRIDE + columns[None, :] * Q_DIM_STRIDE,
        key_mask, other=0,
    ).to(tl.float32)
    k = tl.load(
        key + batch[:, None] * K_BATCH_STRIDE + token[:, None] * K_TOKEN_STRIDE
        + key_head[:, None] * K_HEAD_STRIDE + columns[None, :] * K_DIM_STRIDE,
        key_mask, other=0,
    ).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, 1) + 1e-6)[:, None]
    k *= tl.rsqrt(tl.sum(k * k, 1) + 1e-6)[:, None]
    tl.store(q_out + rows[:, None].to(tl.int64) * KEY_DIM + columns[None, :], q, key_mask)
    tl.store(k_out + rows[:, None].to(tl.int64) * KEY_DIM + columns[None, :], k, key_mask)
    value_mask = active[:, None] & (columns[None, :] < VALUE_DIM)
    v = tl.load(
        value + batch[:, None] * V_BATCH_STRIDE + token[:, None] * V_TOKEN_STRIDE
        + head[:, None] * V_HEAD_STRIDE + columns[None, :] * V_DIM_STRIDE,
        value_mask, other=0,
    )
    tl.store(v_out + rows[:, None].to(tl.int64) * VALUE_DIM + columns[None, :], v, value_mask)
    g = tl.load(gate + batch * G_BATCH_STRIDE + token * G_TOKEN_STRIDE + head * G_HEAD_STRIDE,
                active, other=0).to(tl.float32)
    b = tl.load(beta + batch * B_BATCH_STRIDE + token * B_TOKEN_STRIDE + head * B_HEAD_STRIDE,
                active, other=0).to(tl.float32)
    a = tl.load(A_log + head * A_STRIDE, active, other=0).to(tl.float32)
    bias = tl.load(dt_bias + head * DT_STRIDE, active, other=0).to(tl.float32)
    g += bias
    # Match Torch softplus's threshold and preserve small negative-tail values.
    softplus = tl.where(g > 20.0, g, libdevice.log1p(tl.exp(tl.minimum(g, 20.0))))
    tl.store(gate_out + rows, -tl.exp(a) * softplus, active)
    tl.store(beta_out + rows, tl.sigmoid(b), active)


def prepare_gdn(
    query  : torch.Tensor,
    key    : torch.Tensor,
    value  : torch.Tensor,
    gate   : torch.Tensor,
    beta   : torch.Tensor,
    A_log  : torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare strided [B, T, H, D] GDN views in one launch.

    Outputs are contiguous, Q/K heads are expanded to V heads, and Q/K keep
    their input dtype. Gate is per-token FP32 log-decay; beta is FP32 sigmoid.
    """
    assert query.is_cuda and query.ndim == key.ndim == value.ndim == 4
    batch_size, tokens, key_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[2:]
    assert key.shape == query.shape and value.shape[:2] == query.shape[:2]
    assert key_heads > 0 and value_heads % key_heads == 0
    assert gate.shape == beta.shape == (batch_size, tokens, value_heads)
    assert A_log.shape == dt_bias.shape == (value_heads,)
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32)
    for tensor in (key, value):
        assert tensor.dtype == query.dtype and tensor.device == query.device
    for tensor in (gate, beta, A_log, dt_bias):
        assert tensor.device == query.device
        assert tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    q_out = torch.empty((batch_size, tokens, value_heads, key_dim), device=query.device, dtype=query.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    gate_out = torch.empty(gate.shape, device=gate.device, dtype=torch.float32)
    beta_out = torch.empty_like(gate_out)
    rows = batch_size * tokens * value_heads
    if rows:
        block_dim = triton.next_power_of_2(max(key_dim, value_dim))
        _prepare_gdn_kernel[(triton.cdiv(rows, 4),)](
            query, key, value, gate, beta, A_log, dt_bias,
            q_out, k_out, v_out, gate_out, beta_out,
            *query.stride(), *key.stride(), *value.stride(), *gate.stride(), *beta.stride(),
            A_log.stride(0), dt_bias.stride(0), tokens, key_heads, value_heads,
            key_dim, value_dim, rows, batch_size == 1, 4, block_dim,
            num_warps=4, enable_fp_fusion=False,
        )
    return q_out, k_out, v_out, gate_out, beta_out


@triton.jit
def _norm_rope(
    source,
    weight,
    cos_sin,
    position,
    HEAD_DIM      : tl.constexpr,
    ROTARY_DIM    : tl.constexpr,
    NORMALIZE     : tl.constexpr,
    ZERO_CENTERED : tl.constexpr,
    EPS           : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    columns = tl.arange(0, BLOCK_SIZE)
    values = tl.load(source + columns, columns < HEAD_DIM, other=0).to(tl.float32)
    if NORMALIZE:
        scale = tl.rsqrt(tl.sum(values * values, 0) / HEAD_DIM + EPS)
        weights = tl.load(weight + columns, columns < HEAD_DIM, other=0).to(tl.float32)
        if ZERO_CENTERED:
            weights += 1.0
        # Preserve the rounding boundary of RMSNorm followed by in-place RoPE.
        values = (values * scale * weights).to(source.dtype.element_ty).to(tl.float32)
    half = ROTARY_DIM // 2
    partner = tl.where(columns < half, columns + half, columns - half)
    partner = tl.maximum(tl.minimum(partner, BLOCK_SIZE - 1), 0)
    paired = tl.gather(values, partner, 0)
    frequency = columns % half
    cos = tl.load(cos_sin + position * ROTARY_DIM + frequency)
    sin = tl.load(cos_sin + position * ROTARY_DIM + half + frequency)
    rotated = values * cos + tl.where(columns < half, -paired, paired) * sin
    return tl.where(columns < ROTARY_DIM, rotated, values)


@triton.jit
def _qk_norm_rope_store_kernel(
    query, key, value, positions, cos_sin, out_loc,
    k_cache, v_cache, q_weight, k_weight, output,
    Q_TOKEN_STRIDE : tl.constexpr,
    Q_HEAD_STRIDE  : tl.constexpr,
    K_TOKEN_STRIDE : tl.constexpr,
    K_HEAD_STRIDE  : tl.constexpr,
    V_TOKEN_STRIDE : tl.constexpr,
    V_HEAD_STRIDE  : tl.constexpr,
    KC_PAGE_STRIDE : tl.constexpr,
    KC_TOKEN_STRIDE: tl.constexpr,
    KC_HEAD_STRIDE : tl.constexpr,
    VC_PAGE_STRIDE : tl.constexpr,
    VC_TOKEN_STRIDE: tl.constexpr,
    VC_HEAD_STRIDE : tl.constexpr,
    PAGE_SIZE      : tl.constexpr,
    Q_HEADS        : tl.constexpr,
    KV_HEADS       : tl.constexpr,
    HEAD_DIM       : tl.constexpr,
    ROTARY_DIM     : tl.constexpr,
    Q_NORMALIZE    : tl.constexpr,
    K_NORMALIZE    : tl.constexpr,
    Q_ZERO_CENTERED: tl.constexpr,
    K_ZERO_CENTERED: tl.constexpr,
    Q_EPS          : tl.constexpr,
    K_EPS          : tl.constexpr,
    BLOCK_SIZE     : tl.constexpr,
):
    row = tl.program_id(0)
    token = (row // (Q_HEADS + KV_HEADS)).to(tl.int64)
    head = row % (Q_HEADS + KV_HEADS)
    columns = tl.arange(0, BLOCK_SIZE)
    position = tl.load(positions + token).to(tl.int64)
    if head < Q_HEADS:
        source = query + token * Q_TOKEN_STRIDE + head * Q_HEAD_STRIDE
        result = _norm_rope(
            source, q_weight, cos_sin, position,
            HEAD_DIM, ROTARY_DIM, Q_NORMALIZE, Q_ZERO_CENTERED, Q_EPS, BLOCK_SIZE,
        )
        tl.store(output + (token * Q_HEADS + head) * HEAD_DIM + columns,
                 result, columns < HEAD_DIM)
    else:
        head -= Q_HEADS
        source = key + token * K_TOKEN_STRIDE + head * K_HEAD_STRIDE
        result = _norm_rope(
            source, k_weight, cos_sin, position,
            HEAD_DIM, ROTARY_DIM, K_NORMALIZE, K_ZERO_CENTERED, K_EPS, BLOCK_SIZE,
        )
        location = tl.load(out_loc + token).to(tl.int64)
        page, offset = location // PAGE_SIZE, location % PAGE_SIZE
        key_out = k_cache + page * KC_PAGE_STRIDE + offset * KC_TOKEN_STRIDE + head * KC_HEAD_STRIDE
        value_out = v_cache + page * VC_PAGE_STRIDE + offset * VC_TOKEN_STRIDE + head * VC_HEAD_STRIDE
        mask = (columns < HEAD_DIM) & (location >= 0)
        values = tl.load(value + token * V_TOKEN_STRIDE + head * V_HEAD_STRIDE + columns,
                         columns < HEAD_DIM, other=0)
        tl.store(key_out + columns, result, mask)
        tl.store(value_out + columns, values, mask)


def qk_norm_rope_store(
    query    : torch.Tensor,
    key      : torch.Tensor,
    value    : torch.Tensor,
    positions: torch.Tensor,
    cos_sin  : torch.Tensor,
    out_loc  : torch.Tensor,
    k_cache  : torch.Tensor,
    v_cache  : torch.Tensor,
    *,
    q_weight       : torch.Tensor | None = None,
    k_weight       : torch.Tensor | None = None,
    q_eps          : float = 1e-6,
    k_eps          : float = 1e-6,
    q_zero_centered: bool = False,
    k_zero_centered: bool = False,
) -> torch.Tensor:
    """Normalize/rotate QK and scatter final KV to NHD paged cache in one launch.

    Inputs are strided [T, H, D] projection views; caches are [P, S, Hkv, D],
    including frame-backed views. Q is returned contiguous, source QKV is unchanged.
    Negative output locations suppress KV writes (e.g. graph padding).
    """
    assert query.ndim == key.ndim == value.ndim == 3
    tokens, q_heads, head_dim = query.shape
    assert key.shape == value.shape and key.shape[0] == tokens and key.shape[2] == head_dim
    assert k_cache.ndim == v_cache.ndim == 4 and k_cache.shape == v_cache.shape
    assert k_cache.shape[2:] == key.shape[1:]
    assert positions.shape == out_loc.shape == (tokens,)
    assert positions.dtype in (torch.int32, torch.int64)
    assert out_loc.dtype in (torch.int32, torch.int64)
    assert positions.is_contiguous() and out_loc.is_contiguous()
    assert cos_sin.ndim == 2 and cos_sin.is_contiguous() and cos_sin.dtype == torch.float32
    rotary_dim = cos_sin.shape[1]
    assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert query.is_cuda
    for tensor in (key, value, k_cache, v_cache):
        assert tensor.dtype == query.dtype and tensor.device == query.device
        assert tensor.stride(-1) == 1
    assert query.stride(-1) == 1
    for tensor in (positions, cos_sin, out_loc):
        assert tensor.device == query.device
    for weight in (q_weight, k_weight):
        if weight is not None:
            assert weight.shape == (head_dim,) and weight.is_contiguous()
            assert weight.device == query.device

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if tokens == 0:
        return output
    _qk_norm_rope_store_kernel[(tokens * (q_heads + key.shape[1]),)](
        query, key, value, positions, cos_sin, out_loc,
        k_cache, v_cache, q_weight, k_weight, output,
        *query.stride()[:2], *key.stride()[:2], *value.stride()[:2],
        *k_cache.stride()[:3], *v_cache.stride()[:3],
        k_cache.shape[1], q_heads, key.shape[1], head_dim, rotary_dim,
        q_weight is not None, k_weight is not None,
        q_zero_centered, k_zero_centered, q_eps, k_eps,
        triton.next_power_of_2(head_dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _sigmoid_and_mul_kernel(
    input,
    gate,
    output,
    gate_token_stride: tl.constexpr,
    gate_head_stride : tl.constexpr,
    num_heads        : tl.constexpr,
    num_cols         : tl.constexpr,
    BLOCK_SIZE       : tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row % num_heads
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    value = tl.load(input + row * num_cols + offsets, mask=mask)
    gate_value = tl.load(
        gate + token * gate_token_stride + head * gate_head_stride + offsets,
        mask=mask,
    ).to(tl.float32)
    result = value.to(tl.float32) * tl.sigmoid(gate_value)
    tl.store(output + row * num_cols + offsets, result, mask=mask)


def sigmoid_and_mul(
    input: torch.Tensor,
    gate : torch.Tensor,
    out  : torch.Tensor | None = None,
) -> torch.Tensor:
    assert input.shape == gate.shape
    assert input.is_cuda and gate.is_cuda
    assert input.is_contiguous()
    assert gate.stride(-1) == 1
    assert gate.ndim in (2, 3)

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
        assert out.device == input.device and out.dtype == input.dtype
        assert out.is_contiguous()

    num_cols = input.shape[-1]
    block_size = triton.next_power_of_2(num_cols)
    assert block_size <= 65536, (
        f"sigmoid_and_mul width is too large for one Triton block: {num_cols}"
    )
    num_rows = input.numel() // num_cols
    num_heads = gate.shape[-2] if gate.ndim == 3 else 1
    gate_token_stride = gate.stride(-3) if gate.ndim == 3 else gate.stride(-2)
    gate_head_stride = gate.stride(-2) if gate.ndim == 3 else 0
    _sigmoid_and_mul_kernel[(num_rows,)](
        input,
        gate,
        out,
        gate_token_stride=gate_token_stride,
        gate_head_stride=gate_head_stride,
        num_heads=num_heads,
        num_cols=num_cols,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return out
