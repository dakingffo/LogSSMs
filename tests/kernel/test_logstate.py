from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from tests.logstate_helpers import head_view, write_head
from logssms.kernel.linear_attention.logstate import (
    restore_log_state_chains,
)
from logssms.kernel.linear_attention.vendor.fla.fused_recurrent import (
    _frame_byte_offset,
    fused_recurrent_gated_delta_rule_to_logstate,
)
from tests.logstate_helpers import decay_view


@pytest.mark.parametrize("batch_size", [1, 2, 16, 31, 32, 48, 64, 128])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dims", [(128, 128), (32, 32)])
def test_logstate_decode_selects_launch_config(monkeypatch, batch_size, eager, dims):
    from logssms.kernel.linear_attention.vendor.fla import gdn_replay, fused_recurrent

    launches = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid, kwargs))
            return launch

    monkeypatch.setattr(fused_recurrent, "_fused_recurrent_gated_delta_kernel", Kernel("full"))
    monkeypatch.setattr(gdn_replay, "_gdn_replay_kernel", Kernel("tiled"))
    K, V = dims
    q = torch.empty(batch_size, 1, 2, K, dtype=torch.bfloat16)
    v = torch.empty(batch_size, 1, 6, V, dtype=q.dtype)
    a = torch.empty(batch_size, 1, 6, dtype=q.dtype)
    slots = torch.zeros(batch_size, dtype=torch.int32)
    pool = SimpleNamespace(
        layout=SimpleNamespace(head_k_dim=K, head_v_dim=V, num_value_heads=6, log_decay_bytes=1536,
                               num_linear_layers=1, granularity=32),
        frame_bytes=32768, head_payload_bytes=32768,
        allocator=SimpleNamespace(payload_frames=6), page_size=64,
        storage=torch.empty(0, dtype=torch.uint8),
        payload_frame_table=slots, decay_frame_table=slots,
    )
    fused_recurrent_gated_delta_rule_to_logstate(
        q, q, v, a, a, torch.empty(6), torch.empty(6), pool,
        slots, slots, slots, 0, slots, enable_eager_state=eager,
    )
    name, grid, config = launches.pop()
    tiled = not eager and dims == (128, 128) and batch_size >= 32
    assert name == ("tiled" if tiled else "full")
    assert config["BLOCK_K"] == (64 if tiled else K)
    expected_v = 64 if tiled else (8 if V < 128 else (16 if batch_size == 1 else 32))
    if not eager:
        expected_v = max(expected_v, 16)
    assert config["BLOCK_V"] == expected_v
    assert grid == ((V + expected_v - 1) // expected_v, batch_size * 6)
    assert config["num_warps"] == (1 if eager or tiled else 4)
    assert config["num_stages"] == (4 if tiled else 3)


@triton.jit
def _frame_byte_offset_test_kernel(frames, offsets, FRAME_BYTES: tl.constexpr):
    row = tl.program_id(0)
    frame = tl.load(frames + row)
    tl.store(offsets + row, _frame_byte_offset(frame, FRAME_BYTES))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_decode_frame_address_uses_64_bit_arithmetic():
    frame_bytes = 786432  # Qwen3.5-2B LogState frame size
    frames = torch.tensor([0, 2730, 2731, 2988, 3563], dtype=torch.int32, device="cuda")
    offsets = torch.empty(len(frames), dtype=torch.int64, device="cuda")
    _frame_byte_offset_test_kernel[(len(frames),)](frames, offsets, frame_bytes)
    torch.testing.assert_close(offsets, frames.to(torch.int64) * frame_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("granularity", [16, 32, 64])
@pytest.mark.parametrize("head_ratio", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(128, 128), (48, 80)])
def test_logstate_restore_resolves_fragmented_addresses(
    granularity, head_ratio, dtype, dims,
):
    torch.manual_seed(59)
    K, V = dims
    H, HV, layers = 2, 2 * head_ratio, 2
    device = torch.device("cuda")
    pool = MHALogStatePool(
        num_frames=256, page_size=64, granularity=granularity,
        num_attention_layers=1, num_kv_heads=2, head_dim=64,
        num_linear_layers=layers, num_key_heads=H, num_value_heads=HV,
        head_k_dim=K, head_v_dim=V, conv_dim=32, state_len=3,
        device=device, dtype=dtype,
    )
    blocked = pool.reserve(kv=128)
    assert blocked is not None
    pool.release_kv(blocked[0][::2])
    allocated = pool.reserve(logs=1, works=8)
    assert allocated is not None
    log = allocated[1][0]
    base, restore_zero, restore_base, at16, at35, at64, duplicate1, duplicate2 = allocated[2]
    payload = pool.allocator.slot(log).payload
    assert len(payload) > 1 and all(abs(a - b) > 1 for a, b in zip(payload, payload[1:]))

    initials, expected = [], []
    for layer in range(layers):
        initial = torch.randn(HV, K, V, device=device, dtype=dtype) * .005
        pool.write_states(torch.tensor([base], device=device), layer, initial[None])
        keys = torch.nn.functional.normalize(torch.randn(64, H, K, device=device), dim=-1).to(dtype)
        values = torch.randn(64, HV, V, device=device, dtype=dtype) * .02
        gates = -torch.rand(64, HV, device=device) * .03
        for head in range(HV):
            write_head(pool, log, layer, head, keys[:, head // head_ratio].flatten(),
                       state=False, index=slice(None, 64 * K))
            write_head(pool, log, layer, head, values[:, head].flatten(),
                       state=False, index=slice(64 * K, None))
        decay_view(pool, log)[layer].copy_(gates)
        expanded = keys.float().repeat_interleave(head_ratio, 1)
        zero = torch.zeros(HV, K, V, device=device)
        snapshots = {}
        for token in range(64):
            zero *= gates[token].exp()[:, None, None]
            zero += expanded[token, :, :, None] * values[token].float()[:, None, :]
            if token + 1 in (16, 35, 64):
                snapshots[token + 1] = zero.clone()
        state = initial.float().clone()
        for token in range(9, 47):
            state *= gates[token].exp()[:, None, None]
            state += expanded[token, :, :, None] * values[token].float()[:, None, :]
        initials.append(initial)
        expected.append((snapshots, state))

    segments = [(log, 0, 16), (log, 16, 35), (log, 35, 35), (log, 35, 64)]
    restore_log_state_chains(pool, [
        (None, segments, restore_zero),
        (base, [(log, 9, 47)], restore_base),
    ])
    restore_log_state_chains(pool, [
        (None, [(log, 0, 16)], at16),
        (None, [(log, 0, 16), (log, 16, 35)], at35),
        (None, segments, at64),
        (base, [(log, 9, 47)], duplicate1),
        (base, [(log, 9, 47)], duplicate2),
    ])
    for layer, (snapshots, state) in enumerate(expected):
        for slot, truth in (
            (restore_zero, snapshots[64]), (restore_base, state),
            (at16, snapshots[16]), (at35, snapshots[35]), (at64, snapshots[64]),
            (duplicate1, state), (duplicate2, state),
        ):
            actual = pool.read_states(torch.tensor([slot], device=device), layer)[0].float()
            torch.testing.assert_close(actual, truth, rtol=3e-2, atol=3e-4)
        actual_base = pool.read_states(torch.tensor([base], device=device), layer)[0]
        torch.testing.assert_close(actual_base, initials[layer], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dims", [(128, 128), (256, 128)])
def test_logstate_reconstruction_matches_serial_recurrence(dims):
    torch.manual_seed(7)
    K, V = dims
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=256,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=K,
        head_v_dim=V,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    logs = allocated[1]
    base, output = allocated[2]
    expected = torch.empty((2, 2, K, V), dtype=torch.float32, device="cuda")
    for layer in range(2):
        for head in range(2):
            initial = torch.randn((K, V), device="cuda") * 0.01
            write_head(pool, base, layer, head, initial.to(torch.bfloat16), state=True)
            expected[layer, head] = initial.to(torch.bfloat16).float()
    for slot in logs:
        for layer in range(2):
            for head in range(2):
                keys = (torch.randn((64, K), device="cuda") * 0.01).to(torch.bfloat16)
                values = (torch.randn((64, V), device="cuda") * 0.01).to(torch.bfloat16)
                write_head(pool, slot, layer, head, keys.flatten(), state=False, index=slice(None, 64 * K, None))
                write_head(pool, slot, layer, head, values.flatten(), state=False, index=slice(64 * K, None, None))
                decay = torch.empty((64,), device="cuda").uniform_(0.98, 0.999)
                decay_view(pool, slot)[layer, :, head].copy_(decay.log())
                for token in range(64):
                    expected[layer, head] *= decay[token]
                    expected[layer, head] += (
                        keys[token].float()[:, None]
                        * values[token].float()[None, :]
                    )

    restore_log_state_chains(
        pool, [(base, [(logs[0], 0, 64), (logs[1], 0, 64)], output)]
    )
    for layer in range(2):
        for head in range(2):
            torch.testing.assert_close(
                head_view(pool, output, layer, head, state=True),
                expected[layer, head].to(torch.bfloat16),
                rtol=3e-2, atol=3e-2,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_restore_keeps_concurrent_requests_separate():
    torch.manual_seed(31)
    pool = MHALogStatePool(
        num_frames=256,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=2,
        head_dim=256,
        num_linear_layers=2,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=20,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=8, works=8)
    assert allocated is not None
    logs, works = allocated[1:]
    expected = torch.zeros((8, 2, 2, 128, 128), device="cuda")
    for row, slot in enumerate(logs):
        for layer in range(2):
            for head in range(2):
                keys = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                values = (torch.randn((64, 128), device="cuda") * 0.01).to(torch.bfloat16)
                write_head(pool, slot, layer, head, keys.flatten(), state=False, index=slice(None, 64 * 128, None))
                write_head(pool, slot, layer, head, values.flatten(), state=False, index=slice(64 * 128, None, None))
                decay = torch.empty((64,), device="cuda").uniform_(0.98, 0.999)
                decay_view(pool, slot)[layer, :, head].copy_(decay.log())
                for token in range(64):
                    expected[row, layer, head] *= decay[token]
                    expected[row, layer, head] += (
                        keys[token].float()[:, None]
                        * values[token].float()[None, :]
                    )

    restore_log_state_chains(
        pool,
        [(None, [(log, 0, 64)], work) for log, work in zip(logs, works)],
    )
    for row, slot in enumerate(works):
        for layer in range(2):
            for head in range(2):
                torch.testing.assert_close(
                    head_view(pool, slot, layer, head, state=True),
                    expected[row, layer, head].to(torch.bfloat16),
                    rtol=3e-2, atol=3e-2,
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_restore_keeps_carry_across_long_chains():
    pool = MHALogStatePool(
        num_frames=256,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=64,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=11, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    chains = []
    for row, row_logs in enumerate((logs[:9], logs[9:])):
        for slot in row_logs:
            write_head(pool, slot, 0, 0, 0.01 * (row + 1), state=False, index=slice(None, 64 * 32, None))
            write_head(pool, slot, 0, 0, 0.01, state=False, index=slice(64 * 32, None, None))
            decay_view(pool, slot)[0, :, 0].zero_()
        chains.append((
            None,
            [(slot, 0, 64) for slot in row_logs],
            works[row],
        ))

    restore_log_state_chains(pool, chains)
    for row, count in enumerate((9, 2)):
        expected = 64 * count * 0.01 * (row + 1) * 0.01
        torch.testing.assert_close(
            head_view(pool, works[row], 0, 0, state=True),
            torch.full((32, 32), expected, device="cuda", dtype=torch.bfloat16),
            rtol=5e-2, atol=1e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_log_reconstruction_keeps_later_branches_readable():
    pool = MHALogStatePool(
        num_frames=128,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=64,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=7, works=7)
    assert allocated is not None
    slots = allocated[1]
    works = allocated[2]
    for index, slot in enumerate(slots):
        write_head(pool, slot, 0, 0, 0.01 * (index + 1), state=False, index=slice(None, 64 * 32, None))
        write_head(pool, slot, 0, 0, 0.01, state=False, index=slice(64 * 32, None, None))
        decay_view(pool, slot)[0, :, 0].zero_()

    restore_log_state_chains(pool, [
        (None, [(slots[0], 0, 64)], works[0]),
        *((None, [(slots[0], 0, 64), (slot, 0, 64)], work)
          for slot, work in zip(slots[1:], works[1:])),
    ])

    for index, slot in enumerate(works):
        expected = 64 * 0.01 * 0.01
        if index:
            expected += 64 * 0.01 * (index + 1) * 0.01
        torch.testing.assert_close(
            head_view(pool, slot, 0, 0, state=True),
            torch.full((32, 32), expected, device="cuda", dtype=torch.bfloat16),
            rtol=5e-2, atol=1e-3,
        )


@pytest.mark.parametrize("granularity", [32, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_decode_replays_at_fixed_sixteen_token_intervals(granularity):
    torch.manual_seed(39)
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=granularity,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=128,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=128,
        state_len=3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    for slot in works:
        pool.zero_state(slot)
    eager_log = torch.tensor([logs[0]], dtype=torch.int32, device="cuda")
    eager_work = torch.tensor([works[0]], dtype=torch.int32, device="cuda")
    replay_log = torch.tensor([logs[1]], dtype=torch.int32, device="cuda")
    replay_work = torch.tensor([works[1]], dtype=torch.int32, device="cuda")
    A_log = torch.zeros(1, device="cuda")
    dt_bias = torch.zeros_like(A_log)
    committed = head_view(pool, works[1], 0, 0, state=True).clone()

    for offset in range(granularity):
        query = torch.randn(1, 1, 1, 128, dtype=pool.dtype, device="cuda")
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        gate = torch.zeros(1, 1, 1, dtype=pool.dtype, device="cuda")
        beta = torch.zeros_like(gate)
        position = torch.tensor([offset], dtype=torch.int32, device="cuda")
        count = torch.tensor([offset % 16], dtype=torch.int32, device="cuda")
        eager = fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias, pool,
            eager_work, eager_log, position, 0, count,
            enable_eager_state=True,
        )
        replay = fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias, pool,
            replay_work, replay_log, position, 0, count,
        )
        torch.testing.assert_close(replay, eager, atol=0.04, rtol=0.04)
        if (offset + 1) % 16 == 0:
            torch.testing.assert_close(
                head_view(pool, works[1], 0, 0, state=True),
                head_view(pool, works[0], 0, 0, state=True),
                atol=0.04, rtol=0.04,
            )
            committed = head_view(pool, works[1], 0, 0, state=True).clone()
        else:
            assert torch.equal(
                head_view(pool, works[1], 0, 0, state=True), committed
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("dims", [(128, 128), (80, 48)])
def test_logstate_decode_mixed_flush_matches_fp32_recurrence(dtype, use_graph, dims, eager=False, batch_size=4):
    torch.manual_seed(53)
    bs, H, HV = batch_size, 2, 6
    K, V = dims
    layer = 1
    pool = MHALogStatePool(
        num_frames=max(192, bs * 24), page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=1, head_dim=128,
        num_linear_layers=2, num_key_heads=H, num_value_heads=HV,
        head_k_dim=K, head_v_dim=V, conv_dim=(2 * H + HV) * K, state_len=3,
        device=torch.device("cuda"), dtype=dtype,
    )
    allocation = pool.reserve(logs=bs, works=bs)
    assert allocation is not None
    logs, works = allocation[1:]
    log_slots = torch.tensor(logs, dtype=torch.int32, device="cuda")
    work_slots = torch.tensor(works, dtype=torch.int32, device="cuda")
    positions = torch.zeros(bs, dtype=torch.int32, device="cuda")
    counts = torch.zeros_like(positions)
    initial = torch.randn(bs, HV, K, V, dtype=dtype, device="cuda") * 0.02
    keys = torch.randn(bs, 64, H, K, dtype=dtype, device="cuda").repeat_interleave(HV // H, 2) * 0.02
    values = torch.randn(bs, 64, HV, V, dtype=dtype, device="cuda") * 0.02
    alpha = torch.empty(bs, 64, HV, device="cuda").uniform_(0.95, 0.999)
    query = torch.randn(bs, 1, H, K, dtype=dtype, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn(bs, 1, HV, V, dtype=dtype, device="cuda")
    gate = torch.randn(bs, 1, HV, dtype=dtype, device="cuda")
    beta = torch.randn_like(gate)
    gate, beta = torch.cat((gate, beta), dim=-1).chunk(2, dim=-1)
    A_log = torch.randn(HV, device="cuda") * 0.1
    dt_bias = torch.randn_like(A_log) * 0.1

    def reset_pool():
        pool.storage.fill_(0x2A)
        pool.write_states(work_slots, layer, initial)
        for row, slot in enumerate(logs):
            for head in range(HV):
                write_head(pool, slot, layer, head, keys[row, :, head].flatten(), state=False, index=slice(None, 64 * K, None))
                write_head(pool, slot, layer, head, values[row, :, head].flatten(), state=False, index=slice(64 * K, None, None))
            decay_view(pool, slot)[layer].copy_(alpha[row].log())

    def run():
        return fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias, pool,
            work_slots, log_slots, positions, layer, counts, enable_eager_state=eager,
        )

    reset_pool()
    output = run()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()

    q = query[:, 0].float()
    k = key[:, 0].float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(HV // H, dim=1)
    k = k.repeat_interleave(HV // H, dim=1)
    log_decay = -A_log.exp() * torch.nn.functional.softplus(gate[:, 0].float() + dt_bias)

    for pattern in ((0, 7, 15, 31), (15, 0, 16, 63), (32, 47, 62, 5)):
        offsets = (pattern * ((bs + 3) // 4))[:bs]
        positions.copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
        counts.copy_(positions % 16)
        reset_pool()
        previous = initial.float().clone()
        for row, position in enumerate(offsets):
            for token in range(position if eager else position - position % 16, position):
                previous[row] *= alpha[row, token, :, None, None]
                previous[row] += (
                    keys[row, token].float()[..., :, None]
                    * values[row, token].float()[..., None, :]
                )
        previous *= log_decay.exp()[..., None, None]
        residual = (value[:, 0].float() - torch.einsum("bhkv,bhk->bhv", previous, k))
        residual *= beta[:, 0].float().sigmoid()[..., None]
        expected_state = previous + k[..., :, None] * residual[..., None, :]
        expected_output = torch.einsum("bhkv,bhk->bhv", expected_state, q)
        if use_graph:
            graph.replay()
        else:
            output = run()
        torch.testing.assert_close(output[:, 0].float(), expected_output, rtol=0.03, atol=0.005)
        observed = pool.read_states(work_slots, layer)
        for row, position in enumerate(offsets):
            if eager or (position + 1) % 16 == 0:
                torch.testing.assert_close(observed[row].float(), expected_state[row], rtol=0.03, atol=0.01)
            else:
                torch.testing.assert_close(observed[row], initial[row], rtol=0, atol=0)
            for head in range(HV):
                log = head_view(pool, logs[row], layer, head, state=False)
                stored_keys = log[:64 * K].view(64, K)
                stored_values = log[64 * K:].view(64, V)
                torch.testing.assert_close(stored_keys[position].float(), k[row, head], rtol=0.01, atol=0.001)
                torch.testing.assert_close(stored_values[position].float(), residual[row, head], rtol=0.03, atol=0.01)
                # Flush must not discard the logs used for prefix restoration.
                torch.testing.assert_close(stored_keys[:position], keys[row, :position, head], rtol=0, atol=0)
                torch.testing.assert_close(stored_values[:position], values[row, :position, head], rtol=0, atol=0)
            torch.testing.assert_close(decay_view(pool, logs[row])[layer, position], log_decay[row], rtol=1e-5, atol=1e-5)
            for head in range(HV):
                for slot in (logs[row], works[row]):
                    untouched = head_view(pool, slot, 0, head, state=slot == works[row])
                    assert torch.all(untouched.view(torch.uint8) == 0x2A)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [32, 64])
def test_shared_key_tiled_replay_matches_fp32_recurrence_with_graph(batch_size):
    test_logstate_decode_mixed_flush_matches_fp32_recurrence(
        torch.bfloat16, True, (128, 128), batch_size=batch_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(128, 128), (80, 48)])
def test_shared_key_eager_decode_matches_fp32_recurrence_with_graph(dtype, dims):
    test_logstate_decode_mixed_flush_matches_fp32_recurrence(dtype, True, dims, eager=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_preserves_log_decay_beyond_alpha_underflow():
    pool = MHALogStatePool(
        num_frames=64, page_size=64, granularity=32,
        num_attention_layers=1, num_kv_heads=1, head_dim=128,
        num_linear_layers=1, num_key_heads=1, num_value_heads=1,
        head_k_dim=128, head_v_dim=128, conv_dim=128, state_len=3,
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    allocation = pool.reserve(logs=1, works=1)
    assert allocation is not None
    log, work = allocation[1][0], allocation[2][0]
    pool.zero_state(work)
    qkv = torch.ones(1, 1, 1, 128, device="cuda", dtype=pool.dtype)
    A_log = torch.zeros(1, device="cuda")
    gate = torch.full((1, 1, 1), 1000, device="cuda", dtype=pool.dtype)
    beta = torch.zeros_like(gate)
    output = fused_recurrent_gated_delta_rule_to_logstate(
        qkv, qkv, qkv, gate, beta, A_log, A_log, pool,
        torch.tensor([work], dtype=torch.int32, device="cuda"),
        torch.tensor([log], dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"), 0,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(decay_view(pool, log)[0, 0, 0], -gate[0, 0, 0].float())
    assert torch.isfinite(output).all()
    output = fused_recurrent_gated_delta_rule_to_logstate(
        qkv, qkv, qkv, beta, beta, A_log, A_log, pool,
        torch.tensor([work], dtype=torch.int32, device="cuda"),
        torch.tensor([log], dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"), 0,
        torch.ones(1, dtype=torch.int32, device="cuda"),
    )
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_logstate_restore_prefetches_partial_chunks(dtype: torch.dtype):
    torch.manual_seed(41)
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=32,
        num_linear_layers=1,
        num_key_heads=1,
        num_value_heads=1,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=32,
        state_len=3,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    allocated = pool.reserve(logs=1, works=2)
    assert allocated is not None
    log_slot = allocated[1][0]
    work_slot, checkpoint_slot = allocated[2]
    keys = (torch.randn((64, 32), device="cuda") * 0.01).to(dtype)
    values = (torch.randn((64, 32), device="cuda") * 0.01).to(dtype)
    decays = torch.empty((64,), device="cuda").uniform_(0.98, 0.999)
    write_head(pool, log_slot, 0, 0, keys.flatten(), state=False, index=slice(None, 64 * 32, None))
    write_head(pool, log_slot, 0, 0, values.flatten(), state=False, index=slice(64 * 32, None, None))
    decay_view(pool, log_slot)[0, :, 0].copy_(decays.log())

    expected = torch.zeros((32, 32), device="cuda")
    for token in range(35):
        expected *= decays[token]
        expected += keys[token].float()[:, None] * values[token].float()[None, :]
        if token == 8:
            expected_at_9 = expected.clone()

    restore_log_state_chains(
        pool,
        [(None, [(log_slot, 0, 9)], work_slot),
         (None, [(log_slot, 0, 9), (log_slot, 9, 35)], checkpoint_slot)],
    )
    torch.testing.assert_close(
        head_view(pool, work_slot, 0, 0, state=True),
        expected_at_9.to(dtype), rtol=4e-2, atol=4e-2,
    )
    torch.testing.assert_close(
        head_view(pool, checkpoint_slot, 0, 0, state=True),
        expected.to(dtype), rtol=4e-2, atol=4e-2,
    )
