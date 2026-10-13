import pytest
import torch

from logssms.cache.logstate_pool.mha_pool import MHALogStatePool
from tests.logstate_helpers import head_view
from logssms.core import Batch, Context, Request, set_global_ctx
from logssms.layers.gated_delta import GatedDeltaConfig
from logssms.layers.linear_attention_backend import GatedDeltaLogStateBackend
from logssms.kernel.linear_attention.logstate import construct_log_state_chains
from logssms.kernel.linear_attention.logstate import restore_log_state_chains
from logssms.kernel.linear_attention.logstate import copy_states
from logssms.kernel.linear_attention.vendor.fla import (
    fused_recurrent_gated_delta_rule_to_logstate,
    fused_recurrent_gated_delta_rule,
)
from tests.logstate_helpers import decay_view
from tests.utils import clear_global_ctx


def _make_request(cached_len: int, device_len: int) -> Request:
    return Request(
        input_ids=torch.zeros(device_len, dtype=torch.int32),
        table_idx=0,
        cached_len=cached_len,
        output_len=1,
        uid=0,
        sampling_params=None,  # type: ignore
        cache_handle=None,  # type: ignore
        max_device_len=device_len + 8,
    )


def _make_pool(device: torch.device) -> MHALogStatePool:
    pool = MHALogStatePool(
        num_frames=64,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=device,
        dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=2, works=2)
    assert allocated is not None
    assert allocated[1] == [0, 1] and allocated[2] == [2, 3]
    return pool


def _make_context(device: torch.device) -> Context:
    ctx = Context(page_size=64)
    ctx.linear_state = _make_pool(device)
    ctx.page_table = torch.arange(128, dtype=torch.int32, device=device).repeat(2, 1)
    ctx.state_table = torch.full(
        (2, 128), -1, dtype=torch.int32, device=device
    )
    ctx.state_table[0, :64].fill_(0)
    ctx.state_table[0, 64:].fill_(1)
    ctx.state_table[1].fill_(1)
    ctx.work_state_table = torch.tensor(
        [[2], [3]], dtype=torch.int32, device=device
    )
    ctx.linear_state.zero_state(2)
    ctx.linear_state.zero_work_conv(2)
    return ctx


def test_logstate_decode_metadata_uses_work_and_log_slots() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=20, device_len=21)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([20], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.work_slots.tolist() == [2]
    assert metadata.log_slots.tolist() == [0]
    assert metadata.token_offsets.tolist() == [20]
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]
    assert metadata.work_slots.dtype == torch.int32
    assert metadata.log_slots.dtype == torch.int32
    assert metadata.token_offsets.dtype == torch.int32
    assert metadata.conv_read_slots.dtype == torch.int32
    assert metadata.conv_read_granules.dtype == torch.int32


def test_logstate_new_request_reads_conv_from_work_slot() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=0, device_len=1)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([0], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]


def test_logstate_uses_physical_offsets_after_unaligned_prefix_hit() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    ctx.page_table[0, 16:32] = 64 + torch.arange(
        16, dtype=torch.int32, device=device
    )
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=16, device_len=32)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.arange(16, 32, dtype=torch.int32, device=device)
    batch.out_loc = ctx.page_table[0, 16:32]

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        metadata = batch.linear_attn_metadata
        assert metadata.token_offsets.tolist() == list(range(16))
        assert metadata.conv_snapshot_rows.tolist() == [15]
        assert metadata.conv_write_granules.tolist() == [0]

        decode = Batch(reqs=[_make_request(17, 18)], phase="decode")
        decode.padded_reqs = decode.reqs
        decode.positions = torch.tensor([17], dtype=torch.int32, device=device)
        decode.out_loc = ctx.page_table[0, 17:18]
        backend.prepare_metadata(decode)
        decoded = decode.linear_attn_metadata
        assert decoded.token_offsets.tolist() == [1]
        assert decoded.conv_read_granules.tolist() == [0]
    finally:
        clear_global_ctx()


def test_recycled_checkpoint_prefill_metadata_skips_log_writes() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    reserved = ctx.linear_state.reserve(works=1)
    assert reserved is not None
    checkpoint_slot = reserved[2][0]
    ctx.state_table[0].fill_(-1)
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=0, device_len=128)
    batch = Batch(reqs=[req], phase="prefill")
    batch.positions = torch.arange(128, dtype=torch.int32, device=device)
    batch.out_loc = ctx.page_table[0]
    batch.logstate_first_offsets = [0]
    batch.logstate_checkpoint_writes = [(127, checkpoint_slot)]
    batch.logstate_checkpoint_ranges = [(0, 0, 128)]

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.log_row_indices.numel() == 0
    assert metadata.conv_snapshot_rows.tolist() == [127]
    assert metadata.conv_write_slots.numel() == 0
    assert metadata.checkpoint_slots.tolist() == [checkpoint_slot]
    assert metadata.checkpoint_state_indices.tolist() == [7]
    assert metadata.checkpoint_snapshot_indices.tolist() == [0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_direct_checkpoint_prefill_matches_log_prefill(reuse, chunk_size) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(213)
    mixed_qkv = torch.randn(128, 192, device=device, dtype=dtype) * 0.1
    gate = torch.randn(128, 2, device=device, dtype=dtype) * 0.1
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(192, 1, 4, device=device, dtype=dtype) * 0.02
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(direct: bool):
        clear_global_ctx()
        ctx = _make_context(device)
        pool = ctx.linear_state
        checkpoint_slot = None
        if direct:
            if reuse:
                checkpoint_slot = 0
            else:
                reserved = pool.reserve(works=1)
                assert reserved is not None
                checkpoint_slot = reserved[2][0]
            ctx.state_table[0].fill_(-1)
        backend = GatedDeltaLogStateBackend()
        set_global_ctx(ctx)
        try:
            outputs = []
            for begin in range(0, 128, chunk_size):
                end = begin + chunk_size
                batch = Batch(reqs=[_make_request(begin, end)], phase="prefill")
                batch.positions = torch.arange(begin, end, device=device, dtype=torch.int32)
                batch.out_loc = ctx.page_table[0, begin:end]
                batch.logstate_first_offsets = [0]
                if direct:
                    batch.logstate_checkpoint_ranges = [(0, 0, 128)]
                    if end == 128:
                        batch.logstate_checkpoint_writes = [(chunk_size - 1, checkpoint_slot)]
                backend.prepare_metadata(batch)
                outputs.append(backend.forward(
                    mixed_qkv[begin:end], gate[begin:end], beta[begin:end],
                    conv_weight, A_log, dt_bias, config, batch,
                ))
            output = torch.cat(outputs)
            state_slot = 2 if not direct else checkpoint_slot
            state = pool.read_states(
                torch.tensor([state_slot], device=device), 0
            ).clone()
            conv_slot, conv_granule = (1, 3) if not direct else (checkpoint_slot, 3 if reuse else 0)
            conv = pool.read_conv(
                torch.tensor([conv_slot], device=device),
                torch.tensor([conv_granule], device=device), 0,
            ).clone()
            torch.cuda.synchronize()
            return output.clone(), state, conv
        finally:
            clear_global_ctx()

    log_output, log_state, log_conv = run(False)
    direct_output, direct_state, direct_conv = run(True)
    torch.testing.assert_close(direct_output, log_output, rtol=0, atol=0)
    torch.testing.assert_close(direct_state, log_state, rtol=0, atol=0)
    torch.testing.assert_close(direct_conv, log_conv, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memorized_prefill_reuses_gpu_tail_and_matches_full_state():
    from collections import Counter
    from tests.core.test_logstate_cache_manager import make_manager, new_req
    from logssms.cache.prefix_cache.logstate import LogStateKind

    torch.manual_seed(321)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    x = torch.randn(321, 192, device=device, dtype=dtype) * .1
    gate = torch.randn(321, 2, device=device, dtype=dtype) * .1
    beta = torch.randn_like(gate)
    weight = torch.randn(192, 1, 4, device=device, dtype=dtype) * .02
    A, bias = torch.zeros(2, device=device), torch.zeros(2, device=device)

    def forward(manager, req):
        clear_global_ctx()
        ctx = Context(page_size=64)
        ctx.linear_state = manager.logstate_pool
        ctx.page_table = manager.page_table
        ctx.state_table = manager.state_table
        ctx.work_state_table = manager.work_state_table
        set_global_ctx(ctx)
        backend = GatedDeltaLogStateBackend()
        batch = Batch([req], "prefill")
        manager.allocate([req], phase="prefill")
        manager.prepare_batch(batch)
        begin, end = req.cached_len, req.device_len
        batch.positions = torch.arange(begin, end, device=device, dtype=torch.int32)
        batch.out_loc = manager.page_table[req.table_idx, begin:end]
        try:
            backend.prepare_metadata(batch)
            output = backend.forward(x[begin:end], gate[begin:end], beta[begin:end],
                                     weight, A, bias, config, batch)
            torch.cuda.synchronize()
            return output
        finally:
            clear_global_ctx()

    reference = make_manager(num_pages=32, num_states=32, device=device, granularity=32)
    reference_req = new_req(reference, torch.arange(321, dtype=torch.int32))
    expected_output = forward(reference, reference_req).clone()
    expected_state = reference.logstate_pool.read_states(
        torch.tensor([reference._work_slots[0]], device=device), 0,
    ).clone()

    manager = make_manager(num_pages=32, num_states=32, device=device, granularity=32)
    first = new_req(manager, torch.arange(256, dtype=torch.int32))
    forward(manager, first)
    pool = manager.logstate_pool
    expected_checkpoint = pool.read_states(torch.tensor([manager._work_slots[0]], device=device), 0).clone()
    first.cached_len = 256
    manager.cache_req(first, finished=True)
    entries = manager.tree.iter_entries()
    reuse = new_req(manager, torch.arange(129, dtype=torch.int32), table_idx=1)
    manager.prepare_batch(Batch([reuse], "prefill"))
    manager.cache_req(reuse, finished=True)
    leaf = manager.tree.entry_owner(entries[-1])
    manager.recycle_bin.remember(manager.tree, leaf)
    counts = Counter(leaf._kv_page_counts)
    eviction = manager.tree.evict_nodes((leaf,))
    manager._release_cached_kv(counts)
    pool.release_slots([entry.slot for entry in eviction.entries])
    tail_slot = entries[2].slot

    resumed = new_req(manager, torch.arange(321, dtype=torch.int32))
    assert resumed.cached_len == 192
    actual_output = forward(manager, resumed)
    assert manager._recycled_restores[0].slots[256] == tail_slot
    actual_checkpoint = pool.read_states(torch.tensor([tail_slot], device=device), 0)
    actual_state = pool.read_states(torch.tensor([manager._work_slots[0]], device=device), 0)
    torch.testing.assert_close(actual_output, expected_output[192:], rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(actual_checkpoint, expected_checkpoint, rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-2, atol=3e-3)
    resumed.cached_len = 320
    manager.cache_req(resumed, finished=True)
    assert [(entry.begin, entry.end, entry.kind) for entry in manager.tree.iter_entries()] == [
        (0, 128, LogStateKind.CHECKPOINT), (128, 256, LogStateKind.CHECKPOINT),
        (256, 320, LogStateKind.LOG),
    ]
    manager.logstate_pool.poll_reclamation()
    manager.check_integrity()


def test_logstate_prefill_metadata_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOGSSMS_PROFILE_SCHEDULER", "1")
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=16, device_len=32)
    batch = Batch(reqs=[req], phase="prefill")
    batch.positions = torch.arange(16, 32, dtype=torch.int32, device=device)
    batch.out_loc = ctx.page_table[0, 16:32]
    batch.logstate_first_offsets = [16]

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        profile = backend.profile_result()
    finally:
        clear_global_ctx()

    assert profile is not None
    assert profile["aggregate"]["prefill_batches"] == 1
    assert profile["aggregate"]["item_calls"] == 0
    assert profile["aggregate"]["metadata_cpu_ms"] >= 0
    assert profile["aggregate"]["repeat_interleave_cpu_ms"] >= 0
    assert profile["aggregate"]["item_cpu_ms"] >= 0


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_logstate_unmapped_cached_prefix_reads_work_conv(phase: str) -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    ctx.state_table[0, :16].fill_(-1)
    backend = GatedDeltaLogStateBackend()
    req = _make_request(cached_len=16, device_len=17)
    batch = Batch(reqs=[req], phase=phase)
    if phase == "decode":
        batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([16], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
    finally:
        clear_global_ctx()

    metadata = batch.linear_attn_metadata
    assert metadata.conv_read_slots.tolist() == [2]
    assert metadata.conv_read_granules.tolist() == [0]


def test_logstate_decode_graph_metadata_uses_int32_capture_buffers() -> None:
    clear_global_ctx()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _make_context(device)
    backend = GatedDeltaLogStateBackend()
    real_req = _make_request(cached_len=20, device_len=21)
    dummy_req = _make_request(cached_len=0, device_len=1)
    dummy_req.table_idx = 1

    capture_batch = Batch(reqs=[dummy_req, dummy_req], phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs
    capture_batch.positions = torch.zeros(2, dtype=torch.int32, device=device)
    capture_batch.out_loc = capture_batch.positions
    batch = Batch(reqs=[real_req], phase="decode")
    batch.padded_reqs = [real_req, dummy_req]
    batch.positions = torch.tensor([20, 0], dtype=torch.int32, device=device)
    batch.out_loc = batch.positions

    set_global_ctx(ctx)
    try:
        backend.init_capture_graph(max_seq_len=128, bs_list=[2])
        backend.prepare_for_capture(capture_batch)
        backend.prepare_metadata(batch)
        backend.prepare_for_replay(batch)
    finally:
        clear_global_ctx()

    assert backend.capture is not None
    assert backend.capture.work_slots.tolist() == [2, 3]
    assert backend.capture.log_slots.tolist() == [0, 1]
    assert backend.capture.token_offsets.tolist() == [20, 0]
    assert backend.capture.conv_read_slots.tolist() == [2, 3]
    assert backend.capture.conv_read_granules.tolist() == [0, 0]
    assert backend.capture.work_slots.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_logstate_recurrent_auto_tile_matches_reference_for_128d_heads(
    batch_size: int,
) -> None:
    device = torch.device("cuda")
    pool = MHALogStatePool(
        num_frames=max(64, batch_size * 10 + 1), page_size=64, granularity=16,
        num_attention_layers=1, num_kv_heads=1, head_dim=128,
        num_linear_layers=1, num_key_heads=2, num_value_heads=2,
        head_k_dim=128, head_v_dim=128, conv_dim=768, state_len=3,
        device=device, dtype=torch.bfloat16,
    )
    allocated = pool.reserve(logs=batch_size, works=batch_size)
    assert allocated is not None
    torch.manual_seed(96)
    query = torch.randn(batch_size, 1, 2, 128, device=device, dtype=pool.dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn(batch_size, 1, 2, device=device, dtype=pool.dtype)
    beta = torch.randn_like(gate)
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)
    work = torch.tensor(allocated[2], device=device, dtype=torch.int32)
    logs = torch.tensor(allocated[1], device=device, dtype=torch.int32)
    offsets = torch.zeros(batch_size, device=device, dtype=torch.int32)
    initial = torch.randn(batch_size, 2, 128, 128, device=device, dtype=pool.dtype)
    pool.write_states(work, 0, initial)
    expected, expected_state, *_ = fused_recurrent_gated_delta_rule(
        query, key, value, gate, beta, A_log, dt_bias, initial
    )
    actual = fused_recurrent_gated_delta_rule_to_logstate(
        query, key, value, gate, beta, A_log, dt_bias,
        pool, work, logs, offsets, 0,
        torch.zeros(batch_size, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pool.read_states(work, 0), initial, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("first_offset", [0, 20, 48])
def test_decode_replays_log_and_commits_work_only_at_granule_end(
    first_offset: int,
) -> None:
    device = torch.device("cuda")
    pool = _make_pool(device)
    torch.manual_seed(103)
    work = torch.tensor([2], dtype=torch.int32, device=device)
    log = torch.tensor([0], dtype=torch.int32, device=device)
    initial = torch.randn(1, 2, 32, 32, device=device, dtype=pool.dtype) * 0.1
    pool.write_states(work, 0, initial)
    baseline = initial
    length = 16 - first_offset % 16
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)
    for step in range(length):
        query = torch.randn(1, 1, 2, 32, device=device, dtype=pool.dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        gate = torch.randn(1, 1, 2, device=device, dtype=pool.dtype)
        beta = torch.randn_like(gate)
        expected, states = fused_recurrent_gated_delta_rule(
            query, key, value, gate, beta, A_log, dt_bias, baseline
        )[:2]
        actual = fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias,
            pool, work, log,
            torch.tensor([first_offset + step], dtype=torch.int32, device=device),
            0, torch.tensor([step], dtype=torch.int32, device=device),
        )
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        baseline = states[:, 0]
        stored = pool.read_states(work, 0)
        if step + 1 < length:
            torch.testing.assert_close(stored, initial, rtol=0, atol=0)
        else:
            torch.testing.assert_close(stored, baseline, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_decode_eager_state_writes_work_every_token() -> None:
    device = torch.device("cuda")
    pool = _make_pool(device)
    torch.manual_seed(105)
    work = torch.tensor([2], dtype=torch.int32, device=device)
    log = torch.tensor([0], dtype=torch.int32, device=device)
    initial = torch.randn(1, 2, 32, 32, device=device, dtype=pool.dtype) * 0.1
    pool.write_states(work, 0, initial)
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)
    for step in range(3):
        query = torch.randn(1, 1, 2, 32, device=device, dtype=pool.dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        gate = torch.randn(1, 1, 2, device=device, dtype=pool.dtype)
        beta = torch.randn_like(gate)
        expected, states = fused_recurrent_gated_delta_rule(
            query, key, value, gate, beta, A_log, dt_bias, initial
        )[:2]
        actual = fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias,
            pool, work, log,
            torch.tensor([20 + step], dtype=torch.int32, device=device),
            0, torch.tensor([step], dtype=torch.int32, device=device),
            enable_eager_state=True,
        )
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            pool.read_states(work, 0), states[:, 0], rtol=3e-2, atol=3e-2
        )
        initial = states[:, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_gemm_replay_matches_eager_window(granularity: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    torch.manual_seed(121)
    query = torch.randn(batch_size, granularity, 2, 128, device=device, dtype=dtype) * 0.2
    key = torch.randn_like(query) * 0.2
    value = torch.randn_like(query) * 0.2
    gate = torch.randn(batch_size, granularity, 2, device=device, dtype=dtype) * 0.1
    beta = torch.randn_like(gate) * 0.1
    A_log = torch.full((2,), -3.0, device=device)
    dt_bias = torch.zeros_like(A_log)
    initial = torch.randn(batch_size, 2, 128, 128, device=device, dtype=dtype) * 0.02

    def run(mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        pool = MHALogStatePool(
            num_frames=128, page_size=64, granularity=granularity,
            num_attention_layers=1, num_kv_heads=1, head_dim=128,
            num_linear_layers=1, num_key_heads=2, num_value_heads=2,
            head_k_dim=128, head_v_dim=128, conv_dim=768, state_len=3,
            device=device, dtype=dtype,
        )
        assert pool.reserve(logs=batch_size, works=batch_size) is not None
        work = torch.arange(batch_size, 2 * batch_size, device=device, dtype=torch.int32)
        logs = torch.arange(batch_size, device=device, dtype=torch.int32)
        pool.write_states(work, 0, initial)
        outputs = []
        for step in range(granularity):
            output = fused_recurrent_gated_delta_rule_to_logstate(
                query[:, step:step + 1].contiguous(),
                key[:, step:step + 1].contiguous(),
                value[:, step:step + 1].contiguous(),
                gate[:, step:step + 1].contiguous(),
                beta[:, step:step + 1].contiguous(),
                A_log, dt_bias, pool, work, logs,
                torch.full((batch_size,), step, device=device, dtype=torch.int32),
                0, torch.full((batch_size,), step % 16, device=device, dtype=torch.int32),
                enable_eager_state=mode == "eager",
            )
            outputs.append(output)
        torch.cuda.synchronize()
        return torch.cat(outputs, dim=1), pool.read_states(work, 0).clone()

    eager_output, eager_state = run("eager")
    gemm_output, gemm_state = run("gemm")
    torch.testing.assert_close(gemm_output, eager_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(gemm_state, eager_state, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_count_and_offset_are_mutable_cuda_graph_inputs() -> None:
    device = torch.device("cuda")
    pool = _make_pool(device)
    torch.manual_seed(104)
    work = torch.tensor([2], dtype=torch.int32, device=device)
    log = torch.tensor([0], dtype=torch.int32, device=device)
    initial = torch.randn(1, 2, 32, 32, device=device, dtype=pool.dtype) * 0.1
    pool.write_states(work, 0, initial)
    query = torch.randn(1, 1, 2, 32, device=device, dtype=pool.dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.randn(1, 1, 2, device=device, dtype=pool.dtype)
    beta = torch.randn_like(gate)
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)
    offsets = torch.zeros(1, dtype=torch.int32, device=device)
    counts = torch.zeros_like(offsets)
    first_output, first_states = fused_recurrent_gated_delta_rule(
        query, key, value, gate, beta, A_log, dt_bias, initial
    )[:2]
    first_query = query.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fused_recurrent_gated_delta_rule_to_logstate(
            query, key, value, gate, beta, A_log, dt_bias,
            pool, work, log, offsets, 0, counts,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, first_output, rtol=3e-2, atol=3e-2)
    offsets.fill_(1)
    counts.fill_(1)
    query.copy_(torch.randn_like(query))
    expected, *_ = fused_recurrent_gated_delta_rule(
        query, key, value, gate, beta, A_log, dt_bias, first_states[:, 0]
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(pool.read_states(work, 0), initial, rtol=0, atol=0)

    other_work = torch.tensor([3], dtype=torch.int32, device=device)
    other_log = torch.tensor([1], dtype=torch.int32, device=device)
    other_initial = initial * .7
    pool.write_states(other_work, 0, other_initial)
    fused_recurrent_gated_delta_rule_to_logstate(
        first_query, key, value, gate, beta, A_log, dt_bias,
        pool, other_work, other_log, torch.zeros_like(offsets), 0, torch.zeros_like(counts),
    )
    _, other_first_states, *_ = fused_recurrent_gated_delta_rule(
        first_query, key, value, gate, beta, A_log, dt_bias, other_initial,
    )
    expected, *_ = fused_recurrent_gated_delta_rule(
        query, key, value, gate, beta, A_log, dt_bias, other_first_states[:, 0],
    )
    for table, indices in ((pool.payload_frame_table, [0, 1]),
                           (pool.payload_frame_table, [2, 3]),
                           (pool.decay_frame_table, [0, 1])):
        table[indices] = table[list(reversed(indices))].clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(pool.read_states(work, 0), other_initial, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_prefill_then_decode_produces_lossless_update_log() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    ctx = _make_context(device)
    pool = ctx.linear_state
    backend = GatedDeltaLogStateBackend()
    config = GatedDeltaConfig(
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        layer_idx=0,
    )
    torch.manual_seed(3)
    conv_weight = (
        torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    )
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(req: Request, phase: str) -> torch.Tensor:
        batch = Batch(reqs=[req], phase=phase)  # type: ignore
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            device=device,
        )
        batch.out_loc = batch.positions
        if phase == "decode":
            batch.logstate_replay_counts = [req.cached_len - 20]
        length = req.extend_len
        mixed_qkv = torch.randn(length, 192, dtype=dtype, device=device)
        gate = torch.randn(length, 2, dtype=dtype, device=device)
        beta = torch.randn_like(gate)
        backend.prepare_metadata(batch)
        return backend.forward(
            mixed_qkv,
            gate,
            beta,
            conv_weight,
            A_log,
            dt_bias,
            config,
            batch,
        )

    set_global_ctx(ctx)
    try:
        assert run(_make_request(0, 20), "prefill").shape == (20, 2, 32)
        for token in range(20, 32):
            assert run(_make_request(token, token + 1), "decode").shape == (1, 2, 32)
    finally:
        clear_global_ctx()

    state = torch.zeros(2, 32, 32, device=device)
    for token in range(32):
        for head in range(2):
            log = head_view(pool, 0, 0, head, state=False)
            key = log[token * 32:(token + 1) * 32].float()
            value_begin = 64 * 32 + token * 32
            value = log[value_begin:value_begin + 32].float()
            state[head] = state[head] * decay_view(pool, 0)[0, token, head].exp() + key[:, None] * value[None, :]
    torch.testing.assert_close(
        pool.read_states(torch.tensor([2], dtype=torch.int32, device=device), 0)[0].float(),
        state,
        rtol=2e-2,
        atol=8e-2,
    )
    torch.testing.assert_close(
        pool.read_conv(
            torch.tensor([2], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            0,
        ),
        pool.read_conv(
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            0,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_multpage_decode_log_restores_work_state() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    pool = MHALogStatePool(
        num_frames=96,
        page_size=64,
        granularity=16,
        num_attention_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_linear_layers=1,
        num_key_heads=2,
        num_value_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_dim=192,
        state_len=3,
        device=device,
        dtype=dtype,
    )
    allocated = pool.reserve(logs=5, works=2)
    assert allocated is not None
    logs, works = allocated[1:]
    ctx = Context(page_size=64)
    ctx.linear_state = pool
    ctx.page_table = torch.arange(320, dtype=torch.int32, device=device).view(1, 320)
    ctx.state_table = torch.tensor(
        [slot for slot in logs for _ in range(64)],
        dtype=torch.int32, device=device,
    ).view(1, 320)
    ctx.work_state_table = torch.tensor(
        [[works[0]]], dtype=torch.int32, device=device
    )
    backend = GatedDeltaLogStateBackend()
    pool.zero_state(works[0])
    pool.zero_work_conv(works[0])
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    torch.manual_seed(73)
    mixed_qkv = torch.randn(291, 192, dtype=dtype, device=device)
    gate = torch.randn(291, 2, dtype=dtype, device=device)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(begin: int, end: int, phase: str) -> None:
        req = _make_request(begin, end)
        batch = Batch(reqs=[req], phase=phase)  # type: ignore
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(begin, end, dtype=torch.int32, device=device)
        batch.out_loc = batch.positions
        if phase == "decode":
            batch.logstate_replay_counts = [
                begin - max(35, (begin // 16) * 16)
            ]
        backend.prepare_metadata(batch)
        backend.forward(
            mixed_qkv[begin:end], gate[begin:end], beta[begin:end],
            conv_weight, A_log, dt_bias, config, batch,
        )

    set_global_ctx(ctx)
    try:
        run(0, 35, "prefill")
        for token in range(35, 272):
            run(token, token + 1, "decode")
        expected = pool.read_states(
            torch.tensor([works[0]], dtype=torch.int32, device=device), 0
        ).clone()
        restored = torch.tensor([works[1]], dtype=torch.int32, device=device)
        restore_log_state_chains(
            pool,
            [(None, [(slot, 0, 64) for slot in logs[:4]]
              + [(logs[4], 0, 16)], works[1])],
        )
        torch.testing.assert_close(
            pool.read_states(restored, 0), expected, rtol=2e-2, atol=4e-2,
        )
        torch.testing.assert_close(
            pool.read_conv(
                restored.new_tensor([logs[4]]), restored.new_tensor([0]), 0
            ),
            pool.read_conv(
                restored.new_tensor([works[0]]), restored.new_tensor([0]), 0
            ),
        )
    finally:
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logstate_model_sized_prefill_log_restores_first_granule() -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(16, 16, 128, 128, 6144, 3, 0)
    pool = MHALogStatePool(
        num_frames=128,
        page_size=64,
        granularity=16,
        num_attention_layers=4,
        num_kv_heads=2,
        head_dim=128,
        num_linear_layers=1,
        num_key_heads=16,
        num_value_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_dim=6144,
        state_len=3,
        device=device,
        dtype=dtype,
    )
    allocated = pool.reserve(logs=2, works=3)
    assert allocated is not None
    log_slot, reference_log_slot = allocated[1]
    work_slot, restored_slot, reference_work_slot = allocated[2]
    ctx = Context(page_size=64)
    ctx.linear_state = pool
    ctx.page_table = torch.arange(64, dtype=torch.int32, device=device).view(1, 64)
    ctx.state_table = torch.full(
        (1, 64), log_slot, dtype=torch.int32, device=device
    )
    ctx.work_state_table = torch.tensor(
        [[work_slot]], dtype=torch.int32, device=device
    )
    pool.zero_state(work_slot)
    pool.zero_work_conv(work_slot)
    torch.manual_seed(127)
    mixed_qkv = torch.randn(35, 6144, device=device, dtype=dtype)
    gate = torch.randn(35, 16, device=device, dtype=dtype)
    beta = torch.randn_like(gate)
    conv_weight = torch.randn(6144, 1, 4, device=device, dtype=dtype) * 0.02
    A_log = torch.zeros(16, device=device)
    dt_bias = torch.zeros(16, device=device)
    req = _make_request(0, 35)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.arange(35, dtype=torch.int32, device=device)
    batch.out_loc = batch.positions
    backend = GatedDeltaLogStateBackend()
    set_global_ctx(ctx)
    try:
        backend.prepare_metadata(batch)
        full_output = backend.forward(
            mixed_qkv, gate, beta, conv_weight, A_log, dt_bias, config, batch
        )
        full_state = pool.read_states(
            torch.tensor([work_slot], dtype=torch.int32, device=device), 0
        ).clone()
        restore_log_state_chains(
            pool, [(None, [(log_slot, 0, 16)], restored_slot)]
        )
        actual = pool.read_states(
            torch.tensor([restored_slot], dtype=torch.int32, device=device), 0
        )
        ctx.state_table.fill_(reference_log_slot)
        ctx.work_state_table.fill_(reference_work_slot)
        pool.zero_state(reference_work_slot)
        pool.zero_work_conv(reference_work_slot)
        reference_req = _make_request(0, 16)
        reference_batch = Batch(reqs=[reference_req], phase="prefill")
        reference_batch.padded_reqs = reference_batch.reqs
        reference_batch.positions = torch.arange(
            16, dtype=torch.int32, device=device
        )
        reference_batch.out_loc = reference_batch.positions
        backend.prepare_metadata(reference_batch)
        backend.forward(
            mixed_qkv[:16], gate[:16], beta[:16], conv_weight,
            A_log, dt_bias, config, reference_batch,
        )
        expected = pool.read_states(
            torch.tensor([reference_work_slot], dtype=torch.int32, device=device), 0
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)
        torch.testing.assert_close(
            pool.read_conv(
                torch.tensor([log_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
            pool.read_conv(
                torch.tensor([reference_work_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
        )
        pool.write_conv(
            torch.tensor([restored_slot], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device), 0,
            pool.read_conv(
                torch.tensor([log_slot], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device), 0,
            ),
        )
        ctx.state_table[:, :16].fill_(-1)
        ctx.work_state_table.fill_(restored_slot)
        suffix_req = _make_request(16, 35)
        suffix_batch = Batch(reqs=[suffix_req], phase="prefill")
        suffix_batch.padded_reqs = suffix_batch.reqs
        suffix_batch.positions = torch.arange(
            16, 35, dtype=torch.int32, device=device
        )
        suffix_batch.out_loc = suffix_batch.positions
        backend.prepare_metadata(suffix_batch)
        suffix_output = backend.forward(
            mixed_qkv[16:], gate[16:], beta[16:], conv_weight,
            A_log, dt_bias, config, suffix_batch,
        )
        torch.testing.assert_close(
            suffix_output, full_output[16:], rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            pool.read_states(
                torch.tensor([restored_slot], dtype=torch.int32, device=device), 0
            ),
            full_state, rtol=2e-2, atol=4e-2,
        )
    finally:
        clear_global_ctx()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("materialized", [False, True])
def test_logstate_prefix_restore_resumes_prefill_at_page_boundary(
    materialized: bool,
) -> None:
    clear_global_ctx()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = GatedDeltaConfig(2, 2, 32, 32, 192, 3, 0)
    torch.manual_seed(9)
    mixed_qkv = torch.randn(81, 192, dtype=dtype, device=device)
    gate = torch.randn(81, 2, dtype=dtype, device=device)
    beta = torch.randn_like(gate)
    conv_weight = (
        torch.randn(192, 1, 4, dtype=dtype, device=device) * 0.02
    )
    A_log = torch.zeros(2, device=device)
    dt_bias = torch.zeros(2, device=device)

    def run(
        ctx  : Context,
        begin: int,
        end  : int,
    ) -> torch.Tensor:
        req = _make_request(begin, end)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.positions = torch.arange(
            begin, end, dtype=torch.int32, device=device
        )
        batch.out_loc = batch.positions
        backend = ctx.linear_attn_backend
        backend.prepare_metadata(batch)
        return backend.forward(
            mixed_qkv[begin:end],
            gate[begin:end],
            beta[begin:end],
            conv_weight,
            A_log,
            dt_bias,
            config,
            batch,
        )

    full_ctx = _make_context(device)
    full_ctx.linear_attn_backend = GatedDeltaLogStateBackend()
    set_global_ctx(full_ctx)
    try:
        full = run(full_ctx, 0, 81).clone()
    finally:
        clear_global_ctx()

    resumed_ctx = _make_context(device)
    resumed_ctx.linear_attn_backend = GatedDeltaLogStateBackend()
    set_global_ctx(resumed_ctx)
    try:
        first = run(resumed_ctx, 0, 64)
        pool = resumed_ctx.linear_state
        expected_state = pool.read_states(torch.tensor([2], dtype=torch.int32, device=device), 0).clone()
        checkpoint = pool.reserve(works=1)[2][0] if materialized else 2
        construct_log_state_chains(
            pool,
            [(None, [(0, 0, 64)], [(
                checkpoint,
            )])],
        )
        if materialized:
            copy_states(pool, [checkpoint], [2])
        source = torch.tensor([0], dtype=torch.int32, device=device)
        destination = torch.tensor([2], dtype=torch.int32, device=device)
        pool.write_conv(
            destination, torch.tensor([0], dtype=torch.int32, device=device), 0,
            pool.read_conv(source, torch.tensor([3], dtype=torch.int32, device=device), 0),
        )
        torch.testing.assert_close(
            pool.read_states(destination, 0), expected_state, rtol=2e-2, atol=4e-2
        )
        resumed_ctx.state_table[0, :64].fill_(-1)
        second = run(resumed_ctx, 64, 81)
        resumed = torch.cat([first, second])
    finally:
        clear_global_ctx()

    torch.testing.assert_close(
        full, resumed, rtol=2e-2, atol=2e-2
    )
