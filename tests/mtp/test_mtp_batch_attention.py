import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_indexed_native_kv_pool_matches_dense_batch_attention() -> None:
    from logssms.speculator.drafters.attention import DraftAttentionBackend
    from logssms.speculator.drafters.pool import DraftKVPool

    device = torch.device("cuda")
    num_qo_heads = 4
    num_kv_heads = 2
    head_dim = 64
    lengths = [2, 4, 3]
    offsets = [0, 2, 6]
    total = sum(lengths)
    generator = torch.Generator(device=device).manual_seed(31)
    query = torch.randn(
        total,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        total,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        total,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    pool = DraftKVPool(
        max_running_req=3,
        window_size=max(lengths),
        max_batch_size=3,
        num_spec_tokens=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=key.dtype,
        device=device,
    )
    pool.k[:total].copy_(key)
    pool.v[:total].copy_(value)
    indices = torch.zeros(3, max(lengths), dtype=torch.int32, device=device)
    valid = torch.zeros(3, max(lengths), dtype=torch.bool, device=device)
    last_rows = []
    for i, (offset, length) in enumerate(zip(offsets, lengths)):
        indices[i, :length] = torch.arange(
            offset, offset + length, dtype=torch.int32, device=device
        )
        valid[i, :length] = True
        last_rows.append(offset + length - 1)

    last_rows = torch.tensor(last_rows, device=device)
    backend = DraftAttentionBackend(
        "eager",
        num_qo_heads,
        num_kv_heads,
        head_dim,
        key.dtype,
        device,
    )
    pooled = backend.forward(query[last_rows], pool, indices, valid, lengths)
    expected = []
    num_repeats = num_qo_heads // num_kv_heads
    for offset, length, last_row in zip(offsets, lengths, last_rows):
        query_row = query[last_row]
        key_row = key[offset : offset + length].repeat_interleave(
            num_repeats, dim=1
        ).transpose(0, 1)
        value_row = value[offset : offset + length].repeat_interleave(
            num_repeats, dim=1
        ).transpose(0, 1)
        score = torch.matmul(query_row.unsqueeze(1), key_row.transpose(-1, -2))
        score = F.softmax(
            score * head_dim**-0.5, dim=-1, dtype=torch.float32
        )
        output = torch.matmul(score.to(value_row.dtype), value_row).squeeze(1)
        expected.append(output)

    torch.testing.assert_close(pooled, torch.stack(expected), atol=2e-3, rtol=2e-3)
