"""Copy framed convolution snapshots without staging them in a tensor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_framed_conv_snapshots(
    storage, frame_table,
    source_slots, source_granules, output_slots, output_granules,
    NUM_GRANULES: tl.constexpr, SNAPSHOT_ELEMS: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_slot = tl.load(source_slots + task)
    source_granule = tl.load(source_granules + task)
    output_slot = tl.load(output_slots + task)
    output_granule = tl.load(output_granules + task)
    source_frame = tl.load(
        frame_table + source_slot * NUM_GRANULES + source_granule
    )
    output_frame = tl.load(
        frame_table + output_slot * NUM_GRANULES + output_granule
    )
    source_base = source_frame.to(tl.int64) * FRAME_ELEMS
    output_base = output_frame.to(tl.int64) * FRAME_ELEMS
    values = tl.load(storage + source_base + offsets,
                     mask=offsets < SNAPSHOT_ELEMS, other=0)
    tl.store(storage + output_base + offsets, values,
             mask=offsets < SNAPSHOT_ELEMS)


def copy_framed_conv_snapshots(
    pool,
    source_slots: torch.Tensor,
    source_granules: torch.Tensor,
    output_slots: torch.Tensor,
    output_granules: torch.Tensor,
) -> None:
    """Copy every linear layer's conv state for each independent restore task."""
    count = source_slots.numel()
    assert source_granules.numel() == output_slots.numel() == output_granules.numel() == count
    if count == 0:
        return
    assert pool.allocator.conv_frames == 1
    assert all(value.device == pool.device for value in (
        source_slots, source_granules, output_slots, output_granules,
    ))
    snapshot_elems = (
        pool.layout.num_linear_layers * pool.layout.conv_dim * pool.layout.state_len
    )
    frame_elems = pool.frame_bytes // pool.dtype.itemsize
    assert snapshot_elems <= frame_elems
    _copy_framed_conv_snapshots[(count, triton.cdiv(snapshot_elems, 1024))](
        pool.storage.view(pool.dtype), pool.conv_frame_table,
        source_slots, source_granules, output_slots, output_granules,
        pool.layout.num_granules, snapshot_elems, frame_elems, 1024,
    )
