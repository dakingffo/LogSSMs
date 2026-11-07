import torch
import triton
import triton.language as tl
from .layout import conv_pointer


@triton.jit(
    do_not_specialize_on_alignment=['frame_table', 'source_slots', 'source_granules', 'output_slots', 'output_granules'],
)
def _copy_conv_snapshots(
    storage, frame_table,
    source_slots, source_granules, output_slots, output_granules,
    NUM_GRANULES: tl.constexpr, SNAPSHOT_ELEMS: tl.constexpr,
    FRAME_ELEMS: tl.constexpr, FRAMES: tl.constexpr, BLOCK: tl.constexpr,
):
    task = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_slot = tl.load(source_slots + task)
    source_granule = tl.load(source_granules + task)
    output_slot = tl.load(output_slots + task)
    output_granule = tl.load(output_granules + task)
    source = conv_pointer(storage, frame_table, source_slot, source_granule, offsets,
                          NUM_GRANULES, FRAMES, FRAME_ELEMS, offsets < SNAPSHOT_ELEMS)
    output = conv_pointer(storage, frame_table, output_slot, output_granule, offsets,
                          NUM_GRANULES, FRAMES, FRAME_ELEMS, offsets < SNAPSHOT_ELEMS)
    values = tl.load(source, offsets < SNAPSHOT_ELEMS, other=0)
    tl.store(output, values, offsets < SNAPSHOT_ELEMS)


def copy_conv_snapshots(
    pool,
    source_slots: torch.Tensor,
    source_granules: torch.Tensor,
    output_slots: torch.Tensor,
    output_granules: torch.Tensor,
) -> None:
    count = source_slots.numel()
    assert source_granules.numel() == output_slots.numel() == output_granules.numel() == count
    if count == 0:
        return
    assert all(value.device == pool.device for value in (
        source_slots, source_granules, output_slots, output_granules,
    ))
    snapshot_elems = (
        pool.layout.num_linear_layers * pool.layout.conv_dim * pool.layout.state_len
    )
    frame_elems = pool.frame_bytes // pool.dtype.itemsize
    _copy_conv_snapshots[(count, triton.cdiv(snapshot_elems, 1024))](
        pool.storage.view(pool.dtype), pool.conv_frame_table,
        source_slots, source_granules, output_slots, output_granules,
        pool.layout.num_granules, snapshot_elems, frame_elems, pool.allocator.conv_frames, 1024,
    )
