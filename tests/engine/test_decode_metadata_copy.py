import pytest
import torch
from types import SimpleNamespace

from logssms.engine.decode_graph import DecodeGraphCaptureBuffer
from logssms.kernel.metadata import copy_metadata
from logssms.layers.linear_attention_backend.logstate import (
    LogStateLinearAttentionBackend, LogStateLinearAttentionCaptureData,
    LogStateLinearAttentionMetadata,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("size", [1, 8, 32, 160])
def test_metadata_copy_preserves_padding_and_converts_dtype(device, size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    sources = tuple(torch.arange(size, dtype=torch.int64, device=device) + i for i in range(6))
    outputs = tuple(torch.full((size + 8,), -1, dtype=torch.int32, device=device) for _ in sources)
    copy_metadata(sources, tuple(value[:size] for value in outputs))
    for source, output in zip(sources, outputs, strict=True):
        torch.testing.assert_close(output[:size], source.int())
        assert (output[size:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_metadata_copies_update_graph_inputs_without_cross_batch_pollution():
    capture = DecodeGraphCaptureBuffer(4, 8, torch.device("cuda"))
    backend = LogStateLinearAttentionBackend()
    backend.capture = LogStateLinearAttentionCaptureData.make(4, torch.device("cuda"))
    linear = backend.capture
    graph = torch.cuda.CUDAGraph()
    result = torch.empty(4, device="cuda", dtype=torch.int32)
    with torch.cuda.graph(graph):
        result.copy_(capture.input_ids + capture.positions + capture.out_loc
                     + linear.work_slots + linear.log_slots + linear.token_offsets
                     + linear.conv_read_slots + linear.conv_read_granules + linear.replay_counts)
    prepare = torch.cuda.Stream()
    forward = torch.cuda.Stream()
    prepare.wait_stream(torch.cuda.current_stream())
    forward.wait_stream(torch.cuda.current_stream())
    snapshots, packets = [], []
    for step in range(32):
        with torch.cuda.stream(prepare):
            sources = tuple(torch.full((4,), step * 10 + i, device="cuda", dtype=torch.int32)
                            for i in range(9))
            ready = torch.cuda.Event()
            ready.record()
            packets.append(sources)
        with torch.cuda.stream(forward):
            forward.wait_event(ready)
            capture.copy_from(SimpleNamespace(input_ids=sources[0], positions=sources[1],
                                              out_loc=sources[2], padded_size=4))
            metadata = LogStateLinearAttentionMetadata("decode", *sources[3:])
            backend._copy_to_capture(metadata, 4)
            graph.replay()
            snapshots.append(result.clone())
    forward.synchronize()
    for step, actual in enumerate(snapshots):
        torch.testing.assert_close(actual, torch.full_like(actual, step * 90 + 36))
