from contextlib import nullcontext
from types import SimpleNamespace

import torch

from logssms.core import Batch, Request, SamplingParams
from logssms.engine import ForwardInput, ForwardOutput
from logssms.scheduler.decode import DecodeManager


class FakeCacheManager:
    def lazy_free_region(self):
        return nullcontext()


def test_prefill_between_decode_batches_closes_overlap_window():
    cache_manager = FakeCacheManager()
    table_manager = SimpleNamespace(
        token_pool=torch.zeros((1, 8), dtype=torch.int32)
    )
    config = SimpleNamespace(page_size=1, decode_batch_budget=1)
    manager = DecodeManager(
        config=config,
        device=torch.device("cpu"),
        cache_manager=cache_manager,
        table_manager=table_manager,
        eos_token_id=0,
    )
    req = Request(
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        table_idx=0,
        cached_len=1,
        output_len=1,
        uid=7,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=1),
        cache_handle=object(),
        max_device_len=3,
    )
    manager.running_reqs[req.uid] = req

    decode_batch = manager.schedule_next_batch()
    assert decode_batch is not None
    manager.advance_for_overlap(decode_batch)
    assert req.device_len == req.max_device_len

    manager.on_prefill_scheduled()
    assert manager.inflight_uids[1] == set()

    forward_input = ForwardInput(
        batch=decode_batch,
        sample_args=None,  # type: ignore[arg-type]
        input_tuple=(torch.empty(0), torch.empty(0)),
        write_tuple=(torch.empty(0), torch.empty(0)),
    )
    output = ForwardOutput(
        next_tokens_gpu=torch.tensor([3], dtype=torch.int32),
        next_tokens_cpu=torch.tensor([3], dtype=torch.int32),
        copy_done_event=None,  # type: ignore[arg-type]
    )
    replies, finished = manager.process(None, forward_input, output)  # type: ignore[arg-type]

    assert replies[0].finished
    assert finished == {req}
    assert req.uid not in manager.running_reqs
