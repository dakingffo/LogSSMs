from types import SimpleNamespace

import torch

from logssms.core import Request, SamplingParams
from logssms.scheduler.verify import VerifyManager


def test_inflight_verify_reserves_from_committed_frontier():
    req = Request(
        input_ids=torch.arange(100, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=512,
        uid=1,
        cache_handle=None,
        sampling_params=SamplingParams(max_tokens=512),
        max_device_len=612,
    )
    req.cached_len = 99
    req.device_len = 103  # K=3 speculative positions, not settled yet

    manager = object.__new__(VerifyManager)
    manager.config = SimpleNamespace(page_size=256)
    manager.running_reqs = {req.uid: req}
    manager.inflight_uids = [set(), {req.uid}]

    assert req.remain_len == 509
    assert manager._reserve_remain_len(req) == 512
    assert manager.need_tokens == 512


def test_settled_verify_uses_device_frontier():
    req = Request(
        input_ids=torch.arange(100, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=512,
        uid=1,
        cache_handle=None,
        sampling_params=SamplingParams(max_tokens=512),
        max_device_len=612,
    )
    req.cached_len = 101
    req.device_len = 102

    manager = object.__new__(VerifyManager)
    manager.config = SimpleNamespace(page_size=256)
    manager.running_reqs = {req.uid: req}
    manager.inflight_uids = [set(), set()]

    assert manager._reserve_remain_len(req) == req.remain_len
