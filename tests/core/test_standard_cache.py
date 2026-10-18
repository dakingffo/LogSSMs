from types import SimpleNamespace

import pytest
import torch

from logssms.cache.prefix_cache.naive import NaivePrefixCache
from logssms.cache.standard import StandardCacheManager


@pytest.mark.parametrize("with_state", [False, True])
def test_naive_cache_retains_active_pages_at_page_boundary(with_state):
    cache = NaivePrefixCache(torch.device("cpu"), with_state=with_state)
    pages = torch.arange(256, dtype=torch.int32).unsqueeze(0)
    states = torch.arange(2, dtype=torch.int32).repeat_interleave(128).unsqueeze(0)
    manager = object.__new__(StandardCacheManager)
    manager.page_size = 128
    manager.resources = SimpleNamespace(
        page_table=pages.clone(), state_table=states.clone() if with_state else None,
    )
    manager.prefix_cache = cache
    manager.free_pages = torch.empty(0, dtype=torch.int32)
    manager.free_states = torch.empty(0, dtype=torch.int32)
    req = SimpleNamespace(
        input_ids=torch.arange(256), cached_len=128, table_idx=0,
        cache_handle=cache.match_prefix(torch.arange(128)).cuda_handle,
    )
    manager.cache_req(req, finished=False)
    assert req.cache_handle.cached_len == 0
    assert torch.equal(manager.page_table, pages)
    assert len(manager.free_pages) == 0
    if with_state:
        assert torch.equal(manager.state_table, states)
        assert len(manager.free_states) == 0
