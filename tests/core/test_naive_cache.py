from types import SimpleNamespace

import pytest
import torch

from logssms.cache.naive.prefix_cache import NaivePrefixCache
from logssms.cache.naive.manager import NaiveCacheManager
from logssms.cache.base import CacheResources


@pytest.mark.parametrize("with_state", [False, True])
def test_naive_cache_retains_active_pages_at_page_boundary(with_state):
    cache = NaivePrefixCache(torch.device("cpu"), with_state=with_state)
    pages = torch.arange(256, dtype=torch.int32).unsqueeze(0)
    states = torch.arange(2, dtype=torch.int32).repeat_interleave(128).unsqueeze(0)
    manager = object.__new__(NaiveCacheManager)
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


@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("cached_len", [1, 64, 65, 128])
def test_naive_finished_request_releases_all_pages(with_state, cached_len):
    resources = CacheResources(
        kv_pool=None,
        page_table=torch.full((2, 128), -1, dtype=torch.int32),
        linear_state=object() if with_state else None,
        state_table=torch.full((2, 128), -1, dtype=torch.int32) if with_state else None,
        work_state_table=None,
    )
    manager = NaiveCacheManager(2, 64, resources, num_states=2 if with_state else 0)
    req = SimpleNamespace(
        input_ids=torch.arange(cached_len), cached_len=0,
        device_len=cached_len, table_idx=0, profile_tag=None,
    )
    manager.allocate([req])
    req.cached_len = cached_len
    pages_left = len(manager.free_pages)
    manager.cache_req(req, finished=False)
    assert len(manager.free_pages) == pages_left
    manager.cache_req(req, finished=True)
    assert sorted(manager.free_pages.tolist()) == [0, 64]
    if with_state:
        assert sorted(manager.free_states.tolist()) == [0, 1]
        assert torch.all(manager.state_table == -1)
    assert manager.prefix_cache.match_prefix(req.input_ids).cuda_handle.cached_len == 0
    manager.check_integrity()


def test_naive_exhaustion_does_not_consume_free_pages():
    manager = NaiveCacheManager(
        2, 64, CacheResources(
            kv_pool=None, page_table=torch.empty((2, 128), dtype=torch.int32),
            linear_state=None, state_table=None, work_state_table=None,
        ),
    )
    with pytest.raises(RuntimeError, match="Insufficient cache capacity"):
        manager._allocate(needed_pages=3)
    assert manager.free_pages.tolist() == [0, 64]
