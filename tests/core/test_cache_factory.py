from types import SimpleNamespace

import torch

import picosgl.core as core
from picosgl.cache import (
    CacheResources,
    StandardCachePlan,
    SUPPORTED_CACHE_FACTORIES,
    get_cache_factory,
)
from picosgl.cache.prefix_cache.naive import NaivePrefixCache
from picosgl.cache.prefix_cache.radix import HybridRadixTreeNode, RadixPrefixCache


def _make_resources() -> CacheResources:
    return CacheResources(
        kv_pool=None,  # type: ignore[arg-type]
        page_table=torch.zeros((2, 128), dtype=torch.int32),
        linear_state=None,
        state_table=torch.full((2, 128), -1, dtype=torch.int32),
        work_state_table=None,
    )


def _make_plan() -> StandardCachePlan:
    return StandardCachePlan(
        num_pages=2,
        num_tokens=128,
        max_seq_len=128,
        aligned_max_seq_len=128,
        num_states=2,
        num_draft_states=0,
        draft_slots_per_request=0,
        cache_per_page=1,
        cache_per_state=1,
    )


def test_standard_factories_are_public_cache_strategies():
    assert SUPPORTED_CACHE_FACTORIES.supported_names() == ["naive", "radix"]


def test_hybrid_backend_selection_is_internal_to_standard_factory():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=64)
    try:
        config = SimpleNamespace(
            page_size=64,
            model_config=SimpleNamespace(is_hybrid=True),
        )
        resources = _make_resources()
        plan = _make_plan()

        naive = get_cache_factory("naive").make_manager(config, plan, resources)
        assert isinstance(naive.prefix_cache, NaivePrefixCache)
        assert naive.prefix_cache.with_state

        radix = get_cache_factory("radix").make_manager(config, plan, resources)
        assert isinstance(radix.prefix_cache, RadixPrefixCache)
        assert radix.prefix_cache.NodeType is HybridRadixTreeNode
    finally:
        core._GLOBAL_CTX = old_ctx

