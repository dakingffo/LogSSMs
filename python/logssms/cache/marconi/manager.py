from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import multiprocessing as mp

import torch

from logssms.utils import align_ceil, init_logger
from logssms.cache.base import BaseCacheManager, CacheResources
from logssms.cache.marconi.layout import HybridBlockLayout
from logssms.cache.marconi.block_pool import HybridBlockPool
from logssms.cache.marconi.metadata import MarconiCacheMetadata
from logssms.cache.factory import BaseCacheFactory, CachePlan, SUPPORTED_CACHE_FACTORIES
from logssms.cache.marconi.prefix_cache import (
    MarconiCost,
    MarconiHandle,
    MarconiPrefixCache,
    common_prefix,
    tune_marconi,
)


logger = init_logger(__name__)


def marconi_layout(config):
    model = config.model_config
    if not model.is_hybrid:
        raise ValueError("Marconi requires a hybrid model")
    spec = model.make_linear_cache_spec(config.tp_info.size)
    return HybridBlockLayout.make(
        model.num_attention_layers, spec,
        max(1, model.num_kv_heads // config.tp_info.size), model.head_dim,
        config.cache_config.page_size,
    )


def marconi_cost(config, layout):
    model, tp = config.model_config, config.tp_info.size
    spec = model.make_linear_cache_spec(tp)
    d, q, kv = model.hidden_size, model.num_qo_heads * model.head_dim, model.num_kv_heads * model.head_dim
    intermediate = model.intermediate_size
    if model.is_moe:
        intermediate = model.num_experts_per_tok * model.moe_intermediate_size + model.shared_expert_intermediate_size
    mlp = 6 * d * intermediate * model.num_layers
    attn_projection = 2 * d * (2 * q + 2 * kv) * model.num_attention_layers
    conv_dim, history = spec.conv_shape
    heads, rows, columns = spec.state_shape
    # Per-rank recurrent storage describes both GDN and Mamba2 dimensions.
    projection = 2 * d * (conv_dim + heads * columns) * tp
    recurrent = (8 if spec.recurrent_type == "gdn" else 6) * heads * rows * columns * tp
    linear = spec.num_layers * (projection + recurrent + 2 * conv_dim * (history + 1) * tp)
    block_bytes = layout.block_bytes(config.dtype.itemsize)
    return MarconiCost(
        (mlp + attn_projection + linear) / tp,
        2 * q * model.num_attention_layers / tp,
        layout.state_groups * block_bytes,
        block_bytes // layout.page_size,
    )


@dataclass(frozen=True)
class MarconiCachePlan(CachePlan):
    layout: HybridBlockLayout
    num_blocks: int
    metadata_bytes: int


@dataclass
class MarconiRequest:
    handle: MarconiHandle
    prompt: tuple[int, ...]
    work: int
    pages: list[int]
    branch: int = 0
    checkpoints: dict[int, int] = field(default_factory=dict)
    published: set[int] = field(default_factory=set)
    planned: bool = False
    computed_end: int = 0


class MarconiCacheManager(BaseCacheManager):
    def __init__(self, resources, config, plan):
        self.pool = resources.kv_pool
        self.plan = plan
        self.config = config
        self.cost = marconi_cost(config, plan.layout)
        tree = MarconiPrefixCache(self.pool.device, plan.layout.page_size, self.cost, pool=self.pool)
        super().__init__(plan.layout.page_size, resources, prefix_cache=tree)
        self.tree = tree
        self.requests: dict[int, MarconiRequest] = {}
        self.completed = 0
        self.tune_snapshot = None
        self.tune_history = []
        self.tune_remaining = 0
        self.tune_future = None
        self.tune_executor = None
        self.tune_results = None
        self.tree.on_first_eviction = self._start_bootstrap
        self._deferred = None

    def _start_bootstrap(self):
        self.tune_snapshot = self.tree.snapshot()
        self.tune_remaining = max(1, self.completed) * self.config.cache_config.bootstrap_multiplier

    def _record_history(self, prompt, sequence):
        self.completed += 1
        if self.tune_remaining <= 0:
            return
        self.tune_history.append((prompt, sequence))
        self.tune_remaining -= 1
        if self.tune_remaining == 0 and self.config.tp_info.rank == 0:
            self.tune_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            self.tune_future = self.tune_executor.submit(
                tune_marconi, self.tune_snapshot, self.tune_history, self.page_size,
                self.cost, (self.plan.num_blocks - 1 - self.plan.layout.state_groups) * self.plan.layout.block_bytes(self.pool.dtype.itemsize),
            )
            self.tune_history = []
            self.tune_snapshot = None

    def _poll_tuning(self):
        if not self.tree.evicted or self.tune_remaining > 0 or self.tune_results is not None:
            return
        alpha = -1.0
        if self.config.tp_info.rank == 0 and self.tune_future is not None and self.tune_future.done():
            alpha, self.tune_results = self.tune_future.result()
            self.tune_executor.shutdown(wait=False)
        if self.config.tp_info.size > 1:
            packet = torch.tensor([alpha], dtype=torch.float64)
            torch.distributed.broadcast(packet, src=0)
            alpha = packet.item()
        if alpha >= 0:
            self.tree.alpha = alpha
            self.tune_history = []
            self.tune_snapshot = None
            if self.tune_results is None:
                self.tune_results = []
            logger.info(f"Marconi selected eviction alpha={alpha:.1f}")

    @property
    def available_size(self):
        return len(self.pool.allocator.free) + self.tree.reclaimable_blocks()

    def reservation_size(self, token_len, *, handle=None, table_idx=None):
        groups = self.plan.layout.state_groups
        entry = self.requests.get(table_idx)
        if entry is None:
            offset = 0 if handle is None else handle.cached_len % self.page_size
            return (token_len + offset + self.page_size - 1) // self.page_size + 2 * groups
        # One optional branch snapshot plus the remaining KV pages.
        return (token_len + self.page_size - 1) // self.page_size + (
            groups if not entry.planned or (entry.branch and entry.branch not in entry.checkpoints) else 0
        )

    def match_req(self, req):
        match = self.tree.match_prefix(req.input_ids[:-1])
        return match._replace(cuda_handle=replace(match.cuda_handle, input_tokens=tuple(req.input_ids.tolist())))

    def plan_prefill_chunk(self, remain_len, token_budget):
        return remain_len if remain_len <= token_budget else token_budget // 64 * 64

    def _ensure_free(self, count):
        while len(self.pool.allocator.free) < count:
            if not self.tree.evict_one():
                raise RuntimeError(f"Marconi reservation exhausted: need={count}, free={len(self.pool.allocator.free)}")

    def init_with_handle(self, table_idx, handle):
        self._ensure_free(self.plan.layout.state_groups + int(handle.cached_len % self.page_size != 0))
        work = self.pool.allocate_state()
        pages = list(dict.fromkeys(x // self.page_size for x in handle.node.locations))
        self.pool.allocator.retain(pages)
        if handle.cached_len:
            self.pool.copy_state(handle.node.state, work)
            self.tree.touch(handle.node)
        if handle.cached_len % self.page_size:
            old = pages[-1]
            new, = self.pool.allocator.allocate(1)
            for layer in range(self.pool.num_layers):
                self.pool.k_cache(layer)[new].copy_(self.pool.k_cache(layer)[old])
                self.pool.v_cache(layer)[new].copy_(self.pool.v_cache(layer)[old])
            self.pool.allocator.release((old,))
            pages[-1] = new
        entry = MarconiRequest(handle, handle.input_tokens, work, pages, handle.branch_end)
        self.requests[table_idx] = entry
        self.state_table[table_idx].fill_(work)
        self._write_pages(table_idx, pages, 0)

    def _write_pages(self, table_idx, pages, start):
        if not pages:
            return
        end = min((start + len(pages)) * self.page_size, self.page_table.shape[1])
        values = (torch.tensor(pages, device=self.device, dtype=torch.int32)[:, None] * self.page_size
                  + torch.arange(self.page_size, device=self.device, dtype=torch.int32)).flatten()
        self.page_table[table_idx, start * self.page_size:end].copy_(values[:end - start * self.page_size])

    def allocate(self, reqs, *, phase=None):
        if phase == "prefill":
            self._poll_tuning()
            previous = []
            for req in reqs:
                entry = self.requests[req.table_idx]
                if not entry.planned:
                    matched = max([entry.branch, *[common_prefix(entry.prompt, p) for p in previous]])
                    branch = entry.handle.cached_len + (max(0, matched - entry.handle.cached_len) // 64) * 64
                    entry.branch = branch if branch > entry.handle.cached_len else 0
                    entry.planned = True
                previous.append(entry.prompt)
        for req in reqs:
            entry = self.requests[req.table_idx]
            n_pages = (req.device_len + self.page_size - 1) // self.page_size
            needed = max(0, n_pages - len(entry.pages))
            write_branch = req.cached_len < entry.branch <= req.device_len and entry.branch not in entry.checkpoints
            self._ensure_free(needed + int(write_branch) * self.plan.layout.state_groups)
            if needed:
                pages = self.pool.allocator.allocate(needed)
                self._write_pages(req.table_idx, pages, len(entry.pages))
                entry.pages.extend(pages)
            if write_branch:
                entry.checkpoints[entry.branch] = self.pool.allocate_state()
            entry.computed_end = req.device_len

    def prepare_batch(self, batch):
        if not batch.is_prefill:
            return
        batch.cache_metadata = MarconiCacheMetadata()
        writes = []
        for req in batch.reqs:
            entry = self.requests[req.table_idx]
            targets = [(end - req.cached_len, slot) for end, slot in entry.checkpoints.items()
                       if req.cached_len < end < req.device_len]
            targets.append((req.extend_len, entry.work))
            if req.device_len in entry.checkpoints:
                batch.cache_metadata.checkpoint_copies.append((entry.work, entry.checkpoints[req.device_len]))
            writes.append(sorted(targets))
        batch.cache_metadata.checkpoint_writes = writes

    def after_forward(self, batch):
        parts = batch.mixed_parts if batch.is_mixed else (batch,)
        for part in parts:
            if not isinstance(part.cache_metadata, MarconiCacheMetadata):
                continue
            for source, target in part.cache_metadata.checkpoint_copies:
                self.pool.copy_state(source, target)

    def shutdown(self):
        if self.tune_executor is not None:
            self.tune_executor.shutdown(wait=True, cancel_futures=True)

    def _locations(self, entry, length):
        return tuple(entry.pages[i // self.page_size] * self.page_size + i % self.page_size for i in range(length))

    def cache_req(self, req, *, finished):
        entry = self.requests[req.table_idx]
        end = min(entry.computed_end, len(req.input_ids))
        tokens = tuple(req.input_ids[:end].tolist())
        for position, slot in entry.checkpoints.items():
            if position > end or position in entry.published:
                continue
            self.tree.insert(tokens[:position], self._locations(entry, position), slot)
            entry.published.add(position)
        if not finished:
            return
        if end == entry.computed_end and end > 0:
            self.tree.insert(tokens, self._locations(entry, end), entry.work)
            self.record_cached_session(req, end)
            self._record_history(entry.prompt, tokens)
        self.unlock(entry.handle)
        del self.requests[req.table_idx]
        def release():
            self.pool.allocator.release(entry.pages)
            self.pool.release_state(entry.work)
            for slot in entry.checkpoints.values():
                self.pool.release_state(slot)
        if self._deferred is not None:
            self._deferred.append(release)
        else:
            release()

    def free_to_cached(self, req):
        # Completion is fenced by the scheduler; the tail state follows the last submitted forward.
        return None

    def commit_speculative(self, req, C, num_sampled):
        raise NotImplementedError("Marconi speculative decoding is not supported")

    @contextmanager
    def lazy_free_region(self):
        assert self._deferred is None
        self._deferred = []
        try:
            yield
        finally:
            callbacks, self._deferred = self._deferred, None
            for callback in callbacks:
                callback()

    def check_integrity(self):
        self.tree.check_integrity()
        allocator = self.pool.allocator
        assert len(set(allocator.free)) == len(allocator.free)
        assert all(allocator.refs[b] == 0 for b in allocator.free)
        assert sum(ref == 0 for ref in allocator.refs) == len(allocator.free)

    def profile_result(self):
        result = super().profile_result()
        if result is not None:
            result.update(alpha=self.tree.alpha, checkpoints=sum(n.state is not None for n in self.tree.nodes),
                          free_blocks=len(self.pool.allocator.free), total_blocks=self.plan.num_blocks,
                          eviction=dict(self.tree.stats), tuning_complete=self.tune_results is not None)
        return result


class MarconiCacheFactory(BaseCacheFactory):
    def plan(self, config, available_memory, speculator_reserve):
        if speculator_reserve is not None:
            raise ValueError("Marconi supports autoregressive decoding only")
        layout = marconi_layout(config)
        max_len = align_ceil(config.max_seq_len, 64)
        table_bytes = (config.max_running_req + 1) * max_len * 8
        block_bytes = layout.block_bytes(config.dtype.itemsize)
        blocks = (available_memory - table_bytes) // (block_bytes + 8 * layout.state_groups)
        if config.num_page_override is not None:
            blocks = min(blocks, config.num_page_override)
        if blocks < 3 * layout.state_groups + 2:
            raise ValueError("Insufficient cache budget for Marconi shared blocks")
        return MarconiCachePlan(blocks, blocks * layout.page_size, config.max_seq_len, max_len,
                                layout, blocks, table_bytes + blocks * 8 * layout.state_groups)

    def allocate(self, config, plan, device, dtype):
        pool = HybridBlockPool(plan.layout, plan.num_blocks, device, dtype)
        dummy_page, = pool.allocator.allocate(1)
        dummy_state = pool.allocate_state()
        pool.zero_state(dummy_state)
        shape = (config.max_running_req + 1, plan.aligned_max_seq_len)
        pages = torch.zeros(shape, device=device, dtype=torch.int32)
        pages[-1] = dummy_page * plan.layout.page_size + torch.arange(shape[1], device=device) % plan.layout.page_size
        states = torch.full(shape, dummy_state, device=device, dtype=torch.int32)
        logger.info(f"Marconi shared cache: {plan.num_blocks} blocks, "
                    f"{plan.layout.block_bytes(dtype.itemsize)} bytes/block, "
                    f"KV page={plan.layout.page_size}, state groups={plan.layout.state_groups}")
        return CacheResources(pool, pages, pool, states, None)

    def make_manager(self, config, plan, resources):
        return MarconiCacheManager(resources, config, plan)


SUPPORTED_CACHE_FACTORIES.register("marconi")(MarconiCacheFactory())
