from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import torch

from logssms.cache.prefix_cache import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@dataclass(eq=False)
class MarconiNode:
    tokens: tuple[int, ...]
    locations: tuple[int, ...]
    parent: MarconiNode | None = None
    children: dict[int, MarconiNode] = field(default_factory=dict)
    state: int | None = None
    accessed: int = 0
    locks: int = 0

    @property
    def end(self):
        return len(self.tokens)

    @property
    def begin(self):
        return 0 if self.parent is None else self.parent.end


@dataclass(frozen=True)
class MarconiHandle(BaseCacheHandle):
    node: MarconiNode
    device: torch.device
    branch_end: int = 0
    input_tokens: tuple[int, ...] = ()

    def get_matched_indices(self):
        return torch.tensor(self.node.locations, dtype=torch.int32, device=self.device)


@dataclass(frozen=True)
class MarconiCost:
    linear_flops: float
    attention_flops: float
    state_bytes: int
    kv_token_bytes: int

    def saved_flops(self, begin, end):
        return self.linear_flops * (end - begin) + self.attention_flops * (end * end - begin * begin)

    def efficiency(self, begin, end):
        return self.saved_flops(begin, end) / max(1, self.state_bytes + (end - begin) * self.kv_token_bytes)


class MarconiPrefixCache(BasePrefixCache):
    def __init__(self, device, page_size: int, cost: MarconiCost, *, pool=None):
        self.device = torch.device(device)
        self.page_size = page_size
        self.cost = cost
        self.pool = pool
        self.root = MarconiNode((), ())
        self.nodes: set[MarconiNode] = set()
        self.clock = 0
        self.alpha = 0.0
        self.on_first_eviction: Callable | None = None
        self.evicted = False
        self.stats = Counter()

    def _blocks(self, node):
        return set(x // self.page_size for x in node.locations)

    def _retain(self, blocks):
        if self.pool is not None:
            self.pool.allocator.retain(blocks)

    def _release(self, blocks):
        if self.pool is not None:
            self.pool.allocator.release(blocks)

    def _walk(self, tokens):
        node, matched, best = self.root, 0, self.root
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            n = common_prefix(child.tokens[matched:], tokens[matched:])
            matched += n
            if matched < child.end:
                break
            node = child
            if node.state is not None:
                best = node
        return matched, best

    def match_prefix(self, input_ids):
        tokens = tuple(input_ids.tolist()) if isinstance(input_ids, torch.Tensor) else tuple(input_ids)
        matched, best = self._walk(tokens)
        # Snapshot positions use the prefill kernel's chunk boundaries.
        branch = best.end + ((matched - best.end) // 64) * 64
        if branch == best.end:
            branch = 0
        handle = MarconiHandle(best.end, best, self.device, branch, tokens)
        return MatchResult(handle, matched, best.end)

    def touch(self, node):
        self.clock += 1
        node.accessed = self.clock

    def lock_handle(self, handle, unlock=False):
        node = handle.node
        delta = -1 if unlock else 1
        while node is not self.root:
            node.locks += delta
            assert node.locks >= 0
            node = node.parent

    def _ensure(self, tokens, locations):
        parent = self.root
        while parent.end < len(tokens):
            key = tokens[parent.end]
            child = parent.children.get(key)
            if child is None:
                child = MarconiNode(tokens, locations, parent)
                parent.children[key] = child
                self.nodes.add(child)
                self._retain(self._blocks(child))
                return child
            split = parent.end + common_prefix(tokens[parent.end:], child.tokens[parent.end:])
            if split < child.end:
                middle = MarconiNode(child.tokens[:split], child.locations[:split], parent,
                                     accessed=child.accessed, locks=child.locks)
                parent.children[key] = middle
                middle.children[child.tokens[split]] = child
                child.parent = middle
                self.nodes.add(middle)
                self._retain(self._blocks(middle))
                child = middle
            parent = child
        return parent

    def insert(self, tokens, locations, state):
        tokens, locations = tuple(tokens), tuple(locations)
        assert len(tokens) == len(locations) and tokens
        parent = self.root
        while parent.end < len(tokens):
            child = parent.children.get(tokens[parent.end])
            if child is None:
                break
            matched = parent.end + common_prefix(tokens[parent.end:], child.tokens[parent.end:])
            whole_pages = matched // self.page_size * self.page_size
            locations = child.locations[:whole_pages] + locations[whole_pages:]
            if matched < child.end:
                break
            parent = child
        node = self._ensure(tokens, locations)
        if node.state is None:
            node.state = state
            if self.pool is not None:
                self.pool.retain_state(state)
            self.stats["checkpoints_admitted"] += 1
        self.touch(node)
        return node

    def candidates(self):
        return [n for n in self.nodes if not n.locks and len(n.children) <= 1]

    def eviction_order(self):
        candidates = self.candidates()
        if not candidates:
            return []
        times = [n.accessed for n in self.nodes]
        efficiencies = [self.cost.efficiency(n.begin, n.end) for n in self.nodes]
        lo_t, hi_t, lo_f, hi_f = min(times), max(times), min(efficiencies), max(efficiencies)
        def score(n):
            recency = (n.accessed - lo_t) / max(1, hi_t - lo_t)
            efficiency = (self.cost.efficiency(n.begin, n.end) - lo_f) / max(1e-30, hi_f - lo_f)
            return recency + self.alpha * efficiency, n.accessed, n.tokens
        return sorted(candidates, key=score)

    def evict_one(self):
        candidates = self.eviction_order()
        if not candidates:
            return False
        if not self.evicted:
            self.evicted = True
            if self.on_first_eviction is not None:
                self.on_first_eviction()
        node = candidates[0]
        parent = node.parent
        key = node.tokens[parent.end]
        if node.children:
            child = next(iter(node.children.values()))
            child.parent = parent
            parent.children[key] = child
            self.stats["intermediate_evictions"] += 1
        else:
            del parent.children[key]
            self.stats["leaf_evictions"] += 1
        self._release(self._blocks(node))
        if node.state is not None and self.pool is not None:
            self.pool.release_state(node.state)
        self.nodes.remove(node)
        return True

    def reclaimable_blocks(self):
        if self.pool is None:
            return 0
        references = Counter()
        for node in self.nodes:
            if not node.locks:
                references.update(self._blocks(node))
                if node.state is not None and self.pool.state_refs[node.state] == 1:
                    references.update(self.pool.state_blocks[node.state])
        return sum(self.pool.allocator.refs[b] == count for b, count in references.items())

    @property
    def size_info(self):
        protected = sum(n.end - n.begin for n in self.nodes if n.locks)
        total = sum(n.end - n.begin for n in self.nodes)
        return SizeInfo(total - protected, protected)

    def reset(self):
        if any(n.locks for n in self.nodes):
            raise RuntimeError("Cannot reset a cache with active handles")
        while self.evict_one():
            pass

    def check_integrity(self):
        for node in self.nodes:
            assert node.parent is not None and node.end > node.begin
            assert node.tokens[:node.begin] == node.parent.tokens
            assert node.parent.children[node.tokens[node.begin]] is node
            assert len(node.locations) == node.end and node.locks >= 0
            assert node.state is not None or node.children

    def snapshot(self):
        return [(n.tokens, n.accessed) for n in self.nodes if n.state is not None]


def tune_marconi(snapshot, history, page_size, cost, capacity_bytes):
    results = []
    for step in range(21):
        tree = MarconiPrefixCache("cpu", page_size, cost)
        tree.alpha = step / 10
        for tokens, stamp in sorted(snapshot, key=lambda entry: len(entry[0])):
            tree.insert(tokens, range(len(tokens)), 0).accessed = stamp
            tree.clock = max(tree.clock, stamp)
        saved, hits = 0.0, 0
        def usage():
            pages = {n.tokens[:min(end, n.end)] for n in tree.nodes
                     for end in range(page_size, n.end + page_size, page_size)}
            return (len(pages) * page_size * cost.kv_token_bytes
                    + sum(n.state is not None for n in tree.nodes) * cost.state_bytes)
        for prompt, sequence in history:
            match = tree.match_prefix(prompt[:-1])
            h = match.cuda_handle
            tree.touch(h.node)
            hits += h.cached_len
            saved += cost.saved_flops(0, h.cached_len)
            if h.branch_end:
                tree.insert(sequence[:h.branch_end], range(h.branch_end), 0)
            tree.insert(sequence, range(len(sequence)), 0)
            while usage() > capacity_bytes and tree.evict_one():
                pass
        results.append((saved, hits, -step, step / 10))
    return max(results)[-1], results
