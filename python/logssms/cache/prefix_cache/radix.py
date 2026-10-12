from __future__ import annotations

import heapq
import time
from itertools import count
from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, Generic, TypeVar

import torch

from logssms.core import get_global_ctx
from logssms.utils import align_down

from .base import (
    BaseCacheHandle, 
    InsertResult, 
    MatchResult, 
    SizeInfo
)
from .standard import StandardPrefixCacheBase, StandardPrefixValue


KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    else:
        return lambda x: tuple(x[:page_size].tolist())

    
class RadixTreeNode:
    counter = count(0)

    def __init__(self, key_fn: KEY_FN, page_size: int = 1, tic: int | None = None):
        self.uuid = next(RadixTreeNode.counter)
        self.key_fn = key_fn
        self.page_size = page_size
        self.children: dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.timestamp = tic or time.monotonic_ns()


        self._key: torch.Tensor
        self._kv: torch.Tensor
        self._length: int

    def set(
        self,
        begin     : int,
        end       : int,
        input_ids : torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> None:
        assert len(input_ids) == len(kv_indices)
        self._key = input_ids[begin:end]
        self._kv = kv_indices[begin:end].clone()
        self._length = len(self._key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        assert isinstance(parent, RadixTreeNode)
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        return self._parent

    @property
    def kv(self) -> torch.Tensor:
        return self._kv

    @property
    def value(self) -> tuple[torch.Tensor]:
        return self._kv,

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from logssms.kernel import fast_compare_key

        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.page_size, self.timestamp)
        new_node.set(0, pos, self._key, self._kv)
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set(pos, len(self._key), self._key, self._kv)
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


class HybridRadixTreeNode(RadixTreeNode):
    def __init__(self, key_fn: KEY_FN, page_size: int = 1, tic: int | None = None):
        super().__init__(key_fn, page_size, tic)
        self._st: torch.Tensor

    def set(
        self,
        begin     : int,
        end       : int,
        inputy_ids: torch.Tensor,
        kv_indices: torch.Tensor,
        st_indices: torch.Tensor,
    ) -> None:
        assert len(st_indices) == len(kv_indices)
        super().set(begin, end, inputy_ids, kv_indices)
        self._st = st_indices[begin: end].clone()

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = HybridRadixTreeNode(self.key_fn, self.page_size, self.timestamp)
        new_node.set(0, pos, self._key, self._kv, self._st)
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set(pos, len(self._key), self._key, self._kv, self._st)
        self.set_parent(new_node)

        return new_node

    @property
    def st(self) -> torch.Tensor:
        return self._st

    @property
    def value(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv, self._st


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node        : RadixTreeNode
    empty_tensor: torch.Tensor

    def get_matched_indices(self) -> tuple[torch.Tensor, ...] | torch.Tensor:
        node = self.node
        if node.is_root():
            if isinstance(node, HybridRadixTreeNode):
                return self.empty_tensor, self.empty_tensor
            return self.empty_tensor
        num_values = len(node.value)
        value_list: list[list[torch.Tensor]] = [[] for _ in range(num_values)]
        while not node.is_root():
            for i, value in enumerate(node.value):
                value_list[i].append(value)
            node = node.parent
        matched = tuple(torch.cat(list(reversed(values))) for values in value_list)
        return matched if len(matched) > 1 else matched[0]


NodeT = TypeVar("NodeT", bound=RadixTreeNode)
class RadixPrefixCache(StandardPrefixCacheBase, Generic[NodeT]):
    def __init__(self, device: torch.device, node_type: type[NodeT]):
        super().__init__()
        self.NodeType: type[NodeT] = node_type
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = self.NodeType(self.key_fn, self.page_size)
        self.root_node.ref_count = 1

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(
            RadixCacheHandle(prefix_len, node, self.empty_tensor),
            tree_match_len=prefix_len,
            recoverable_len=prefix_len,
        )

    def trace_prefix(self, input_ids: torch.Tensor) -> tuple[NodeT, int]:
        node = self.root_node
        prefix_len = 0
        while prefix_len < len(input_ids):
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                break
            match_len = align_down(
                child.get_match_len(input_ids[prefix_len:]), self.page_size
            )
            if match_len == 0:
                break
            node = child
            prefix_len += match_len
            if match_len != child.length:
                break
        return node, prefix_len

    def insert_prefix(
        self,
        input_ids: torch.Tensor,
        value    : StandardPrefixValue,
    ) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = self.NodeType(self.key_fn, self.page_size)
            if isinstance(new_node, HybridRadixTreeNode):
                assert value.state_indices is not None
                new_node.set(
                    prefix_len, insert_len, input_ids,
                    value.kv_indices, value.state_indices,
                )
            else:
                assert value.state_indices is None
                new_node.set(prefix_len, insert_len, input_ids, value.kv_indices)
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node, self.empty_tensor))

    def evict(
        self,
        size         : int,
        evicted_nodes: list[tuple[int, int, int]] | None = None,
    ) -> StandardPrefixValue:
        if size == 0:
            return StandardPrefixValue(self.empty_tensor)
        assert size <= self.evictable_size, f"Can't evict {size}, only {self.evictable_size}"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: list[torch.Tensor] = []
        evicted_state  : list[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leave_nodes, f"Can't evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            if evicted_nodes is not None:
                end = 0
                ancestor = node
                while not ancestor.is_root():
                    end += ancestor.length
                    ancestor = ancestor.parent
                evicted_nodes.append((node.uuid, end - node.length, end))
            evicted_size += node.length
            values = node.value
            evicted_indices.append(values[0])
            if len(values) > 1:
                evicted_state.append(values[1][:: self.page_size])
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]

            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return StandardPrefixValue(
            torch.cat(evicted_indices),
            torch.cat(evicted_state) if evicted_state else None,
        )

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        evictable_size = 0
        protected_size = 0
        seen: set[int] = set()
        stack: list[NodeT] = [self.root_node]

        while stack:
            node = stack.pop()
            if node.uuid in seen:
                raise RuntimeError(f"Radix cache contains a cycle or duplicate node {node.uuid}.")
            seen.add(node.uuid)

            if node.is_root():
                if node.ref_count != 1:
                    raise RuntimeError(
                        f"Radix root ref_count must be 1, got {node.ref_count}."
                    )
            else:
                if node.parent.children.get(self.key_fn(node._key)) is not node:
                    raise RuntimeError(
                        f"Radix node {node.uuid} is not indexed by its parent."
                    )
                if node.length <= 0 or node.length % self.page_size != 0:
                    raise RuntimeError(
                        f"Radix node {node.uuid} has invalid length {node.length}."
                    )
                if len(node.kv) != node.length:
                    raise RuntimeError(
                        f"Radix node {node.uuid} has {len(node.kv)} KV indices for"
                        f" length {node.length}."
                    )
                if isinstance(node, HybridRadixTreeNode):
                    expected_states = node.length
                    if len(node.st) != expected_states:
                        raise RuntimeError(
                            f"Radix node {node.uuid} has {len(node.st)} state indices;"
                            f" expected {expected_states}."
                        )
                if node.ref_count == 0:
                    evictable_size += node.length
                elif node.ref_count > 0:
                    protected_size += node.length
                else:
                    raise RuntimeError(
                        f"Radix node {node.uuid} has negative ref_count {node.ref_count}."
                    )

            stack.extend(node.children.values())

        if evictable_size != self.evictable_size or protected_size != self.protected_size:
            raise RuntimeError(
                "Radix cache size accounting mismatch:"
                f" recorded=({self.evictable_size} evictable,"
                f" {self.protected_size} protected),"
                f" actual=({evictable_size} evictable, {protected_size} protected)."
            )

    def total_state_pages(self) -> int:
        total = 0
        stack: list[NodeT] = [self.root_node]
        while stack:
            node = stack.pop()
            if not node.is_root() and len(node.value) > 1:
                total += len(node.value[1]) // self.page_size
            stack.extend(node.children.values())
        return total

    def _collect_leave_nodes_for_evict(self) -> list[NodeT]:
        nodes: list[NodeT] = [self.root_node]
        leave_nodes: list[NodeT] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[NodeT, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                break
            node = child_node


            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len


            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                break


            node.timestamp = tic

        return node, prefix_len
