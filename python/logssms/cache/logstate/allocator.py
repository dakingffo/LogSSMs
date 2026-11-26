from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from .native import load_logstate_core

_KIND_NAMES = ("free", "log", "work", "checkpoint")


@dataclass(frozen=True)
class DecayRecord:
    frame: int
    index: int


@dataclass(frozen=True)
class LinearCacheSlot:
    payload: tuple[int, ...]
    conv: tuple[tuple[int, ...], ...]
    decay: DecayRecord | None
    kind: Literal["log", "work", "checkpoint"]


@dataclass(frozen=True)
class RetiredResources:
    retire_id: int
    frames: tuple[int, ...] = ()
    decays: tuple[DecayRecord, ...] = ()


class LogStateAllocator:
    """Thin Python facade over the C++ AllocatorCore."""

    def __init__(
        self, *, num_frames: int, frame_bytes: int, payload_bytes: int,
        conv_snapshot_bytes: int, decay_bytes: int, num_granules: int,
        log_payload_bytes: int | None = None,
    ) -> None:
        self._core = dict(load_logstate_core().make_allocator(
            num_frames, frame_bytes, payload_bytes, conv_snapshot_bytes,
            decay_bytes, num_granules, -1 if log_payload_bytes is None else log_payload_bytes,
        ))
        self.core_id: int = int(self._core["_id"])
        for attr in (
            "num_frames", "frame_bytes", "payload_frames", "log_payload_frames",
            "descriptor_frames", "checkpoint_log_count", "conv_frames",
            "decay_records_per_frame", "num_granules", "materialize_log_count",
        ):
            setattr(self, attr, int(self._core[attr]))

    @property
    def free_frames(self) -> int:
        return int(self._core["free_frames"]())

    @property
    def _free_decay_records(self) -> int:
        return int(self._core["free_decay_records"]())

    @property
    def allocated_kv(self) -> frozenset[int]:
        return frozenset(torch.from_dlpack(self._core["allocated_kv"]()).tolist())

    @property
    def allocated_slots(self) -> frozenset[int]:
        return frozenset(torch.from_dlpack(self._core["allocated_slots"]()).tolist())

    @property
    def _reclaimable_slots(self) -> set[int]:
        return set(torch.from_dlpack(self._core["reclaimable_slots"]()).tolist())

    def set_reclaimable(self, slot: int, reclaimable: bool) -> None:
        self._core["set_reclaimable"](int(slot), bool(reclaimable))

    def slot(self, slot: int) -> LinearCacheSlot:
        flat = torch.from_dlpack(self._core["slot_info"](int(slot))).tolist()
        kind, granules, df, di, n_p = _KIND_NAMES[flat[0]], flat[1], flat[2], flat[3], flat[4]
        off, cf = 5 + n_p, self.conv_frames
        conv = tuple(tuple(flat[off + g * cf:off + (g + 1) * cf]) for g in range(granules))
        return LinearCacheSlot(tuple(flat[5:off]), conv, DecayRecord(df, di) if df >= 0 else None, kind)

    def seed_checkpoint_slot(self, slot: int) -> None:
        self._core["seed_checkpoint_slot"](int(slot))

    def profile_usage(self) -> dict[str, int]:
        u = self._core["usage_stats"]()
        return {k: int(u[k]) for k in (
            "total_frames", "free_frames", "kv_frames", "payload_frames",
            "conv_frames", "decay_frames", "free_decay_records",
            "log_slots", "checkpoint_slots", "work_slots",
        )}

    def frame_usage_summary(self) -> dict[str, int]:
        u = self._core["usage_stats"]()
        decay, log_pc = int(u["decay_frames"]), int(u["log_payload_conv"])
        return {
            "log": log_pc + decay, "log_payload_conv": log_pc, "decay": decay,
            "checkpoint": int(u["checkpoint_payload_conv"]),
            "work": int(u["work_payload_conv"]),
            "kv": int(u["kv_frames"]), "free": int(u["free_frames"]),
        }

    def required_frames(
        self, kv: int, logs: int, states: int, free_decay_records: int | None = None,
    ) -> int:
        return int(self._core["required_frames"](
            int(kv), int(logs), int(states),
            -1 if free_decay_records is None else int(free_decay_records),
        ))

    def projected_free_capacity(self, released_slots: Iterable[int] | None = None) -> tuple[int, int]:
        if released_slots is None:
            ff, fd = self._core["projected_free_capacity"]()
            return int(ff), int(fd)
        ff, fd = self.free_frames, self._free_decay_records
        rel_decays: dict[int, set[int]] = {}
        for s in set(released_slots):
            rec = self.slot(s)
            ff += len(rec.payload) + sum(len(c) for c in rec.conv)
            if rec.decay is not None:
                rel_decays.setdefault(rec.decay.frame, set()).add(rec.decay.index)
        for frame, idxs in rel_decays.items():
            occ = sum(1 for s in self.allocated_slots if (d := self.slot(s).decay) and d.frame == frame)
            if len(idxs) < occ:
                fd += len(idxs)
            else:
                ff += 1
                fd -= self.decay_records_per_frame - occ
        return ff, fd

    def lock_frame_cost(self, slots: Iterable[int]) -> int:
        return int(self._core["lock_frame_cost"]([int(s) for s in slots]))

    def admissible_pages(
        self, *, evictable_kv_frames: int, evictable_slots: Iterable[int] | None = None,
        free_capacity: tuple[int, int] | None = None,
    ) -> int:
        if evictable_slots is None and free_capacity is None:
            return int(self._core["admissible_pages"](int(evictable_kv_frames)))
        ff, fdr = self.projected_free_capacity(evictable_slots) if free_capacity is None else free_capacity
        ff += evictable_kv_frames
        pf = 1 + self.log_payload_frames + self.num_granules * self.conv_frames
        wf = self.payload_frames + self.conv_frames
        lo, hi = 0, max(0, (ff - wf) // pf)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.required_frames(mid, mid, 1, fdr) <= ff:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def reserve(
        self, *, kv: int = 0, logs: int = 0, states: int = 0,
    ) -> tuple[list[int], list[int], list[int]] | None:
        res = self._core["reserve"](int(kv), int(logs), int(states))
        if not res["ok"]:
            return None
        return (
            torch.from_dlpack(res["kv"]).tolist(),
            torch.from_dlpack(res["logs"]).tolist(),
            torch.from_dlpack(res["works"]).tolist(),
        )

    def checkpoint_frame_capacity(self, slots: Iterable[int]) -> int:
        return int(self._core["checkpoint_frame_capacity"]([int(s) for s in slots]))

    def checkpoint_frame_savings(self, slots: Iterable[int]) -> int:
        return int(self._core["checkpoint_frame_savings"]([int(s) for s in slots]))

    def materialize_group(self, slots: tuple[int, ...]) -> RetiredResources:
        assert len(slots) == len(set(slots)) and len(slots) >= 1
        try:
            res = self._core["materialize_group"]([int(s) for s in slots])
        except RuntimeError as exc:
            raise AssertionError(str(exc)) from exc
        return RetiredResources(int(res["id"]), tuple(torch.from_dlpack(res["frames"]).tolist()))

    def release_retirement(self, record: RetiredResources) -> None:
        self._core["release_retirement"](int(record.retire_id))

    def release_kv(self, frames: list[int] | tuple[int, ...]) -> None:
        if frames:
            self._core["release_kv"]([int(f) for f in frames])

    def release_slot(self, slot: int) -> None:
        self._core["release_slot"](int(slot))

    def release_slots(self, slots: list[int] | tuple[int, ...]) -> None:
        if slots:
            self._core["release_slots"]([int(s) for s in slots])

    def pack_descriptors(
        self, slots: list[int] | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self._core["pack_descriptors"]([int(s) for s in slots])
        return tuple(torch.from_dlpack(p[k]) for k in ("indices", "payload", "conv", "decay"))  # type: ignore[return-value]

    def check_integrity(self) -> None:
        self._core["check_integrity"]()
