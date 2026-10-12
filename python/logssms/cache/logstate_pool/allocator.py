from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


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
    frames: tuple[int, ...]
    decays: tuple[DecayRecord, ...]


class LogStateAllocator:

    def __init__(
        self,
        *,
        num_frames         : int,
        frame_bytes        : int,
        payload_bytes      : int,
        conv_snapshot_bytes: int,
        decay_bytes        : int,
        num_granules       : int,
        log_payload_bytes  : int | None = None,
    ) -> None:
        assert num_frames > 0 and frame_bytes > 0
        assert payload_bytes > 0 and conv_snapshot_bytes > 0 and decay_bytes > 0
        assert num_granules > 0
        assert decay_bytes <= frame_bytes, "decay records must fit within one frame"
        self.num_frames = num_frames
        self.frame_bytes = frame_bytes
        self.payload_frames = (payload_bytes + frame_bytes - 1) // frame_bytes
        self.log_payload_frames = self.payload_frames if log_payload_bytes is None else (
            log_payload_bytes + frame_bytes - 1
        ) // frame_bytes
        assert 0 < self.log_payload_frames <= self.payload_frames
        self.conv_frames = (conv_snapshot_bytes + frame_bytes - 1) // frame_bytes
        self.decay_records_per_frame = frame_bytes // decay_bytes
        self.num_granules = num_granules
        self._free = list(range(num_frames - 1, -1, -1))
        self._kv: set[int] = set()
        self._slots: dict[int, LinearCacheSlot] = {}
        self._next_slot = 0
        self._free_slot_ids: list[int] = []
        self._decay_masks: dict[int, int] = {}
        self._open_decay_frames: set[int] = set()
        self._free_decay_records = 0
        self._reclaimable_slots: set[int] = set()
        self._reclaimable_frames = 0
        self._reclaimable_decay_masks: dict[int, int] = {}
        self._reclaimable_decay_records = 0
        self._reclaimable_decay_frames = 0
        self._retired_resources: dict[int, RetiredResources] = {}

    def set_reclaimable(self, slot: int, reclaimable: bool) -> None:
        if reclaimable == (slot in self._reclaimable_slots):
            return
        record = self._slots[slot]
        delta = 1 if reclaimable else -1
        if reclaimable:
            self._reclaimable_slots.add(slot)
        else:
            self._reclaimable_slots.remove(slot)
        self._reclaimable_frames += delta * (
            len(record.payload) + sum(map(len, record.conv))
        )
        if record.decay is not None:
            self._set_decay_reclaimable(record.decay, reclaimable)

    def _set_decay_reclaimable(self, record: DecayRecord, reclaimable: bool) -> None:
        frame, index = record.frame, record.index
        mask = self._reclaimable_decay_masks.get(frame, 0)
        occupied = self._decay_masks[frame]
        self._reclaimable_decay_frames -= int(mask == occupied)
        mask = mask | (1 << index) if reclaimable else mask & ~(1 << index)
        if mask:
            self._reclaimable_decay_masks[frame] = mask
        else:
            self._reclaimable_decay_masks.pop(frame, None)
        self._reclaimable_decay_frames += int(mask == occupied)
        self._reclaimable_decay_records += 1 if reclaimable else -1

    def _set_decay_mask(self, frame: int, mask: int) -> None:
        reclaimable = self._reclaimable_decay_masks.get(frame, 0)
        occupied = self._decay_masks.get(frame, 0)
        self._reclaimable_decay_frames -= int(occupied != 0 and reclaimable == occupied)
        if mask:
            self._decay_masks[frame] = mask
        else:
            self._decay_masks.pop(frame, None)
        self._reclaimable_decay_frames += int(mask != 0 and reclaimable == mask)

    @property
    def free_frames(self) -> int:
        return len(self._free)

    @property
    def allocated_kv(self) -> frozenset[int]:
        return frozenset(self._kv)

    @property
    def allocated_slots(self) -> frozenset[int]:
        return frozenset(self._slots)

    def slot(self, slot: int) -> LinearCacheSlot:
        return self._slots[slot]

    def profile_usage(self) -> dict[str, int]:
        kinds = {"log": 0, "checkpoint": 0, "work": 0}
        payload_frames = conv_frames = 0
        for record in self._slots.values():
            kinds[record.kind] += 1
            payload_frames += len(record.payload)
            conv_frames += sum(map(len, record.conv))
        payload_frames += sum(len(record.frames) for record in self._retired_resources.values())
        return {
            "total_frames": self.num_frames,
            "free_frames": self.free_frames,
            "kv_frames": len(self._kv),
            "payload_frames": payload_frames,
            "conv_frames": conv_frames,
            "decay_frames": len(self._decay_masks),
            "free_decay_records": self._free_decay_records,
            "log_slots": kinds["log"],
            "checkpoint_slots": kinds["checkpoint"],
            "work_slots": kinds["work"],
        }

    def _take_decay(self) -> DecayRecord:
        full_mask = (1 << self.decay_records_per_frame) - 1
        if self._open_decay_frames:
            frame = next(iter(self._open_decay_frames))
            mask = self._decay_masks[frame]
            available = ~mask & full_mask
            index = (available & -available).bit_length() - 1
            mask |= 1 << index
            self._set_decay_mask(frame, mask)
            self._free_decay_records -= 1
            if mask == full_mask:
                self._open_decay_frames.remove(frame)
            return DecayRecord(frame, index)
        frame = self._free.pop()
        self._set_decay_mask(frame, 1)
        self._free_decay_records += self.decay_records_per_frame - 1
        if self.decay_records_per_frame > 1:
            self._open_decay_frames.add(frame)
        return DecayRecord(frame, 0)

    def _release_decay(self, record: DecayRecord) -> None:
        mask = self._decay_masks[record.frame]
        bit = 1 << record.index
        assert mask & bit
        mask &= ~bit
        if mask:
            self._set_decay_mask(record.frame, mask)
            self._open_decay_frames.add(record.frame)
            self._free_decay_records += 1
        else:
            self._set_decay_mask(record.frame, 0)
            self._open_decay_frames.discard(record.frame)
            self._free_decay_records -= self.decay_records_per_frame - 1
            self._free.append(record.frame)

    def _take_frames(self, count: int) -> tuple[int, ...]:
        assert count <= len(self._free)
        return tuple(self._free.pop() for _ in range(count))

    def _new_slot(self, kind: Literal["log", "work"]) -> int:
        slot = self._free_slot_ids.pop() if self._free_slot_ids else self._next_slot
        if slot == self._next_slot:
            self._next_slot += 1
        granules = self.num_granules if kind == "log" else 1
        payload = self._take_frames(self.log_payload_frames if kind == "log" else self.payload_frames)
        conv = tuple(self._take_frames(self.conv_frames) for _ in range(granules))
        decay = self._take_decay() if kind == "log" else None
        self._slots[slot] = LinearCacheSlot(payload, conv, decay, kind)
        return slot

    def required_frames(
        self,
        kv                : int,
        logs              : int,
        states            : int,
        free_decay_records: int | None = None,
    ) -> int:
        if free_decay_records is None:
            free_decay_records = self._free_decay_records
        decay_frames = (
            max(0, logs - free_decay_records) + self.decay_records_per_frame - 1
        ) // self.decay_records_per_frame
        return (
            kv
            + logs * self.log_payload_frames + states * self.payload_frames
            + (logs * self.num_granules + states) * self.conv_frames
            + decay_frames
        )

    def projected_free_capacity(
        self, released_slots: Iterable[int] | None = None,
    ) -> tuple[int, int]:
        if released_slots is None:
            return (
                self.free_frames + self._reclaimable_frames + self._reclaimable_decay_frames,
                self._free_decay_records + self._reclaimable_decay_records
                - self._reclaimable_decay_frames * self.decay_records_per_frame,
            )
        free_frames = self.free_frames + sum(len(record.frames) for record in self._retired_resources.values())
        free_decay_records = self._free_decay_records
        released_decay_masks: dict[int, int] = {}
        for record in self._retired_resources.values():
            for decay in record.decays:
                released_decay_masks[decay.frame] = released_decay_masks.get(decay.frame, 0) | 1 << decay.index
        for slot in set(released_slots):
            record = self._slots[slot]
            free_frames += len(record.payload) + sum(
                len(snapshot) for snapshot in record.conv
            )
            if record.decay is not None:
                decay = record.decay
                released_decay_masks[decay.frame] = (
                    released_decay_masks.get(decay.frame, 0) | 1 << decay.index
                )

        for frame, released_mask in released_decay_masks.items():
            occupied_mask = self._decay_masks[frame]
            assert released_mask & occupied_mask == released_mask
            if occupied_mask & ~released_mask:
                free_decay_records += released_mask.bit_count()
            else:
                free_frames += 1
                free_decay_records -= (
                    self.decay_records_per_frame - occupied_mask.bit_count()
                )
        return free_frames, free_decay_records

    def lock_frame_cost(self, slots: Iterable[int]) -> int:
        frames = 0
        decay_frames = set()
        for slot in slots:
            if slot not in self._reclaimable_slots:
                continue
            record = self._slots[slot]
            frames += len(record.payload) + sum(map(len, record.conv))
            if record.decay is not None:
                frame = record.decay.frame
                if self._reclaimable_decay_masks.get(frame, 0) == self._decay_masks[frame]:
                    decay_frames.add(frame)
        return frames + len(decay_frames)

    def admissible_pages(
        self,
        *,
        evictable_kv_frames: int,
        evictable_slots    : Iterable[int] | None = None,
        free_capacity     : tuple[int, int] | None = None,
    ) -> int:
        free_frames, free_decay_records = (
            self.projected_free_capacity(evictable_slots)
            if free_capacity is None else free_capacity
        )
        free_frames += evictable_kv_frames
        page_frames = 1 + self.log_payload_frames + self.num_granules * self.conv_frames
        work_frames = self.payload_frames + self.conv_frames
        upper = max(0, (free_frames - work_frames) // page_frames)
        lower = 0
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if self.required_frames(
                middle, middle, 1, free_decay_records
            ) <= free_frames:
                lower = middle
            else:
                upper = middle - 1
        return lower

    def reserve(
        self, *, kv: int = 0, logs: int = 0, states: int = 0
    ) -> tuple[list[int], list[int], list[int]] | None:
        assert kv >= 0 and logs >= 0 and states >= 0
        if self.required_frames(kv, logs, states) > self.free_frames:
            return None
        kv_frames = list(self._take_frames(kv))
        self._kv.update(kv_frames)
        log_slots = [self._new_slot("log") for _ in range(logs)]
        state_slots = [self._new_slot("work") for _ in range(states)]
        return kv_frames, log_slots, state_slots

    def materialize(self, slot: int) -> tuple[int, ...]:
        record = self._slots[slot]
        assert record.kind == "log" and record.decay is not None
        assert len(record.payload) == self.payload_frames, "Compact Log needs merged checkpoint storage"
        reclaimable = slot in self._reclaimable_slots
        self.set_reclaimable(slot, False)
        released = tuple(frame for snapshot in record.conv[:-1] for frame in snapshot)
        self._free.extend(released)
        self._release_decay(record.decay)
        self._slots[slot] = LinearCacheSlot(
            record.payload, (record.conv[-1],), None, "checkpoint"
        )
        if reclaimable:
            self.set_reclaimable(slot, True)
        return released

    def checkpoint_frame_capacity(self, slots: Iterable[int]) -> int:
        records = [self._slots[slot] for slot in slots]
        return sum(len(record.payload) + sum(map(len, record.conv)) for record in records) - self.conv_frames

    def checkpoint_frame_savings(self, slots: Iterable[int]) -> int:
        slots = tuple(slots)
        savings = self.checkpoint_frame_capacity(slots) - self.payload_frames
        decay_masks = {}
        for slot in slots:
            decay = self._slots[slot].decay
            if decay is not None:
                decay_masks[decay.frame] = decay_masks.get(decay.frame, 0) | (1 << decay.index)
        return savings + sum(mask == self._decay_masks[frame] for frame, mask in decay_masks.items())

    def coalesce_to_checkpoint(self, slots: tuple[int, ...]) -> RetiredResources:
        assert len(slots) == len(set(slots)) and len(slots) > 1
        assert self.checkpoint_frame_capacity(slots) >= self.payload_frames
        assert self.checkpoint_frame_savings(slots) > 0, "Materialization must release more than its expansion"
        records = [self._slots[slot] for slot in slots]
        reclaimable = [slot for slot in slots if slot in self._reclaimable_slots]
        for slot in reclaimable:
            self.set_reclaimable(slot, False)
        terminal = records[-1].conv[-1]
        frames = [frame for record in reversed(records) for frame in record.payload]
        frames.extend(frame for record in records for conv in record.conv for frame in conv if frame not in terminal)
        payload, surplus = tuple(frames[:self.payload_frames]), tuple(frames[self.payload_frames:])
        for slot in slots[:-1]:
            self._slots[slot] = LinearCacheSlot((), (), None, "checkpoint")
        self._slots[slots[-1]] = LinearCacheSlot(payload, (terminal,), None, "checkpoint")
        retirement = RetiredResources(surplus, tuple(record.decay for record in records if record.decay is not None))
        self._retired_resources[id(retirement)] = retirement
        self._reclaimable_frames += len(surplus)
        for decay in retirement.decays:
            self._set_decay_reclaimable(decay, True)
        for slot in reclaimable:
            self.set_reclaimable(slot, True)
        return retirement

    def release_retirement(self, record: RetiredResources) -> None:
        assert self._retired_resources.pop(id(record)) is record
        self._reclaimable_frames -= len(record.frames)
        self._free.extend(record.frames)
        for decay in record.decays:
            self._set_decay_reclaimable(decay, False)
            self._release_decay(decay)

    def commit_checkpoint(self, slot: int) -> None:
        record = self._slots[slot]
        assert record.kind == "work" and record.decay is None
        self._slots[slot] = LinearCacheSlot(
            record.payload, record.conv, None, "checkpoint"
        )

    def release_kv(self, frames: list[int] | tuple[int, ...]) -> None:
        assert len(frames) == len(set(frames)) and set(frames) <= self._kv
        self._kv.difference_update(frames)
        self._free.extend(frames)

    def release_slot(self, slot: int) -> None:
        self.set_reclaimable(slot, False)
        record = self._slots.pop(slot)
        self._free.extend(record.payload)
        for snapshot in record.conv:
            self._free.extend(snapshot)
        if record.decay is not None:
            self._release_decay(record.decay)
        self._free_slot_ids.append(slot)

    def check_integrity(self) -> None:
        assert self._reclaimable_slots <= self._slots.keys()
        assert self.projected_free_capacity() == self.projected_free_capacity(self._reclaimable_slots)
        owned = list(self._kv) + list(self._free) + list(self._decay_masks)
        decay_records: set[DecayRecord] = set()
        for record in self._retired_resources.values():
            owned.extend(record.frames)
            decay_records.update(record.decays)
        for record in self._slots.values():
            owned.extend(record.payload)
            for snapshot in record.conv:
                owned.extend(snapshot)
            assert (record.decay is not None) == (record.kind == "log")
            if record.decay is not None:
                assert self._decay_masks[record.decay.frame] & (1 << record.decay.index)
                assert record.decay not in decay_records
                decay_records.add(record.decay)
        assert sum(mask.bit_count() for mask in self._decay_masks.values()) == len(decay_records)
        assert self._open_decay_frames == {
            frame for frame, mask in self._decay_masks.items()
            if mask.bit_count() < self.decay_records_per_frame
        }
        assert self._free_decay_records == sum(
            self.decay_records_per_frame - mask.bit_count()
            for mask in self._decay_masks.values()
        )
        assert len(owned) == self.num_frames
        assert set(owned) == set(range(self.num_frames))
        assert len(set(owned)) == len(owned)
