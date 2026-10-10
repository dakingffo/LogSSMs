import random

import pytest
import torch

from logssms.kernel import logstate as planning


@pytest.mark.parametrize("granularity", [16, 32, 64])
def test_native_branches_preserve_token_prefixes_and_ignore_partial_granules(granularity):
    common = torch.arange(granularity, dtype=torch.int64)
    inputs = [
        torch.cat((common, torch.full((granularity + 3,), value, dtype=torch.int64)))
        for value in (101, 102, 103)
    ]
    inputs[1] = inputs[1].to(torch.int32)
    assert planning.batch_branch_points(inputs, granularity) == [[granularity, 0, 1, 2]]
    assert planning.batch_branch_points([], granularity) == []
    assert planning.batch_branch_points(inputs[:1], granularity) == []


def test_native_log_coverage_reports_gap_and_original_entry_indices():
    # Unsorted intervals; checkpoint does not count as a replay log.
    path = [
        3, 128, 192, -1, 128, 1, 0, 0,
        1, 0, 64, -1, 0, 1, 0, 0,
        2, 64, 128, 1, 64, 1, 0, 0,
    ]
    assert planning.cover_logs(path, 0, 192) == [64, 1]
    assert planning.cover_logs(path, 128, 160) == [160, 0]


def test_native_materialize_respects_fragmented_and_protected_slots():
    path = [
        0, 0, 64, 1, 0, 1, 0, 0,
        1, 64, 128, -1, 64, 1, 0, 0,
        2, 128, 192, -1, 128, 1, 0, 1,
    ]
    assert planning.plan_materialize([path], [()], [(0, 1), (0, 2)], 64) == [[0, 0, 1]]
    # A pending branch inside the physical page protects the entire Log.
    assert planning.plan_materialize([path], [(96,)], [(0, 1)], 64) == []
    path[8 + 6] = 1
    assert planning.plan_materialize([path], [()], [(0, 1)], 64) == []


def test_native_coalesce_matches_nondecreasing_group_oracle():
    rng = random.Random(744)
    for _ in range(100):
        spans = [rng.randint(1, 9) for _ in range(rng.randint(1, 32))]
        intensity = rng.randint(2, 8)
        path = []
        tail = 0
        for slot, span in enumerate(spans):
            end = tail + span * 64
            path.extend((slot, tail, end, span, tail, 1, 0, 0))
            tail = end
        expected = []
        begin = 0
        while begin < len(spans):
            end = begin + 1
            while (end < len(spans) and end - begin < intensity
                   and spans[end - 1] <= spans[end]):
                end += 1
            if end - begin > 1:
                expected.append(tuple(range(begin, end)))
            begin = end
        groups, _ = planning.plan_coalesce(
            [path], [()], [()], [(0, tail, intensity, 1)], (), 64, False,
        )
        assert [tuple(group[3::2]) for group in groups] == expected
        assert all(group[2] == sum(group[4::2]) for group in groups)


def test_native_coalesce_obeys_protections_locks_and_unique_slot_ownership():
    path = [value for slot in range(6) for value in (
        slot, slot * 64, (slot + 1) * 64, 1, slot * 64, 1, int(slot == 4), 0,
    )]
    groups, diagnostics = planning.plan_coalesce(
        [path], [(128,)], [(256,)], [(0, 384, 8, 1), (0, 384, 8, 1)],
        (), 64, True,
    )
    assert [tuple(group[3::2]) for group in groups] == [(0, 1), (2, 3)]
    assert diagnostics["planned_slot_releases"] == 2


def test_native_metadata_validation_precedes_pointer_access():
    module = planning._load_logstate_module()
    bad = torch.tensor([10, 1], dtype=torch.int64)
    empty = torch.empty(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="Invalid packed planning row"):
        module.plan_materialize(bad, empty, empty, 64)
    with pytest.raises(RuntimeError, match="CPU int64"):
        module.plan_materialize(bad.to(torch.int32), empty, empty, 64)


def test_native_eviction_matches_shared_frame_and_decay_capacity_oracle():
    import heapq
    from collections import Counter

    rng = random.Random(381)
    for _ in range(200):
        count = rng.randrange(2, 40)
        parents = [-1] + [rng.randrange(i) for i in range(1, count)]
        locked = [1] + [int(rng.random() < .1) for _ in range(1, count)]
        for i in range(count - 1, 0, -1):
            locked[parents[i]] |= locked[i]
        nodes = [(i, parents[i], locked[i], rng.randrange(2), rng.randrange(100))
                 for i in range(count)]
        children = Counter(parents[1:])
        heap = [(row[3], row[4], row[0]) for row in nodes
                if not locked[row[0]] and not children[row[0]]]
        heapq.heapify(heap)
        order = []
        while heap:
            _, _, i = heapq.heappop(heap)
            order.append(i)
            parent = parents[i]
            if parent >= 0:
                children[parent] -= 1
                if not children[parent] and not locked[parent]:
                    row = nodes[parent]
                    heapq.heappush(heap, (row[3], row[4], row[0]))
        assert planning.eviction_candidates(nodes) == order

        page_refs = [[rng.randrange(10) for _ in range(rng.randrange(4))]
                     for _ in nodes]
        slot_refs = [[rng.randrange(12) for _ in range(rng.randrange(4))]
                     for _ in nodes]
        page_totals = Counter(frame for refs in page_refs for frame in refs)
        slot_totals = Counter(slot for refs in slot_refs for slot in refs)
        kv = [[value for frame, refs in Counter(row).items()
               for value in (frame, refs, page_totals[frame])] for row in page_refs]
        # Three partially occupied decay frames, including records owned by
        # live requests outside this tree. Checkpoints do not own decay.
        slots = [(slot, refs, rng.randrange(1, 8), slot // 4 if rng.random() > .25 else -1, slot % 4)
                 for slot, refs in slot_totals.items()]
        own_decay = Counter(row[3] for row in slots if row[3] >= 0)
        occupied = [(frame, rng.randrange(max(1, own_decay[frame]), 5)) for frame in range(3)]
        config = [rng.randrange(3), sum(4 - n for _, n in occupied), 2, 1, 2, 4,
                  rng.randrange(1, 12), rng.randrange(1, 12), rng.randrange(3)]
        selected, pages, refs, released = [], Counter(), Counter(), set()
        expected = []
        for i in order:
            selected.append(i)
            pages.update(page_refs[i])
            refs.update(slot_refs[i])
            released = {slot for slot in refs if refs[slot] == slot_totals[slot]}
            free, decay = config[:2]
            decay_counts = Counter()
            for slot, _, frames, frame, _ in slots:
                if slot in released:
                    free += frames
                    if frame >= 0:
                        decay_counts[frame] += 1
            for frame, n in decay_counts.items():
                occupancy = dict(occupied)[frame]
                if n == occupancy:
                    free += 1
                    decay -= 4 - occupancy
                else:
                    decay += n
            free += sum(n == page_totals[frame] for frame, n in pages.items())
            required = (config[6] + (config[7] + config[8]) * config[2]
                        + (config[7] * config[4] + config[8]) * config[3]
                        + (max(0, config[7] - decay) + 3) // 4)
            if required <= free:
                expected = [selected, [v for pair in pages.items() for v in pair],
                            sorted(released)]
                break
        assert planning.plan_eviction(nodes, kv, slot_refs, slots, occupied, config) == expected


def test_native_protections_deduplicate_forced_and_batch_branches():
    prefix = torch.arange(64, dtype=torch.int32)
    inputs = [torch.cat((prefix, torch.full((32,), value, dtype=torch.int32)))
              for value in (100, 101, 102)]
    assert planning.branch_protections(inputs, [(2, 64), (0, 32), (1, 32)], 32) == [
        [32, 0, 1], [64, 0, 1, 2],
    ]
    with pytest.raises(RuntimeError, match="Invalid forced branch"):
        planning.branch_protections(inputs, [(0, 33)], 32)


def test_native_construction_rejects_invalid_entry_index():
    with pytest.raises(RuntimeError, match="entry index"):
        planning.construction_plan([(4, 0, 64, 0)], [(0, 10, 2)], [])
