import random

import pytest
import torch

from logssms.kernel import logstate as planning


def eviction_oracle(nodes):
    from collections import defaultdict

    rows = {row[0]: row for row in nodes}
    indices = {row[0]: i for i, row in enumerate(nodes)}
    children = defaultdict(set)
    for row in nodes:
        if row[1] in rows:
            children[row[1]].add(row[0])
    remaining = set(rows)
    order = []
    while True:
        leaves = [uuid for uuid in remaining if not children[uuid] and not rows[uuid][2]]
        if not leaves:
            return order
        uuid = min(leaves, key=lambda uuid: (rows[uuid][3], rows[uuid][4], uuid))
        order.append(indices[uuid])
        remaining.remove(uuid)
        children[rows[uuid][1]].discard(uuid)


def eviction_chains(count, length, checkpoint=False):
    nodes = [(0, -1, 1, 0, 0)]
    for chain in range(count):
        for position in range(length):
            uuid = len(nodes)
            nodes.append((uuid, 0 if position == 0 else uuid - 1, 0,
                          int(checkpoint), chain * 10 + position))
    return nodes


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


def test_native_materialize_respects_fragmented_and_protected_slots():
    path = [
        0, 0, 64, 1, 0, 1, 0, 0,
        1, 64, 128, -1, 64, 1, 0, 0,
        2, 128, 192, -1, 128, 1, 0, 1,
    ]
    assert planning.plan_materialize([path], [()], [(0, 1), (0, 2)], 64) == [[0]]
    # A pending branch inside the physical page protects the entire Log.
    assert planning.plan_materialize([path], [(96,)], [(0, 1)], 64) == []
    path[8 + 6] = 1
    assert planning.plan_materialize([path], [()], [(0, 1)], 64) == []


def test_materialize_eligibility_matches_serial_recovery_oracle():
    rng = random.Random(188)
    for _ in range(200):
        values = []
        position = 0
        for slot in range(rng.randint(1, 80)):
            position += rng.choice((0, 0, 32, 64))
            size = rng.choice((32, 64, 128))
            values.append((slot, position, position + size,
                           size // 64 if rng.random() < .25 else -1,
                           position, int(rng.random() > .1),
                           int(rng.random() < .1), int(rng.random() < .1)))
            position += size
        boundaries = sorted({rng.randrange(position + 1) for _ in range(4)})
        expected = []
        for i, entry in enumerate(values):
            _, begin, end, span, _, eligible, protected, fragmented = entry
            if (not eligible or span >= 0 or fragmented or (end - begin) % 64
                or any(begin < p < end for p in boundaries)
                or any(v[6] and begin <= v[1] and v[2] <= end for v in values)):
                continue
            cursor = max((v[2] for v in values if v[3] >= 0 and v[2] <= begin), default=0)
            for value in values:
                if value[3] >= 0 or value[2] <= cursor or value[1] > cursor:
                    continue
                cursor = min(end, value[2])
                if cursor == end:
                    break
            if cursor == end:
                expected.append([i])
        path = [v for entry in values for v in entry]
        actual = planning.plan_materialize([path], [boundaries], [(0, i) for i in range(len(values))], 64)
        assert actual == expected


def test_materialize_returns_bounded_metadata_for_long_log_chain():
    path = [v for i in range(128) for v in (i, i * 64, (i + 1) * 64, -1, i * 64, 1, 0, 0)]
    result = planning.plan_materialize([path], [()], [(0, i) for i in range(128)], 64)
    assert result == [[i] for i in range(128)]


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
    from collections import Counter

    rng = random.Random(381)
    for _ in range(200):
        count = rng.randrange(2, 40)
        parents = [-1] + [rng.randrange(i) for i in range(1, count)]
        locked = [1] + [int(rng.random() < .1) for _ in range(1, count)]
        for i in range(count - 1, 0, -1):
            locked[parents[i]] |= locked[i]
        nodes = [(i, parents[i], locked[i], 0, rng.randrange(100))
                 for i in range(count)]
        order = eviction_oracle(nodes)

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
                  rng.randrange(1, 12), rng.randrange(1, 12), rng.randrange(3), 2, 0]
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
                            sorted(released), []]
                break
        assert planning.plan_eviction(
            nodes, kv, slot_refs, slots, occupied, config, (),
        ) == expected


def assert_log_capacity_order(nodes, expected):
    """Each Log owns one frame; pressure must reclaim exactly the coldest safe prefix."""
    refs = [[uuid] if parent >= 0 else [] for uuid, parent, *_ in nodes]
    slots = [(slot, 1, 1, -1, -1) for row in refs for slot in row]
    for needed in range(1, len(expected) + 2):
        result = planning.plan_eviction(
            nodes, [[] for _ in nodes], refs, slots, [],
            [0, 0, 1, 0, 1, 1, needed, 0, 0, 1, 0], (),
        )
        if needed > len(expected):
            assert result == []
        else:
            selected, pages, released, compressed = result
            assert [nodes[index][0] for index in selected] == expected[:needed]
            assert released == sorted(expected[:needed])
            assert pages == [] and compressed == []


def test_log_pressure_globally_orders_cold_nodes_without_chain_rotation():
    nodes = eviction_chains(6, 3)
    assert_log_capacity_order(nodes, [
        3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10, 15, 14, 13, 18, 17, 16,
    ])


def test_eviction_uses_node_access_time_and_is_independent_of_row_order():
    nodes = eviction_chains(5, 2)
    uuid, parent, locked, kind, _ = nodes[1]
    nodes[1] = (uuid, parent, locked, kind, 999)
    expected = [2, 4, 3, 6, 5, 8, 7, 10, 9, 1]
    assert_log_capacity_order(nodes, expected)
    # Input row order is not a chain identity or priority.
    rng = random.Random(12)
    rng.shuffle(nodes)
    assert_log_capacity_order(nodes, expected)


def test_eviction_exposes_shared_parents_only_after_children_and_keeps_live_locks():
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 0, 1000),
             (2, 1, 0, 0, 1), (3, 2, 0, 0, 1),
             (4, 1, 0, 0, 2), (5, 4, 0, 0, 2),
             (6, 0, 1, 0, 0), (7, 6, 0, 0, 3)]
    assert_log_capacity_order(nodes, [3, 2, 5, 4, 7, 1])


def test_small_capacity_requests_select_the_globally_coldest_leaf():
    nodes = eviction_chains(4, 2)
    refs = [[]] + [[i] for i in range(1, len(nodes))]
    slots = [(i, 1, 1, -1, -1) for i in range(1, len(nodes))]
    result = planning.plan_eviction(
        nodes, [[] for _ in nodes], refs, slots, [], [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], (),
    )
    assert result == [[2], [], [2], []]


def test_checkpoint_pressure_thins_one_fixed_heat_snapshot():
    nodes = eviction_chains(10, 1, checkpoint=True)
    refs = [[]] + [[i] for i in range(1, len(nodes))]
    slots = [(i, 1, 1, -1, -1) for i in range(1, len(nodes))]
    actions = [(i, -1, nodes[i][4], i) for i in range(1, len(nodes))]
    order = [2, 4, 6, 8, 3, 7, 5, 1, 9, 10]
    for needed in (4, 8, 9, 10):
        result = planning.plan_eviction(
            nodes, [[] for _ in nodes], refs, slots, [],
            [0, 0, 1, 0, 1, 1, needed, 0, 0, 1, 2], actions,
        )
        assert result == [order[:needed], [], sorted(order[:needed]), []]


def test_eviction_prunes_logs_before_dropping_private_checkpoints():
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 1, 100),
             (2, 1, 0, 1, 1), (3, 0, 0, 0, 999)]
    refs = [[], [10], [11], [12]]
    slots = [(10, 1, 3, -1, -1), (11, 1, 3, -1, -1), (12, 1, 8, -1, -1)]
    actions = [(10, 11, 100, 1), (11, -1, 1, 2, 1)]
    config = [0, 0, 3, 0, 1, 1, 8, 0, 0, 3, 1]
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == [
        [3], [], [12], [],
    ]
    config[6] = 11
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == [
        [3, 2], [], [11, 12], [],
    ]
    config[6] = 14
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == [
        [3, 2, 1], [], [10, 11, 12], [],
    ]


def test_eviction_keeps_coldest_candidate_until_later_thinning_pass():
    nodes = [(0, -1, 1, 0, 0), (1, 0, 0, 1, 999),
             (2, 1, 0, 1, 1), (3, 2, 0, 1, 0)]
    refs = [[], [10], [11], [12]]
    slots = [(slot, 1, 3, -1, -1) for slot in (10, 11, 12)]
    actions = [(11, 12, 1, 2, 1), (10, 11, 999, 1), (12, -1, 0, 3, 2, 1)]
    config = [0, 0, 3, 0, 1, 1, 3, 0, 0, 3, 1]
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == [
        [], [], [11], [11, 12],
    ]
    config[6] = 6
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == [
        [3, 2], [], [11, 12], [11, 12],
    ]
    config[6] = 100
    assert planning.plan_eviction(nodes, [[], [], [], []], refs, slots, [], config, actions) == []


def test_native_protections_deduplicate_forced_and_batch_branches():
    prefix = torch.arange(64, dtype=torch.int32)
    inputs = [torch.cat((prefix, torch.full((32,), value, dtype=torch.int32)))
              for value in (100, 101, 102)]
    assert planning.branch_protections(inputs, [(2, 64), (0, 32), (1, 32)], 32) == [
        [32, 0, 1], [64, 0, 1, 2],
    ]
    with pytest.raises(RuntimeError, match="Invalid forced branch"):
        planning.branch_protections(inputs, [(0, 33)], 32)
