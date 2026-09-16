from array import array

from datasets.parser import DatasetMessage, DatasetRecord
from experiment.prefix_cache_hit_rate import HitDensity, TokenRadixTree, _model_calls


def test_token_radix_tree_finds_real_match_endpoint():
    tree = TokenRadixTree()
    tree.insert(array("I", [1, 2, 3, 4, 5]))

    matched, tail_distance = tree.lookup(array("I", [1, 2, 9]))
    assert matched == 2
    assert tail_distance == 3

    density = HitDensity(page_sizes=(2,), bin_width=2)
    density.observe(matched, tail_distance)
    stats = density.stats[2]
    assert stats.hits == 1
    assert stats.branch_hits == 1
    assert stats.branch_bins == {1: 1}


def test_continuation_hit_has_zero_tail_distance():
    tree = TokenRadixTree()
    tree.insert(array("I", [1, 2, 3, 4]))

    matched, tail_distance = tree.lookup(array("I", [1, 2, 3, 4, 5]))
    density = HitDensity(page_sizes=(2,), bin_width=2)
    density.observe(matched, tail_distance)

    assert matched == 4
    assert density.stats[2].all_bins == {0: 1}
    assert density.stats[2].branch_hits == 0


def test_extending_a_leaf_moves_the_cache_tail():
    tree = TokenRadixTree()
    tree.insert(array("I", [1, 2]))
    tree.insert(array("I", [1, 2, 3, 4]))

    assert tree.lookup(array("I", [1, 2, 9])) == (2, 2)


def test_page_size_one_preserves_exact_distance():
    density = HitDensity(page_sizes=(1, 4), bin_width=1)
    density.observe(matched=5, tail_distance=3)

    assert density.stats[1].branch_bins == {3: 1}
    assert density.stats[4].branch_bins == {4: 1}


def test_model_calls_exclude_system_prompt():
    record = DatasetRecord(
        uid="session",
        category=None,
        messages=(
            DatasetMessage(role="system", content="shared instructions"),
            DatasetMessage(role="user", content="task"),
            DatasetMessage(role="assistant", content="answer"),
        ),
    )

    query, completed = next(_model_calls(record, None))
    assert [message.role for message in query] == ["user"]
    assert [message.role for message in completed] == ["user", "assistant"]
