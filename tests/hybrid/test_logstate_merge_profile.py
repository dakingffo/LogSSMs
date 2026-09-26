from types import SimpleNamespace

import pytest

from logssms.cache.logstate import LogStateCacheManager


@pytest.mark.parametrize(
    ("diagnostics", "coalesce", "released", "expected"),
    [
        ({"merge_anchors": 0}, (), 0, "no_match_anchor"),
        ({"merge_anchors": 1, "checkpoint_candidates": 0}, (), 0,
         "no_checkpoint_candidate"),
        ({"merge_anchors": 1, "checkpoint_candidates": 1,
          "candidate_groups": 0}, (), 0, "no_adjacent_candidate"),
        ({"merge_anchors": 1, "checkpoint_candidates": 2,
          "candidate_groups": 1}, (), 0, "candidates_rejected"),
        ({}, (object(),), 0, "commit_skipped"),
        ({}, (object(),), 1, "merged"),
    ],
)
def test_merge_batch_outcome(diagnostics, coalesce, released, expected) -> None:
    plan = SimpleNamespace(
        merge_diagnostics=tuple(diagnostics.items()),
        coalesce=coalesce,
    )
    assert LogStateCacheManager._merge_batch_outcome(plan, released) == expected
