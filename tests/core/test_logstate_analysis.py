import math

import pytest
import torch

from logssms.cache.logstate.online_analysis_scheduler import OnlineAnalysisScheduler
from logssms.cache.logstate.prefix_cache import LogRadixTree


def test_shared_prefix_keeps_statistics_after_a_partial_match():
    scheduler = OnlineAnalysisScheduler(32)
    tree = LogRadixTree(32, torch.device("cpu"))
    tree.analysis_factory = scheduler.make_hit_stats
    ids = torch.arange(256, dtype=torch.int32)
    node = tree.insert(ids, ids).node
    stats = node.analysis_owner
    scheduler.advance_hit_stats(stats, 256)
    branch = tree.insert(torch.cat((ids[:192], ids[192:] + 1000)), ids).node
    assert branch.analysis_owner is not stats
    assert tree.trace(ids[:192]).node.analysis_owner is stats
    scheduler.record_hit(stats, 128)
    assert stats.hits[-1][:2] == (128, 256)
    tree.check_integrity()


def test_exposure_counts_only_existing_distances_and_decays_old_queries():
    scheduler = OnlineAnalysisScheduler(32, hit_half_life=4)
    stats = scheduler.make_hit_stats(0)
    scheduler.advance_hit_stats(stats, 32)
    for _ in range(4):
        scheduler.record_hit(None, 0)
    scheduler.advance_hit_stats(stats, 64)
    for _ in range(4):
        scheduler.record_hit(None, 0)
    exposure = scheduler._bucket_exposures(stats)
    expected = [
        sum(2 ** (-(8 - q) / 4) for q in range(1, 9)),
        sum(2 ** (-(8 - q) / 4) for q in range(5, 9)),
    ]
    assert exposure == pytest.approx(expected)


def test_sparse_fit_has_finite_priors_and_no_exponent_floor():
    scheduler = OnlineAnalysisScheduler(32)
    for _ in range(100):
        scheduler.record_hit(None, 0)
    k, log_a = scheduler._fit_hit_bins({7: 1.0}, [100.0] * 8)
    assert 0 <= k < 2 and math.isfinite(log_a)
    k, log_a = scheduler._fit_hit_bins({}, [100.0] * 8)
    assert k == 2 and math.isfinite(log_a)


def test_model_refresh_is_lazy_and_misses_lower_the_amplitude():
    scheduler = OnlineAnalysisScheduler(32, hit_half_life=8)
    stats = scheduler.make_hit_stats(0)
    scheduler.advance_hit_stats(stats, 256)
    scheduler.record_hit(stats, 224)
    assert stats.log_amplitude is None
    scheduler.refresh_models([stats])
    before = stats.log_amplitude
    for _ in range(64):
        scheduler.record_hit(None, 0)
    scheduler.refresh_models([stats])
    assert stats.log_amplitude < before


def test_tail_hits_are_weighted_and_old_distances_are_frozen():
    scheduler = OnlineAnalysisScheduler(32, hit_half_life=8, min_sampling_len=128)
    stats = scheduler.make_hit_stats(0)
    scheduler.advance_hit_stats(stats, 256)
    scheduler.record_hit(stats, 256)
    scheduler.record_hit(stats, 224)
    scheduler.advance_hit_stats(stats, 288)
    assert [v[:2] for v in stats.hits] == [(256, 256), (224, 256)]
    scheduler.refresh_models([stats])
    point, = stats.past_tail_probabilities
    assert point[0] == 256
    assert point[1] == pytest.approx(2 ** (-1 / 8) / (5 + 2 ** (-1 / 8)))
    _, tail_prior = scheduler._prior()
    assert stats.tail_probability == pytest.approx(4 * tail_prior / (5 + 2 ** (-1 / 8)))
    scheduler.advance_hit_stats(stats, 544)
    for _ in range(64):
        scheduler.record_hit(None, 0)
    scheduler.refresh_models([stats])
    decayed, = stats.past_tail_probabilities
    assert decayed[0] == 256
    assert 0 < decayed[1] < point[1]


def test_hit_updates_and_prefill_analysis_leave_density_fitting_lazy():
    scheduler = OnlineAnalysisScheduler(32, checkpoint_log_count=2)
    tree = LogRadixTree(32, torch.device("cpu"))
    tree.analysis_factory = scheduler.make_hit_stats
    ids = torch.arange(256, dtype=torch.int32)
    node = tree.insert(ids, ids).node
    scheduler.advance_hit_stats(node.analysis_owner, 256)
    scheduler.record_hit(node.analysis_owner, 224)
    scheduler.analyze(tree, [ids])
    assert node.analysis_owner.log_amplitude is None


@pytest.mark.parametrize("observed_tail", [False, True])
def test_default_fallback_requires_zero_internal_observations(observed_tail):
    scheduler = OnlineAnalysisScheduler(32, min_sampling_len=64)
    tree = LogRadixTree(32, torch.device("cpu"))
    tree.analysis_factory = scheduler.make_hit_stats
    ids = torch.arange(64, dtype=torch.int32)
    leaf = tree.insert(ids, ids).node
    stats = leaf.analysis_owner
    scheduler.advance_hit_stats(stats, 64)
    def flag():
        models, _ = scheduler.eviction_models(tree.eviction_snapshot()[0])
        return models[-1][-1]
    assert flag() == 1
    if observed_tail:
        scheduler.record_hit(stats, 64)
        assert flag() == 1
    scheduler.record_hit(stats, 32)
    assert flag() == 0
    scheduler.advance_hit_stats(stats, 256)
    assert flag() == int(not observed_tail)
