import math

import pytest
import torch

from logssms.cache.logstate.hit_model import ReuseModel
from logssms.cache.logstate.online_analysis_scheduler import OnlineAnalysisScheduler
from logssms.cache.logstate.prefix_cache import LogRadixTree


def make_tree():
    model = ReuseModel(16, 8192, 1024, 16., False)
    tree = LogRadixTree(16, torch.device("cpu"))
    tree.analysis_factory = model.new_chain
    return model, tree


def publish(model, tree, ids, chain=None):
    node = tree.insert(ids, ids, analysis_owner=chain).node
    model.publish(node.analysis_owner, node.depth)
    return node


def test_lazy_split_hit_keeps_original_tail_and_targets():
    model, tree = make_tree()
    old = torch.arange(160, dtype=torch.int32)
    a = publish(model, tree, old, model.new_chain(0, "session")).analysis_owner
    model.record_query(tree, tree.trace(old[:64]), "session")
    prefix = old[:64]
    other = torch.cat((prefix, torch.arange(3_000, 3_032, dtype=torch.int32)))
    b = publish(model, tree, other, model.new_chain(64, "session")).analysis_owner
    assert tree.trace(old).node.analysis_owner is a
    assert tree.trace(prefix).node.analysis_owner is None

    model.record_query(tree, tree.trace(prefix), "session")
    model.refresh()
    assert a.hit_counts == {2: pytest.approx(.5 + model._weight(1))}
    assert b.hit_counts == {1: .5}


def test_branch_exposure_uses_the_same_fraction_and_counts_misses():
    model, tree = make_tree()
    prefix = torch.arange(64, dtype=torch.int32)
    old = torch.cat((prefix, torch.arange(64, 80, dtype=torch.int32)))
    a = publish(model, tree, old, model.new_chain(0, "session")).analysis_owner
    other = torch.cat((prefix, torch.arange(3_000, 3_016, dtype=torch.int32)))
    b = publish(model, tree, other, model.new_chain(64, "session")).analysis_owner
    model.record_query(tree, tree.trace(prefix), "session")
    model.record_query(tree, None, "session")
    model.refresh()
    expected = (1 + model._weight(1)) * 16 / 2
    assert a.exposure[0] == b.exposure[0] == pytest.approx(expected)
    assert a.hit_counts[0] == b.hit_counts[0] == pytest.approx(model._weight(1) / 2)


def test_fit_shares_exponent_and_keeps_distinct_amplitudes():
    model = ReuseModel(16, 8192, 1024, 16., False)
    a, b, c = (model.new_chain(0, key) for key in ("shared", "shared", "other"))
    for chain, multiplier, far in ((a, 1, 40), (b, 4, 40), (c, 1, 80)):
        model.publish(chain, 128)
        chain.exposure.update({0: 16000., 1: 32000.})
        chain.hit_counts.update({0: 80 * multiplier, 1: far * multiplier})
    model.refresh()
    assert a.exponent == b.exponent == pytest.approx(2., abs=.001)
    assert c.exponent == pytest.approx(1., abs=.1)
    assert math.exp(b.log_amplitude - a.log_amplitude) == pytest.approx(4., rel=.05)


def test_unobserved_owner_uses_same_prior_as_default():
    online = OnlineAnalysisScheduler(16)
    default = OnlineAnalysisScheduler(16, enable_merge_adjustment=False)
    _, tree = make_tree()
    ids = torch.arange(64, dtype=torch.int32)
    owner = online.make_hit_stats(0)
    node = publish(online.reuse, tree, ids, owner)
    snapshots, _ = tree.eviction_snapshot()
    assert node.analysis_owner is owner
    on_models = online.eviction_models(snapshots)
    def_models = default.eviction_models(snapshots)
    assert [row[1:] for row in on_models] == pytest.approx([row[1:] for row in def_models])
    on_active = online.active_density(owner, 64)
    def_active = default.active_density(None, 64)
    assert on_active.amplitude == pytest.approx(def_active.amplitude)
    assert (on_active.tail, on_active.granularity, on_active.bins,
            on_active.exponent, on_active.tail_probability, on_active.origin) == (
        def_active.tail, def_active.granularity, def_active.bins,
        def_active.exponent, def_active.tail_probability, def_active.origin,
    )


def test_tail_hit_accumulates_across_tail_growth():
    model, tree = make_tree()
    ids = torch.arange(128, dtype=torch.int32)
    chain = publish(model, tree, ids[:64]).analysis_owner
    model.record_query(tree, tree.trace(ids[:64]), chain.group)
    publish(model, tree, ids)
    model.record_query(tree, None, None)
    model.refresh()
    assert chain.tail_hits == pytest.approx(model._weight(1))
    assert chain.tail_probability > 0
    assert chain.log_amplitude is not None
    assert not chain.hit_counts
