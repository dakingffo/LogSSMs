from types import SimpleNamespace

import pytest
import torch

from logssms.engine.engine import Engine


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_prepare_profile(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    engine = object.__new__(Engine)
    engine._profile_prepare_enabled = enabled
    engine._prepare_profile_totals = {}
    engine._prepare_profile_counts = {}
    engine._prepare_profile_prefill_samples = []
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=False))
    engine.graph_runner = SimpleNamespace(pad_batch=lambda batch: None)
    engine.attn_backend = SimpleNamespace(prepare_metadata=lambda batch: None)
    engine.sampler = SimpleNamespace(prepare=lambda batch: "sample args")
    engine.page_table = torch.zeros((1, 1), dtype=torch.int32)
    mapping = (torch.tensor([0]), torch.tensor([0]))
    monkeypatch.setattr(engine, "_make_positions", lambda batch, device: mapping[0])
    monkeypatch.setattr(engine, "_make_input_tuple", lambda batch, device: mapping)
    monkeypatch.setattr(engine, "_make_write_tuple", lambda batch, device: mapping)
    allocation_phases = []
    cache_manager = SimpleNamespace(
        allocate=lambda reqs, *, phase=None: allocation_phases.append(phase),
        prepare_batch=lambda batch: None,
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(extend_len=3)],
        phase="prefill",
        is_prefill=True,
        size=1,
    )

    result = engine.prepare_batch(batch, cache_manager)

    assert result.sample_args == "sample args"
    assert allocation_phases == ["prefill"]
    profile = engine.prepare_profile_result()
    if not enabled:
        assert profile is None
        return
    assert profile is not None
    sample = profile["prefill_batches"][0]
    assert sample["logical_tokens"] == 3
    assert profile["phases"]["prefill"]["batches"] == 1
    stages = (
        "pad_batch_ms", "allocate_ms", "positions_mappings_ms",
        "cache_prepare_ms", "attention_metadata_ms", "linear_metadata_ms",
        "sampler_prepare_ms",
    )
    assert all(sample[stage] >= 0 for stage in stages)
    assert sample["total_ms"] == pytest.approx(sum(sample[stage] for stage in stages))
