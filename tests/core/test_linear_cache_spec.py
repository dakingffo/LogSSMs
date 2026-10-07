from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from logssms.cache.pools import linear_state_slot_bytes_for_config, make_linear_state_pool
from logssms.distributed import DistributedInfo
from logssms.models.config import LinearCacheSpec
from logssms.layers.linear_attention_backend import (
    BaseLinearAttentionBackend, SUPPORTED_LINEAR_ATTENTION_BACKENDS,
    make_linear_attention_backend,
)


def test_parallel_hybrid_storage_does_not_require_qwen_fields():
    # Both mixers may occur in every decoder, independently of total layer count.
    spec = LinearCacheSpec("other", 4, (2, 64, 32), (192, 3))
    model = SimpleNamespace(
        is_hybrid=True, num_attention_layers=4, num_linear_layers=4,
        make_linear_cache_spec=lambda tp_size: spec,
    )
    with patch("logssms.cache.pools.try_get_tp_info", return_value=DistributedInfo(0, 1)):
        assert linear_state_slot_bytes_for_config(model, torch.float16) == spec.slot_bytes(2)
        pool = make_linear_state_pool(model, 3, torch.device("cpu"), torch.float16)
    assert pool.recurrent_state.shape == (3, 4, 2, 64, 32)
    assert pool.conv_state.shape == (3, 4, 192, 3)


def test_backend_lifecycle_does_not_require_gdn_compute_arguments():
    class OtherBackend(BaseLinearAttentionBackend):
        def prepare_metadata(self, batch): pass
        def init_capture_graph(self, max_seq_len, bs_list): pass
        def prepare_for_capture(self, batch): pass
        def prepare_for_replay(self, batch): pass
        def forward(self, x, dt, B, C): return x

    with patch.dict(SUPPORTED_LINEAR_ATTENTION_BACKENDS._registry, {"other/radix": OtherBackend}):
        backend = make_linear_attention_backend("radix", recurrent_type="other")
    assert isinstance(backend, OtherBackend)
    assert backend.forward(123, None, None, None) == 123


def test_missing_operator_backend_is_not_silently_treated_as_gdn():
    with pytest.raises(KeyError, match="unknown/logstate"):
        make_linear_attention_backend("logstate", recurrent_type="unknown")
