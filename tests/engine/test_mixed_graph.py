from logssms.core import Batch
from logssms.engine.mixed_graph import MixedGraphRunner


def test_mixed_graph_eligibility_before_padding_uses_next_bucket():
    runner = object.__new__(MixedGraphRunner)
    runner.graphs = {2: None, 4: None}
    batch = Batch([None, None, None], "mixed")

    assert not hasattr(batch, "padded_reqs")
    assert runner.can_use_cuda_graph(batch)
