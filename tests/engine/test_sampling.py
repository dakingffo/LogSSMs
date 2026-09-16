from types import SimpleNamespace

import pytest
import torch

from logssms.core import SamplingParams
from logssms.engine import Sampler


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _one_hot_logits(tokens: list[int], vocab_size: int) -> torch.Tensor:
    logits = torch.full((len(tokens), vocab_size), float("-inf"), device="cuda")
    logits[torch.arange(len(tokens), device="cuda"), tokens] = 0.0
    return logits


def test_probabilities_apply_per_request_filters() -> None:
    sampler = Sampler(torch.device("cuda"), vocab_size=5)
    params = [
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=1.0, top_k=2),
        SamplingParams(temperature=1.0, top_p=0.5),
    ]
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 8.0],
        ],
        device="cuda",
    )

    probs = sampler.probabilities(logits, sampler.prepare_params(params))

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3, device="cuda"))
    assert torch.count_nonzero(probs[0]).item() == 5
    assert torch.count_nonzero(probs[1]).item() == 2
    assert torch.count_nonzero(probs[2]).item() == 1


def test_reject_sample_groups_lengths_and_sampling_modes() -> None:
    vocab_size = 5
    sampler = Sampler(torch.device("cuda"), vocab_size)
    greedy = SamplingParams(temperature=0.0)
    sampling = SamplingParams(temperature=1.0)
    reqs = [
        SimpleNamespace(extend_len=3, sampling_params=greedy),
        SimpleNamespace(extend_len=3, sampling_params=sampling),
        SimpleNamespace(extend_len=5, sampling_params=sampling),
        SimpleNamespace(extend_len=3, sampling_params=sampling),
    ]
    batch = SimpleNamespace(
        reqs=reqs,
        size=len(reqs),
        draft_tokens=torch.tensor(
            [
                [1, 2, -1, -1],
                [2, 1, -1, -1],
                [0, 1, 2, 3],
                [3, 4, -1, -1],
            ],
            dtype=torch.int32,
            device="cuda",
        ),
    )
    draft_probs = torch.zeros(
        len(reqs), 4, vocab_size, dtype=torch.float32, device="cuda"
    )
    draft_probs[1, 0, 2] = 1.0
    draft_probs[1, 1, 1] = 1.0
    draft_probs[2, torch.arange(4, device="cuda"), torch.arange(4, device="cuda")] = 1.0
    draft_probs[3, 0, 3] = 1.0
    draft_probs[3, 1, 4] = 1.0
    batch.draft_probs = draft_probs

    logits = torch.cat(
        [
            _one_hot_logits([1, 3, 4], vocab_size),
            _one_hot_logits([2, 4, 0], vocab_size),
            _one_hot_logits([0, 1, 2, 3, 4], vocab_size),
            _one_hot_logits([3, 4, 1], vocab_size),
        ]
    )

    output = sampler.reject_sample(logits, batch)

    expected = torch.tensor(
        [
            [1, 3, -1, -1, -1],
            [2, 4, -1, -1, -1],
            [0, 1, 2, 3, 4],
            [3, 4, 1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(output, expected)
