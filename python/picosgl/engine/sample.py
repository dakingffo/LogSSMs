from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from picosgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from picosgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k       : torch.Tensor | None = None
    top_k_rows  : torch.Tensor | None = None
    top_p       : torch.Tensor | None = None
    top_p_rows  : torch.Tensor | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def normalize_probs(
    logits      : torch.Tensor,
    temperatures: torch.Tensor,
    top_k       : torch.Tensor | None,
    top_k_rows  : torch.Tensor | None,
    top_p       : torch.Tensor | None,
    top_p_rows  : torch.Tensor | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is not None:
        assert top_k_rows is not None
        filtered = sampling.top_k_renorm_probs(
            probs.index_select(0, top_k_rows), top_k
        )
        probs.index_copy_(0, top_k_rows, filtered)
    if top_p is not None:
        assert top_p_rows is not None
        filtered = sampling.top_p_renorm_probs(
            probs.index_select(0, top_p_rows), top_p
        )
        probs.index_copy_(0, top_p_rows, filtered)
    return probs


def sample_impl(
    logits: torch.Tensor,
    args  : BatchSamplingArgs,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    assert args.temperatures is not None
    probs = normalize_probs(
        logits,
        args.temperatures,
        args.top_k,
        args.top_k_rows,
        args.top_p,
        args.top_p_rows,
    )
    return sampling.sampling_from_probs(probs)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        return self.prepare_params([r.sampling_params for r in batch.reqs])

    def prepare_params(self, params: list) -> BatchSamplingArgs:
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_k_rows = [i for i, p in enumerate(params) if 1 <= p.top_k < self.vocab_size]
        top_p_rows = [i for i, p in enumerate(params) if p.top_p < 1.0]
        temperatures = make_device_tensor(ts, torch.float32, self.device)

        return BatchSamplingArgs(
            temperatures,
            top_k=(
                make_device_tensor(
                    [params[i].top_k for i in top_k_rows], torch.int32, self.device
                ) if top_k_rows else None
            ),
            top_k_rows=(
                make_device_tensor(top_k_rows, torch.int64, self.device)
                if top_k_rows else None
            ),
            top_p=(
                make_device_tensor(
                    [max(params[i].top_p, MIN_P) for i in top_p_rows],
                    torch.float32,
                    self.device,
                ) if top_p_rows else None
            ),
            top_p_rows=(
                make_device_tensor(top_p_rows, torch.int64, self.device)
                if top_p_rows else None
            ),
        )

    def probabilities(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        assert args.temperatures is not None
        return normalize_probs(
            logits.float(),
            args.temperatures,
            args.top_k,
            args.top_k_rows,
            args.top_p,
            args.top_p_rows,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.is_greedy:  # greedy sampling
            return torch.argmax(logits, dim=-1)
        else:
            return sample_impl(logits.float(), args)

    def _greedy_reject(
        self,
        target_logits: torch.Tensor,
        drafts       : torch.Tensor,
    ) -> torch.Tensor:
        target_tokens = target_logits.argmax(dim=-1).to(torch.int32)
        num_drafts = drafts.shape[1]
        if num_drafts == 0:
            return target_tokens[:, :1]

        matches = target_tokens[:, :num_drafts] == drafts
        accepted = matches.to(torch.int64).cumprod(dim=1).sum(dim=1)
        positions = torch.arange(num_drafts, device=self.device)
        output = torch.where(
            positions[None, :] < accepted[:, None],
            drafts,
            -1,
        )
        output = torch.cat(
            [output, output.new_full((output.shape[0], 1), -1)],
            dim=1,
        )
        bonus = target_tokens.gather(1, accepted[:, None])
        output.scatter_(1, accepted[:, None], bonus)
        return output

    def _sampling_reject(
        self,
        target_logits: torch.Tensor,
        drafts       : torch.Tensor,
        draft_probs  : torch.Tensor | None,
        params       : list,
    ) -> torch.Tensor:
        import flashinfer.sampling as sampling

        batch_size, seq_len, _ = target_logits.shape
        args = self.prepare_params([
            param for param in params for _ in range(seq_len)
        ])
        target_probs = self.probabilities(
            target_logits.flatten(0, 1), args
        ).view(batch_size, seq_len, self.vocab_size)
        if drafts.shape[1] == 0:
            return sampling.sampling_from_probs(target_probs[:, 0]).unsqueeze(1)

        assert draft_probs is not None, "sampling verify batch needs draft_probs"
        output, _, _ = sampling.chain_speculative_sampling(
            draft_probs,
            drafts,
            target_probs,
        )
        return output

    @nvtx_annotate("Sampler")
    def reject_sample(self, logits, batch) -> torch.Tensor:
        """Return accepted drafts followed by one target bonus token and -1 padding."""
        bs = batch.size
        K = 0 if batch.draft_tokens is None else batch.draft_tokens.shape[1]
        extend_token = torch.full((bs, K + 1), -1, dtype=torch.int32, device=self.device)
        if K == 0:
            return extend_token

        groups: dict[tuple[bool, int], list[tuple[int, int]]] = {}
        offset = 0
        for batch_idx, req in enumerate(batch.reqs):
            num_drafts = req.extend_len - 1
            groups.setdefault((req.sampling_params.is_greedy, num_drafts), []).append(
                (batch_idx, offset)
            )
            offset += num_drafts + 1
        assert offset == logits.shape[0]

        for (is_greedy, num_drafts), entries in groups.items():
            seq_len = num_drafts + 1
            batch_indices = torch.tensor(
                [batch_idx for batch_idx, _offset in entries],
                dtype=torch.int64,
                device=self.device,
            )
            row_indices = torch.tensor(
                [row for _batch_idx, begin in entries
                 for row in range(begin, begin + seq_len)],
                dtype=torch.int64,
                device=self.device,
            )
            target_logits = logits.index_select(0, row_indices).view(
                len(entries), seq_len, self.vocab_size
            )
            drafts = batch.draft_tokens.index_select(
                0, batch_indices
            )[:, :num_drafts].contiguous()

            if is_greedy:
                output = self._greedy_reject(target_logits, drafts)
            else:
                params = [
                    batch.reqs[batch_idx].sampling_params
                    for batch_idx, _offset in entries
                ]
                draft_probs = (
                    batch.draft_probs.index_select(
                        0, batch_indices
                    )[:, :num_drafts].contiguous()
                    if batch.draft_probs is not None else None
                )
                output = self._sampling_reject(
                    target_logits,
                    drafts,
                    draft_probs,
                    params,
                )
            extend_token[batch_indices, :seq_len] = output

        return extend_token
