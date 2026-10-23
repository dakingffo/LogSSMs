from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch


@dataclass
class BatchIndices:
    table_ids          : torch.Tensor
    cached_lens        : torch.Tensor
    previous_positions : torch.Tensor
    seq_lens_cpu       : torch.Tensor
    seq_lens_gpu       : torch.Tensor
    cu_seqlens_q_cpu   : torch.Tensor
    cu_seqlens_q_gpu   : torch.Tensor
    cu_seqlens_k_cpu   : torch.Tensor
    cu_seqlens_k_gpu   : torch.Tensor
    last_page_len_cpu  : torch.Tensor
    replay_counts      : torch.Tensor
    write_tuple        : tuple[torch.Tensor, torch.Tensor]
    page_size          : int
    num_tokens         : int
    num_pages          : int
    max_extend_len     : int
    max_pages          : int
    sample_tuple       : tuple[torch.Tensor, torch.Tensor]
    prefill_padding    : int = 0
    padding_after     : int = 0
    token_table_ids    : torch.Tensor | None = None
    page_indices      : torch.Tensor | None = None

    @classmethod
    def make(cls, batch, device, page_size: int):
        return cls._make(batch, device, page_size)[0]

    @classmethod
    def make_mixed(cls, batch, device, page_size: int):
        return cls._make(batch, device, page_size, split=batch.prefill_count)

    @classmethod
    def _make(cls, batch, device, page_size: int, split: int | None = None):
        from logssms.core import ChunkedRequest

        reqs = batch.padded_reqs
        tables = [req.table_idx for req in reqs]
        cached = [req.cached_len for req in reqs]
        lengths = [req.device_len for req in reqs]
        extended = [length - start for length, start in zip(lengths, cached)]
        pages = [(length + page_size - 1) // page_size for length in lengths]
        cu_q = list(accumulate(extended, initial=0))
        cu_k = list(accumulate(pages, initial=0))
        padding_after = batch.prefill_count if batch.is_mixed else 0
        padding = max(0, batch.prefill_token_capacity - cu_q[padding_after]) if padding_after else 0
        counts = batch.cache_metadata.replay_counts if batch.cache_metadata is not None else None
        if counts is None:
            counts = [0] * len(reqs)
        assert len(counts) == len(reqs)
        groups = [(0, len(reqs))]
        if split is not None:
            groups += [(0, split), (split, len(reqs))]
        descriptions = []
        flat = []
        for begin, end in groups:
            selected = [i for i in range(begin, min(end, len(batch.reqs)))
                        if not batch.is_verify and not isinstance(reqs[i], ChunkedRequest)]
            q = [v - cu_q[begin] for v in cu_q[begin:end + 1]]
            k = [v - cu_k[begin] for v in cu_k[begin:end + 1]]
            vectors = (tables[begin:end], cached[begin:end],
                       [p - 1 for p in cached[begin:end]], lengths[begin:end], q, k,
                       [(length - 1) % page_size + 1 for length in lengths[begin:end]],
                       counts[begin:end], [tables[i] for i in selected],
                       [reqs[i].device_len if reqs[i].can_decode else -1 for i in selected],
                       [cu_q[i + 1] - cu_q[begin] - 1 + (padding if begin == 0 and i >= padding_after else 0)
                        for i in selected],
                       [i - begin for i in selected])
            boundaries = []
            for vector in vectors:
                flat.extend([0] * (-len(flat) % 4))
                start = len(flat)
                flat.extend(vector)
                boundaries.append((start, len(flat)))
            descriptions.append((begin, end, q, k, boundaries))
        host = torch.tensor(flat, dtype=torch.int32, pin_memory=device.type == "cuda")
        packed = host.to(device, non_blocking=True)
        write_packet = packed.to(torch.int64)
        packets = []
        for begin, end, q, k, boundaries in descriptions:
            cpu = tuple(host[a:b] for a, b in boundaries)
            gpu = tuple(packed[a:b] for a, b in boundaries)
            packets.append(cls(
                gpu[0], gpu[1], gpu[2], cpu[3], gpu[3], cpu[4], gpu[4],
                cpu[5], gpu[5], cpu[6], gpu[7],
                tuple(write_packet[a:b] for a, b in boundaries[8:10]),
                page_size, q[-1], k[-1], max(extended[begin:end]), max(pages[begin:end]),
                tuple(write_packet[a:b] for a, b in boundaries[10:12]),
                padding if begin == 0 and end == len(reqs) else 0,
                padding_after if begin == 0 and end == len(reqs) else 0,
            ))
        return tuple(packets)
