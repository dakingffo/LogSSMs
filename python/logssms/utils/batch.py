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
    token_table_ids    : torch.Tensor | None = None

    @classmethod
    def make(cls, batch, device, page_size: int):
        reqs = batch.padded_reqs
        tables = [req.table_idx for req in reqs]
        cached = [req.cached_len for req in reqs]
        lengths = [req.device_len for req in reqs]
        extended = [length - start for length, start in zip(lengths, cached)]
        pages = [(length + page_size - 1) // page_size for length in lengths]
        cu_q = list(accumulate(extended, initial=0))
        cu_k = list(accumulate(pages, initial=0))
        counts = batch.logstate_replay_counts
        if counts is None:
            counts = [0] * len(reqs)
        assert len(counts) == len(reqs)
        write_tables = [] if batch.is_verify else [req.table_idx for req in batch.reqs]
        write_positions = [] if batch.is_verify else [
            req.device_len if req.can_decode else -1 for req in batch.reqs
        ]
        vectors = (tables, cached, [p - 1 for p in cached], lengths, cu_q, cu_k,
                   [(length - 1) % page_size + 1 for length in lengths],
                   counts, write_tables, write_positions)
        boundaries = [0]
        flat = []
        for vector in vectors:
            flat.extend(vector)
            boundaries.append(len(flat))
        host = torch.tensor(flat, dtype=torch.int32, pin_memory=device.type == "cuda")
        packed = host.to(device, non_blocking=True)
        cpu = tuple(host[a:b] for a, b in zip(boundaries, boundaries[1:]))
        gpu = tuple(packed[a:b] for a, b in zip(boundaries, boundaries[1:]))


        writes = packed[boundaries[8]:].to(torch.int64).view(2, len(write_tables))
        return cls(gpu[0], gpu[1], gpu[2], cpu[3], gpu[3], cpu[4], gpu[4],
                   cpu[5], gpu[5], cpu[6], gpu[7], (writes[0], writes[1]),
                   page_size, cu_q[-1], cu_k[-1], max(extended), max(pages))
