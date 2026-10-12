from __future__ import annotations

import torch

from .chunk_delta_h import ChunkLogIO, ChunkStateIO, chunk_gated_delta_rule_fwd_h
from .chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .chunk_o import chunk_fwd_o
from .cumsum import chunk_local_cumsum
from .index import prepare_chunk_indices


_RCP_LN2 = 1.4426950216


@torch.compiler.disable
def chunk_gated_delta_rule_to_cache(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    state_io     : ChunkStateIO,
    cu_seqlens   : torch.Tensor,
    chunk_size   : int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    *,
    log_io: ChunkLogIO | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    assert chunk_size in (16, 32, 64)
    assert cu_seqlens.ndim == 1
    assert len(cu_seqlens) == len(state_io.read_slots) + 1

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    cumulative_gate = chunk_local_cumsum(
        gate,
        chunk_size=chunk_size,
        scale=_RCP_LN2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w, u, inverse = chunk_gated_delta_rule_fwd_intra(
        k=key,
        v=value,
        g=cumulative_gate,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    hidden, value_new = chunk_gated_delta_rule_fwd_h(
        k=key,
        w=w,
        u=u,
        g=cumulative_gate,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
        state_io=state_io,
        log_io=log_io,
    )
    assert value_new is not None
    output = chunk_fwd_o(
        q=query,
        k=key,
        v=value_new,
        h=hidden,
        g=cumulative_gate,
        scale=query.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


    del inverse
    return output, value_new


__all__ = ["ChunkLogIO", "ChunkStateIO", "chunk_gated_delta_rule_to_cache"]
