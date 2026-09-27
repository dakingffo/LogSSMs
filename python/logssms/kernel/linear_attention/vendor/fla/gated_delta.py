"""Inference-only Gated DeltaNet chunk forward derived from FLA 0.5.2."""

from __future__ import annotations

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_fwd import chunk_gated_delta_rule_fwd_intra
from .chunk_o import chunk_fwd_o
from .cumsum import chunk_local_cumsum
from .index import prepare_chunk_indices


_RCP_LN2 = 1.4426950216


@torch.compiler.disable
def chunk_gated_delta_rule(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens   : torch.Tensor,
    chunk_size   : int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    continuing: torch.Tensor | None = None,
    endings: torch.Tensor | None = None,
    ending_sequences: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, final_state, chunk_states, _ = _chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_size,
        chunk_indices,
        chunk_offsets,
        continuing,
        endings,
        ending_sequences,
    )
    return output, final_state, chunk_states


@torch.compiler.disable
def chunk_gated_delta_rule_with_log(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens   : torch.Tensor,
    chunk_size   : int = 64,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    continuing: torch.Tensor | None = None,
    endings: torch.Tensor | None = None,
    ending_sequences: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run packed GDN and also return every token's realized value update."""
    return _chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_size,
        chunk_indices,
        chunk_offsets,
        continuing,
        endings,
        ending_sequences,
    )


def _chunk_gated_delta_rule(
    query        : torch.Tensor,
    key          : torch.Tensor,
    value        : torch.Tensor,
    gate         : torch.Tensor,
    beta         : torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens   : torch.Tensor,
    chunk_size   : int,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    continuing: torch.Tensor | None,
    endings: torch.Tensor | None,
    ending_sequences: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run packed variable-length GDN and expose every post-chunk state.

    Inputs use ``[1, total_tokens, heads, dim]`` and ``cu_seqlens`` describes
    the packed requests.  ``chunk_states[i]`` is the recurrent state after the
    chunk represented by ``chunk_indices[i]``; this is the information the
    serving cache needs at page boundaries but upstream FLA keeps internal.
    """
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    assert chunk_size in (16, 32, 64)
    assert cu_seqlens.ndim == 1 and len(cu_seqlens) == len(initial_state) + 1

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
    hidden, value_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=key,
        w=w,
        u=u,
        g=cumulative_gate,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
    )
    assert value_new is not None and final_state is not None
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

    chunk_states = torch.empty_like(hidden[0])
    if continuing is None or endings is None or ending_sequences is None:
        sequence_ids = chunk_indices[:, 0].to(torch.int64)
        is_last = torch.ones(len(chunk_indices), dtype=torch.bool, device=query.device)
        is_last[:-1] = sequence_ids[:-1] != sequence_ids[1:]
        continuing = (~is_last[:-1]).nonzero().flatten()
        endings = is_last.nonzero().flatten()
        ending_sequences = sequence_ids.index_select(0, endings)
    chunk_states.index_copy_(0, continuing, hidden[0].index_select(0, continuing + 1))
    chunk_states.index_copy_(
        0,
        endings,
        final_state.index_select(0, ending_sequences).to(chunk_states),
    )
    # Keep every asynchronous kernel input alive until all dependent launches
    # above have been enqueued. In particular, the solved intra-chunk matrix is
    # otherwise released immediately after producing ``w`` and ``u``.
    del inverse
    return output, final_state, chunk_states, value_new


__all__ = ["chunk_gated_delta_rule", "chunk_gated_delta_rule_with_log"]
