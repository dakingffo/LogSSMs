from __future__ import annotations

import torch

from picosgl.utils import init_logger

from .pool import DraftKVPool

logger = init_logger(__name__)


class DraftAttentionBackend:
    """One-query paged attention over ``DraftKVPool``.

    The FlashInfer adapter deliberately consumes only Q, physical page indices and the
    drafter pool. It does not construct a fake target ``Batch`` or touch global ``Context``.
    Non-FlashInfer configurations retain the numerically equivalent eager implementation.
    """

    def __init__(
        self,
        backend_name: str,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim    : int,
        dtype       : torch.dtype,
        device      : torch.device,
    ) -> None:
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.dtype        = dtype
        self.device       = device
        self.scaling      = head_dim**-0.5
        self.wrapper      = None
        self.block_wrapper = None
        self.last_event   = None
        self._plan_inputs = None

        decode_backend = backend_name.rsplit(",", 1)[-1]
        if device.type == "cuda" and decode_backend in ("auto", "fi"):
            try:
                from flashinfer import (
                    BatchDecodeWithPagedKVCacheWrapper,
                    BatchPrefillWithPagedKVCacheWrapper,
                )

                workspace = torch.empty(
                    32 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                self.wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    use_tensor_cores=num_qo_heads // num_kv_heads >= 4,
                    kv_layout="NHD",
                    backend="fa2",
                )
                self.block_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    backend="fa2",
                )
                self.workspace = workspace
                self.last_event = torch.cuda.Event()
                self.last_event.record()
            except ImportError:
                logger.warning(
                    "FlashInfer unavailable for drafter; using eager pooled attention."
                )

    def forward(
        self,
        query      : torch.Tensor,
        pool       : DraftKVPool,
        indices    : torch.Tensor,
        valid_mask : torch.Tensor,
        cache_lens : list[int],
    ) -> torch.Tensor:
        if self.wrapper is None:
            return self._forward_eager(query, pool, indices, valid_mask)
        return self._forward_flashinfer(query, pool, indices, cache_lens)

    def _forward_flashinfer(
        self,
        query     : torch.Tensor,
        pool      : DraftKVPool,
        indices   : torch.Tensor,
        cache_lens: list[int],
    ) -> torch.Tensor:
        assert self.wrapper is not None and self.last_event is not None
        batch_size = len(cache_lens)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        seq_lens = torch.tensor(cache_lens, **cpu_kwargs)
        indptr = torch.tensor([0] + cache_lens, **cpu_kwargs).cumsum_(dim=0)
        last_page_len = torch.ones(batch_size, **cpu_kwargs)
        flat_indices = torch.cat(
            [indices[i, :length] for i, length in enumerate(cache_lens)]
        ).contiguous()

        # FlashInfer reuses pinned staging storage internally.  Do not let the next plan
        # mutate it before this plan's asynchronous H2D transfer has completed.
        self.last_event.synchronize()
        self._plan_inputs = (seq_lens, indptr, last_page_len, flat_indices)
        self.wrapper.plan(
            indptr=indptr,
            indices=flat_indices,
            last_page_len=last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens=seq_lens,
            data_type=self.dtype,
            q_data_type=query.dtype,
            kv_data_type=pool.dtype,
            non_blocking=True,
        )
        self.last_event.record()
        k_cache = pool.k.view(pool.num_slots, 1, pool.num_kv_heads, pool.head_dim)
        v_cache = pool.v.view(pool.num_slots, 1, pool.num_kv_heads, pool.head_dim)
        return self.wrapper.run(q=query, paged_kv_cache=(k_cache, v_cache))

    def _forward_eager(
        self,
        query     : torch.Tensor,
        pool      : DraftKVPool,
        indices   : torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        key   = pool.k[indices.to(torch.int64)]
        value = pool.v[indices.to(torch.int64)]
        n_rep = self.num_qo_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=2)
            value = value.repeat_interleave(n_rep, dim=2)

        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = torch.matmul(query.unsqueeze(2), key.transpose(-1, -2)).squeeze(2)
        attn = attn * self.scaling
        attn = attn.masked_fill(~valid_mask[:, None, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn.unsqueeze(2), value).squeeze(2)

    def forward_block(
        self,
        query     : torch.Tensor,
        pool      : DraftKVPool,
        indices   : torch.Tensor,
        valid_mask: torch.Tensor,
        cache_lens: list[int],
    ) -> torch.Tensor:
        """Non-causal attention for a parallel DFlash block.

        Every query in a block sees the complete committed context and every noisy token
        in the current block, matching DFlash's bidirectional block attention.
        """
        assert query.ndim == 4
        if self.block_wrapper is None:
            return self._forward_block_eager(query, pool, indices, valid_mask)
        return self._forward_block_flashinfer(query, pool, indices, cache_lens)

    def _forward_block_flashinfer(
        self,
        query     : torch.Tensor,
        pool      : DraftKVPool,
        indices   : torch.Tensor,
        cache_lens: list[int],
    ) -> torch.Tensor:
        assert self.block_wrapper is not None and self.last_event is not None
        batch_size, block_size = query.shape[:2]
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        qo_indptr = torch.arange(
            0, (batch_size + 1) * block_size, block_size, **cpu_kwargs
        )
        kv_indptr = torch.tensor([0] + cache_lens, **cpu_kwargs).cumsum_(dim=0)
        last_page_len = torch.ones(batch_size, **cpu_kwargs)
        flat_indices = torch.cat(
            [indices[i, :length] for i, length in enumerate(cache_lens)]
        ).contiguous()

        self.last_event.synchronize()
        self._plan_inputs = (qo_indptr, kv_indptr, last_page_len, flat_indices)
        self.block_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=flat_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            causal=False,
            pos_encoding_mode="NONE",
            q_data_type=query.dtype,
            kv_data_type=pool.dtype,
            non_blocking=True,
        )
        self.last_event.record()
        k_cache = pool.k.view(pool.num_slots, 1, pool.num_kv_heads, pool.head_dim)
        v_cache = pool.v.view(pool.num_slots, 1, pool.num_kv_heads, pool.head_dim)
        output = self.block_wrapper.run(
            q=query.flatten(0, 1),
            paged_kv_cache=(k_cache, v_cache),
        )
        return output.view_as(query)

    def _forward_block_eager(
        self,
        query     : torch.Tensor,
        pool      : DraftKVPool,
        indices   : torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        indices = indices.masked_fill(~valid_mask, 0).to(torch.int64)
        key   = pool.k[indices]
        value = pool.v[indices]
        n_rep = self.num_qo_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=2)
            value = value.repeat_interleave(n_rep, dim=2)

        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        attn = attn.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, value).transpose(1, 2)
