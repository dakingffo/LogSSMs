from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.utils import torch_dtype
from logssms.engine import Sampler
from logssms.models.drafters import Qwen3_5MTPDrafter

from ...base import EngineBase
from ..attention import DraftAttentionBackend
from ..pool import DraftKVPool
from .config import MTPSpeculatorConfig
from .state import MTPState

if TYPE_CHECKING:
    from logssms.engine.config import EngineConfig
    from logssms.scheduler.config import SchedulerConfig


class MTPEngine(EngineBase):

    def __init__(
        self,
        drafter          : Qwen3_5MTPDrafter,
        device           : torch.device,
        vocab_size       : int,
        num_spec_tokens  : int,
        max_running_req  : int,
        window_size      : int,
        max_batch_size   : int,
        attention_backend: str,
    ) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.num_spec_tokens = num_spec_tokens
        self.vocab_size = vocab_size
        self.drafter = drafter
        self.sampler = Sampler(device, vocab_size)

        attention = self.drafter.layers.op_list[0].self_attn
        self.pool = DraftKVPool(
            max_running_req=max_running_req,
            window_size=window_size,
            max_batch_size=max_batch_size,
            num_spec_tokens=num_spec_tokens,
            num_kv_heads=attention.num_kv_heads,
            head_dim=attention.head_dim,
            dtype=attention.qkv_proj.weight.dtype,
            device=device,
        )
        self.attention_backend = DraftAttentionBackend(
            backend_name=attention_backend,
            num_qo_heads=attention.num_qo_heads,
            num_kv_heads=attention.num_kv_heads,
            head_dim=attention.head_dim,
            dtype=attention.qkv_proj.weight.dtype,
            device=device,
        )

    @classmethod
    def from_config(
        cls,
        device: torch.device,
        config: SchedulerConfig,
    ) -> MTPEngine:
        drafter = cls.load_drafter(device, config)
        speculator_config = config.speculator_config
        assert isinstance(speculator_config, MTPSpeculatorConfig)
        K = speculator_config.num_draft_tokens
        return cls(
            drafter,
            device,
            config.model_config.vocab_size,
            K,
            config.max_running_req,
            speculator_config.window_size,
            min(config.max_running_req, config.decode_batch_budget // K),
            config.attention_backend,
        )

    @staticmethod
    def load_drafter(
        device: torch.device, config: EngineConfig
    ) -> Qwen3_5MTPDrafter:
        from logssms.layers import set_rope_device

        mc = config.model_config
        set_rope_device(device)
        with torch.device("meta"), torch_dtype(config.dtype):
            drafter = Qwen3_5MTPDrafter(mc)
        model_path = config.speculative_draft_model_path
        assert model_path is not None
        drafter.load_weights(model_path, device)
        return drafter

    def draft(self, states: list[MTPState]) -> None:
        K = self.num_spec_tokens
        for st in states:
            assert 0 <= st.n_drafts <= K
            st.draft_tokens = []
            st.draft_probs = (
                torch.zeros(K, self.vocab_size, dtype=torch.float32, device=self.device)
                if not st.sampling_params.is_greedy else None
            )

        active_states = [st for st in states if st.n_drafts > 0]
        if not active_states:
            return

        batch_size = len(active_states)
        assert batch_size <= self.pool.max_batch_size, (
            f"MTP draft batch {batch_size} exceeds scratch capacity "
            f"{self.pool.max_batch_size}"
        )
        table_indices = [st.table_idx for st in active_states]
        assert len(set(table_indices)) == batch_size
        batch_rows = torch.arange(batch_size, dtype=torch.int64, device=self.device)

        pending_counts = []
        for st in active_states:
            if not st.cache_initialized:
                self.pool.reset(st.table_idx)
                st.cache_initialized = True
            count = len(st.pending_tokens)
            assert count > 0, "every MTP round must carry at least one Target-hidden row"
            assert count == len(st.pending_positions) == st.pending_hidden.shape[0]
            pending_counts.append(count)

        input_ids = torch.tensor(
            [token for st in active_states for token in st.pending_tokens],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [position for st in active_states for position in st.pending_positions],
            dtype=torch.int64,
            device=self.device,
        )
        hidden = torch.cat([st.pending_hidden for st in active_states], dim=0)
        residual, query, gate, key, value = self.drafter.prepare_cache_rows(
            input_ids, positions, hidden
        )

        slots = self.pool.append_persistent_batch(table_indices, pending_counts)
        self.pool.store(slots, key, value)

        last_rows = []
        offset = 0
        for st, count in zip(active_states, pending_counts):
            last_rows.append(offset + count - 1)
            offset += count
            st.clear_pending()

        last_rows = torch.tensor(last_rows, dtype=torch.int64, device=self.device)
        cache_indices, cache_valid = self.pool.batch_indices(
            table_indices, batch_rows, scratch_depth=0
        )
        attention_output = self.attention_backend.forward(
            query.index_select(0, last_rows),
            self.pool,
            cache_indices,
            cache_valid,
            self.pool.cache_lengths(table_indices),
        )
        logits, mtp_hidden = self.drafter.finish_cache_rows(
            residual.index_select(0, last_rows),
            gate.index_select(0, last_rows),
            attention_output,
        )
        next_tokens = self._sample_batch(logits, active_states, 0)
        token_matrix = torch.full(
            (batch_size, K), -1, dtype=torch.int32, device=self.device
        )
        token_matrix[:, 0] = next_tokens

        first_positions = torch.tensor(
            [st.window_positions[-1] for st in active_states],
            dtype=torch.int64,
            device=self.device,
        )
        for j in range(1, max(st.n_drafts for st in active_states)):
            active_indices = [i for i, st in enumerate(active_states) if st.n_drafts > j]
            active_rows = torch.tensor(active_indices, dtype=torch.int64, device=self.device)
            step_states = [active_states[i] for i in active_indices]
            step_batch_rows = batch_rows.index_select(0, active_rows)
            step_ids = token_matrix[active_rows, j - 1]
            step_positions = first_positions.index_select(0, active_rows) + j
            step_hidden = mtp_hidden.index_select(0, active_rows)

            residual, query, gate, key, value = self.drafter.prepare_cache_rows(
                step_ids, step_positions, step_hidden
            )
            slots = self.pool.scratch_slots(step_batch_rows, j - 1)
            self.pool.store(slots, key, value)
            step_table_indices = [st.table_idx for st in step_states]
            cache_indices, cache_valid = self.pool.batch_indices(
                step_table_indices, step_batch_rows, scratch_depth=j
            )
            attention_output = self.attention_backend.forward(
                query,
                self.pool,
                cache_indices,
                cache_valid,
                self.pool.cache_lengths(step_table_indices, scratch_depth=j),
            )
            logits, step_hidden = self.drafter.finish_cache_rows(
                residual, gate, attention_output
            )
            next_tokens = self._sample_batch(logits, step_states, j)
            token_matrix[active_rows, j] = next_tokens
            mtp_hidden[active_rows] = step_hidden

        token_rows = token_matrix.tolist()
        for st, row in zip(active_states, token_rows):
            st.draft_tokens = row[: st.n_drafts]

    def _sample_batch(
        self,
        logits: torch.Tensor,
        states: list[MTPState],
        step: int,
    ) -> torch.Tensor:
        tokens = logits.argmax(dim=-1).to(torch.int32)
        sampling_rows = [
            i for i, st in enumerate(states)
            if step < st.n_drafts and not st.sampling_params.is_greedy
        ]
        if not sampling_rows:
            return tokens

        rows = torch.tensor(sampling_rows, dtype=torch.int64, device=self.device)
        params = [states[i].sampling_params for i in sampling_rows]
        args = self.sampler.prepare_params(params)
        probs = self.sampler.probabilities(logits.index_select(0, rows), args)

        import flashinfer.sampling as sampling

        sampled = sampling.sampling_from_probs(probs).to(torch.int32)
        tokens.index_copy_(0, rows, sampled)
        for prob, i in zip(probs, sampling_rows):
            st = states[i]
            assert st.draft_probs is not None
            st.draft_probs[step].copy_(prob)
        return tokens
