from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from logssms.engine import Sampler
from logssms.models.drafters import Eagle3Drafter
from logssms.utils import load_model_config, torch_dtype

from ...base import EngineBase
from ..attention import DraftAttentionBackend
from ..pool import DraftKVPool
from .config import Eagle3SpeculatorConfig
from .state import Eagle3State

if TYPE_CHECKING:
    from logssms.engine.config import EngineConfig
    from logssms.scheduler.config import SchedulerConfig


class Eagle3Engine(EngineBase):
    def __init__(
        self,
        drafter          : Eagle3Drafter,
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
        self.sampler = Sampler(device, drafter.draft_vocab_size)

        attention = drafter.midlayer.self_attn
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
    ) -> Eagle3Engine:
        drafter = cls.load_drafter(device, config)
        speculator_config = config.speculator_config
        assert isinstance(speculator_config, Eagle3SpeculatorConfig)
        K = speculator_config.num_draft_tokens
        return cls(
            drafter=drafter,
            device=device,
            vocab_size=config.model_config.vocab_size,
            num_spec_tokens=K,
            max_running_req=config.max_running_req,
            window_size=speculator_config.window_size,
            max_batch_size=min(
                config.max_running_req,
                config.decode_batch_budget // K,
            ),
            attention_backend=config.attention_backend,
        )

    @staticmethod
    def load_drafter(
        device: torch.device, config: EngineConfig
    ) -> Eagle3Drafter:
        from logssms.layers import set_rope_device

        model_path = config.speculative_draft_model_path
        assert model_path is not None
        draft_config = load_model_config(model_path)
        set_rope_device(device)
        with torch.device("meta"), torch_dtype(config.dtype):
            drafter = Eagle3Drafter(
                hidden_size=draft_config.hidden_size,
                head_dim=draft_config.head_dim,
                num_qo_heads=draft_config.num_attention_heads,
                num_kv_heads=draft_config.num_key_value_heads,
                intermediate_size=draft_config.intermediate_size,
                hidden_act=draft_config.hidden_act,
                rms_norm_eps=draft_config.rms_norm_eps,
                max_position=draft_config.max_position_embeddings,
                rope_base=draft_config.rope_theta,
                vocab_size=draft_config.vocab_size,
                draft_vocab_size=draft_config.draft_vocab_size,
            )
        drafter.load_weights(model_path, device)
        drafter.load_target_embedding(config.model_path, device)
        return drafter

    def draft(self, states: list[Eagle3State]) -> None:
        K = self.num_spec_tokens
        for state in states:
            assert 0 <= state.n_drafts <= K
            state.draft_tokens = []
            state.draft_probs = (
                torch.zeros(
                    K,
                    self.vocab_size,
                    dtype=torch.float32,
                    device=self.device,
                )
                if not state.sampling_params.is_greedy else None
            )

        active_states = [state for state in states if state.n_drafts > 0]
        if not active_states:
            return

        batch_size = len(active_states)
        assert batch_size <= self.pool.max_batch_size
        table_indices = [state.table_idx for state in active_states]
        assert len(set(table_indices)) == batch_size
        batch_rows = torch.arange(batch_size, dtype=torch.int64, device=self.device)

        pending_counts = []
        for state in active_states:
            if not state.cache_initialized:
                self.pool.reset(state.table_idx)
                state.cache_initialized = True
            count = len(state.pending_tokens)
            assert count > 0
            assert count == len(state.pending_positions) == state.pending_hidden.shape[0]
            pending_counts.append(count)

        input_ids = torch.tensor(
            [token for state in active_states for token in state.pending_tokens],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [position for state in active_states for position in state.pending_positions],
            dtype=torch.int64,
            device=self.device,
        )
        hidden = torch.cat([state.pending_hidden for state in active_states], dim=0)
        residual, query, key, value = self.drafter.prepare_cache_rows(
            input_ids, positions, hidden
        )
        slots = self.pool.append_persistent_batch(table_indices, pending_counts)
        self.pool.store(slots, key, value)

        last_rows = []
        offset = 0
        for state, count in zip(active_states, pending_counts):
            last_rows.append(offset + count - 1)
            offset += count
            state.clear_pending()

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
        logits, recurrent_hidden = self.drafter.finish_cache_rows(
            residual.index_select(0, last_rows),
            attention_output,
        )
        next_tokens = self._sample_batch(logits, active_states, 0)
        token_matrix = torch.full(
            (batch_size, K), -1, dtype=torch.int32, device=self.device
        )
        token_matrix[:, 0] = next_tokens
        first_positions = torch.tensor(
            [state.window_positions[-1] for state in active_states],
            dtype=torch.int64,
            device=self.device,
        )

        for depth in range(1, max(state.n_drafts for state in active_states)):
            active_indices = [
                idx for idx, state in enumerate(active_states)
                if state.n_drafts > depth
            ]
            active_rows = torch.tensor(
                active_indices, dtype=torch.int64, device=self.device
            )
            step_states = [active_states[idx] for idx in active_indices]
            step_batch_rows = batch_rows.index_select(0, active_rows)
            step_ids = token_matrix[active_rows, depth - 1]
            step_positions = first_positions.index_select(0, active_rows) + depth
            step_hidden = recurrent_hidden.index_select(0, active_rows)

            residual, query, key, value = self.drafter.prepare_cache_rows(
                step_ids, step_positions, step_hidden
            )
            slots = self.pool.scratch_slots(step_batch_rows, depth - 1)
            self.pool.store(slots, key, value)
            step_table_indices = [state.table_idx for state in step_states]
            cache_indices, cache_valid = self.pool.batch_indices(
                step_table_indices, step_batch_rows, scratch_depth=depth
            )
            attention_output = self.attention_backend.forward(
                query,
                self.pool,
                cache_indices,
                cache_valid,
                self.pool.cache_lengths(step_table_indices, scratch_depth=depth),
            )
            logits, step_hidden = self.drafter.finish_cache_rows(
                residual, attention_output
            )
            next_tokens = self._sample_batch(logits, step_states, depth)
            token_matrix[active_rows, depth] = next_tokens
            recurrent_hidden[active_rows] = step_hidden

        token_rows = token_matrix.tolist()
        for state, row in zip(active_states, token_rows):
            state.draft_tokens = row[: state.n_drafts]

    def _sample_batch(
        self,
        logits: torch.Tensor,
        states: list[Eagle3State],
        step  : int,
    ) -> torch.Tensor:
        draft_ids = logits.argmax(dim=-1)
        sampling_rows = [
            idx for idx, state in enumerate(states)
            if step < state.n_drafts and not state.sampling_params.is_greedy
        ]
        if sampling_rows:
            import flashinfer.sampling as sampling

            rows = torch.tensor(sampling_rows, dtype=torch.int64, device=self.device)
            params = [states[idx].sampling_params for idx in sampling_rows]
            probs = self.sampler.probabilities(
                logits.index_select(0, rows),
                self.sampler.prepare_params(params),
            )
            sampled = sampling.sampling_from_probs(probs).to(draft_ids.dtype)
            draft_ids.index_copy_(0, rows, sampled)
            target_probs = torch.zeros(
                len(sampling_rows),
                self.vocab_size,
                dtype=probs.dtype,
                device=self.device,
            )
            target_probs.index_copy_(
                1,
                self.drafter.hot_token_id.to(torch.int64),
                probs,
            )
            for prob, idx in zip(target_probs, sampling_rows):
                draft_probs = states[idx].draft_probs
                assert draft_probs is not None
                draft_probs[step].copy_(prob)

        return self.drafter.hot_token_id[draft_ids].to(torch.int32)


__all__ = ["Eagle3Engine"]
