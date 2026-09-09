from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from picosgl.engine import Sampler
from picosgl.models.drafters import DFlashDrafter
from picosgl.utils import load_model_config, torch_dtype

from ...base import EngineBase
from ..attention import DraftAttentionBackend
from ..pool import DraftKVPool
from .config import DFlashSpeculatorConfig
from .state import DFlashState

if TYPE_CHECKING:
    from picosgl.engine.config import EngineConfig
    from picosgl.scheduler.config import SchedulerConfig


class DFlashEngine(EngineBase):
    def __init__(
        self,
        drafter          : DFlashDrafter,
        device           : torch.device,
        vocab_size       : int,
        block_size       : int,
        max_running_req  : int,
        window_size      : int,
        max_batch_size   : int,
        attention_backend: str,
    ) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.num_spec_tokens = block_size - 1
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.window_size = window_size
        self.drafter = drafter
        self.sampler = Sampler(device, vocab_size)

        attention = drafter.layers.op_list[0].self_attn
        self.pools = [
            DraftKVPool(
                max_running_req=max_running_req,
                window_size=window_size,
                max_batch_size=max_batch_size,
                num_spec_tokens=block_size,
                num_kv_heads=attention.num_kv_heads,
                head_dim=attention.head_dim,
                dtype=attention.qkv_proj.weight.dtype,
                device=device,
            )
            for _ in drafter.layers.op_list
        ]
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
    ) -> DFlashEngine:
        drafter = cls.load_drafter(device, config)
        speculator_config = config.speculator_config
        assert isinstance(speculator_config, DFlashSpeculatorConfig)
        K = speculator_config.num_draft_tokens
        return cls(
            drafter=drafter,
            device=device,
            vocab_size=config.model_config.vocab_size,
            block_size=speculator_config.block_size,
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
    ) -> DFlashDrafter:
        from picosgl.layers import set_rope_device

        model_path = config.speculative_draft_model_path
        assert model_path is not None
        draft_config = load_model_config(model_path)
        dflash_config = draft_config.dflash_config
        target_layer_ids = tuple(dflash_config["target_layer_ids"])
        set_rope_device(device)
        with torch.device("meta"), torch_dtype(config.dtype):
            drafter = DFlashDrafter(
                hidden_size=draft_config.hidden_size,
                head_dim=draft_config.head_dim,
                num_qo_heads=draft_config.num_attention_heads,
                num_kv_heads=draft_config.num_key_value_heads,
                intermediate_size=draft_config.intermediate_size,
                hidden_act=draft_config.hidden_act,
                num_layers=draft_config.num_hidden_layers,
                max_position=draft_config.max_position_embeddings,
                rope_base=draft_config.rope_theta,
                rms_norm_eps=draft_config.rms_norm_eps,
                vocab_size=draft_config.vocab_size,
                target_layer_ids=target_layer_ids,
                mask_token_id=dflash_config["mask_token_id"],
                tie_word_embeddings=config.model_config.tie_word_embeddings,
            )
        drafter.load_weights(model_path, device)
        drafter.load_target_weights(config.model_path, device)
        return drafter

    def draft(self, states: list[DFlashState]) -> None:
        K = self.num_spec_tokens
        for state in states:
            assert 0 <= state.n_drafts <= K
            state.draft_tokens = []
            if state.n_drafts > 0 and not state.sampling_params.is_greedy:
                state.draft_probs = torch.empty(
                    state.n_drafts,
                    self.vocab_size,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                state.draft_probs = None

        active_states = [state for state in states if state.n_drafts > 0]
        if not active_states:
            return
        batch_size = len(active_states)
        assert batch_size <= self.pools[0].max_batch_size
        table_indices = [state.table_idx for state in active_states]
        assert len(set(table_indices)) == batch_size
        batch_rows = torch.arange(batch_size, dtype=torch.int64, device=self.device)

        pending_counts = []
        for state in active_states:
            if not state.cache_initialized:
                for pool in self.pools:
                    pool.reset(state.table_idx)
                state.cache_initialized = True
            count = len(state.pending_positions)
            assert count > 0, "every DFlash round must append target context"
            assert count == state.pending_hidden.shape[0]
            pending_counts.append(count)

        context_positions = torch.tensor(
            [
                position
                for state in active_states
                for position in state.pending_positions
            ],
            dtype=torch.int64,
            device=self.device,
        )
        target_hidden = self.drafter.project_target_hidden(
            torch.cat([state.pending_hidden for state in active_states], dim=0)
        )
        for layer, pool in zip(self.drafter.layers.op_list, self.pools):
            key, value = layer.self_attn.project_context(
                target_hidden, context_positions
            )
            slots = pool.append_persistent_batch(table_indices, pending_counts)
            pool.store(slots, key, value)
        for state in active_states:
            state.clear_pending()

        groups: dict[tuple[int, bool], list[int]] = {}
        for index, state in enumerate(active_states):
            groups.setdefault(
                (state.n_drafts, state.sampling_params.is_greedy), []
            ).append(index)
        for (num_drafts, is_greedy), indices in groups.items():
            self._draft_group(
                active_states,
                batch_rows,
                indices,
                num_drafts,
                is_greedy,
            )

    def _draft_group(
        self,
        states       : list[DFlashState],
        batch_rows   : torch.Tensor,
        state_indices: list[int],
        num_drafts   : int,
        is_greedy    : bool,
    ) -> None:
        block_size = num_drafts + 1
        rows = torch.tensor(state_indices, dtype=torch.int64, device=self.device)
        group_rows = batch_rows.index_select(0, rows)
        group_states = [states[index] for index in state_indices]
        table_indices = [state.table_idx for state in group_states]
        group_size = len(group_states)

        input_ids = torch.full(
            (group_size, block_size),
            self.drafter.mask_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        input_ids[:, 0] = torch.tensor(
            [state.anchor_token for state in group_states],
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.tensor(
            [state.anchor_position for state in group_states],
            dtype=torch.int64,
            device=self.device,
        )[:, None] + torch.arange(block_size, device=self.device)[None, :]
        hidden_states = self.drafter.embed(input_ids.flatten())
        flat_positions = positions.flatten()

        for layer, pool in zip(self.drafter.layers.op_list, self.pools):
            residual, query, key, value = layer.prepare_block(
                hidden_states, flat_positions
            )
            slots = pool.scratch_block(group_rows, block_size)
            pool.store(slots, key, value)
            cache_indices, cache_valid = pool.batch_indices(
                table_indices,
                group_rows,
                scratch_depth=block_size,
            )
            attention_output = self.attention_backend.forward_block(
                query.view(
                    group_size,
                    block_size,
                    query.shape[-2],
                    query.shape[-1],
                ),
                pool,
                cache_indices,
                cache_valid,
                pool.cache_lengths(table_indices, scratch_depth=block_size),
            )
            hidden_states = layer.finish_block(
                residual,
                attention_output.flatten(0, 1),
            )

        hidden_states = self.drafter.norm.forward(hidden_states)
        prediction_hidden = hidden_states.view(
            group_size, block_size, -1
        )[:, 1:].flatten(0, 1)
        logits = self.drafter.get_logits(prediction_hidden)
        if is_greedy:
            draft_tokens = logits.argmax(dim=-1).to(torch.int32)
        else:
            params = [
                state.sampling_params
                for state in group_states
                for _ in range(num_drafts)
            ]
            sampling_args = self.sampler.prepare_params(params)
            probs = self.sampler.probabilities(logits, sampling_args)
            import flashinfer.sampling as sampling

            draft_tokens = sampling.sampling_from_probs(probs).to(torch.int32)
            probs = probs.view(group_size, num_drafts, self.vocab_size)
            for state, state_probs in zip(group_states, probs):
                assert state.draft_probs is not None
                state.draft_probs.copy_(state_probs)

        token_rows = draft_tokens.view(group_size, num_drafts).tolist()
        for state, tokens in zip(group_states, token_rows):
            state.draft_tokens = tokens


__all__ = ["DFlashEngine"]
