from __future__ import annotations

import torch
import torch.nn.functional as F

from picosgl.layers import (
    BaseOP,
    GatedMLP,
    LinearColParallelPartitioned,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    get_rope,
)
from picosgl.models.weight import iter_checkpoint_weights

from .base import BaseDrafterModel


class DFlashAttention(BaseOP):
    def __init__(
        self,
        hidden_size : int,
        head_dim    : int,
        num_qo_heads: int,
        num_kv_heads: int,
        max_position: int,
        rope_base   : float,
        rms_norm_eps: float,
    ) -> None:
        self.head_dim = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_proj = LinearColParallelPartitioned(
            input_size=hidden_size,
            partition_size=head_dim,
            partitions=[(num_qo_heads, False)] + [(num_kv_heads, True)] * 2,
            has_bias=False,
        )
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=max_position,
            base=rope_base,
            rope_scaling=None,
        )
        self.o_proj = LinearRowParallel(
            num_qo_heads * head_dim,
            hidden_size,
            has_bias=False,
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj.forward(hidden_states)
        query, key, value = qkv.split(
            [
                self.num_qo_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        query = query.view(shape + (self.num_qo_heads, self.head_dim))
        key   = key.view(shape + (self.num_kv_heads, self.head_dim))
        value = value.view(shape + (self.num_kv_heads, self.head_dim))
        return query, key, value

    def project_context(
        self,
        hidden_states: torch.Tensor,
        positions    : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # qkv_proj is fused in pico-sglang, so Q is produced here as well. Rotating it
        # together with K lets the common FlashInfer RoPE path remain the only RoPE
        # implementation; the context query is discarded immediately afterwards.
        query, key, value = self._project(hidden_states)
        key = self.k_norm.forward(key)
        _, key = self.rotary.forward(positions, query, key)
        return key, value

    def project_block(
        self,
        hidden_states: torch.Tensor,
        positions    : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._project(hidden_states)
        query = self.q_norm.forward(query)
        key   = self.k_norm.forward(key)
        query, key = self.rotary.forward(positions, query, key)
        return query, key, value

    def finish(self, attention_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(attention_output.flatten(-2))


class DFlashDecoderLayer(BaseOP):
    def __init__(
        self,
        hidden_size      : int,
        head_dim         : int,
        num_qo_heads     : int,
        num_kv_heads     : int,
        intermediate_size: int,
        hidden_act       : str,
        max_position     : int,
        rope_base        : float,
        rms_norm_eps     : float,
    ) -> None:
        self.self_attn = DFlashAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            max_position=max_position,
            rope_base=rope_base,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def prepare_block(
        self,
        hidden_states: torch.Tensor,
        positions    : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        query, key, value = self.self_attn.project_block(hidden_states, positions)
        return residual, query, key, value

    def finish_block(
        self,
        residual        : torch.Tensor,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = residual + self.self_attn.finish(attention_output)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        return residual + hidden_states


class DFlashDrafter(BaseDrafterModel):
    def __init__(
        self,
        *,
        hidden_size        : int,
        head_dim           : int,
        num_qo_heads       : int,
        num_kv_heads       : int,
        intermediate_size  : int,
        hidden_act         : str,
        num_layers         : int,
        max_position       : int,
        rope_base          : float,
        rms_norm_eps       : float,
        vocab_size         : int,
        target_layer_ids   : tuple[int, ...],
        mask_token_id      : int,
        tie_word_embeddings: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.target_layer_ids = target_layer_ids
        self.mask_token_id = mask_token_id
        self.fc = LinearReplicated(
            hidden_size * len(target_layer_ids), hidden_size, has_bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.layers = OPList(
            [
                DFlashDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                    max_position=max_position,
                    rope_base=rope_base,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self._embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self._lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            tied_embedding=self._embed_tokens if tie_word_embeddings else None,
        )

    def load_weights(self, model_path: str, device: torch.device) -> None:
        state_dict: dict[str, torch.Tensor] = {}
        merge_buf: dict[str, dict[str, torch.Tensor]] = {}
        merge_info = {
            "q_proj": ("qkv_proj", ("q", "k", "v")),
            "k_proj": ("qkv_proj", ("q", "k", "v")),
            "v_proj": ("qkv_proj", ("q", "k", "v")),
            "gate_proj": ("gate_up_proj", ("gate", "up")),
            "up_proj": ("gate_up_proj", ("gate", "up")),
        }
        slot_names = {
            "q_proj": "q",
            "k_proj": "k",
            "v_proj": "v",
            "gate_proj": "gate",
            "up_proj": "up",
        }

        for name, tensor in iter_checkpoint_weights(model_path):
            component = name.removesuffix(".weight").rsplit(".", 1)[-1]
            if component not in merge_info:
                state_dict[name] = tensor.to(device)
                continue

            fused_component, slots = merge_info[component]
            fused_name = name.replace(f".{component}.", f".{fused_component}.")
            parts = merge_buf.setdefault(fused_name, {})
            parts[slot_names[component]] = tensor
            if all(slot in parts for slot in slots):
                state_dict[fused_name] = torch.cat(
                    [parts[slot] for slot in slots], dim=0
                ).to(device)
                del merge_buf[fused_name]

        assert not merge_buf, f"Incomplete DFlash merge groups: {list(merge_buf)}"
        self.load_state_dict(state_dict)

    def load_target_weights(self, model_path: str, device: torch.device) -> None:
        embedding_names = {
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        }
        lm_head_names = {
            "lm_head.weight",
            "language_model.lm_head.weight",
        }
        embedding: torch.Tensor | None = None
        lm_head: torch.Tensor | None = None
        for name, tensor in iter_checkpoint_weights(
            model_path,
            names=embedding_names | lm_head_names,
        ):
            if name in embedding_names:
                embedding = tensor
            elif name in lm_head_names:
                lm_head = tensor

        assert embedding is not None, (
            f"Target checkpoint {model_path} does not contain embed_tokens.weight"
        )
        self._embed_tokens.weight = embedding.to(
            device=device,
            dtype=self._embed_tokens.weight.dtype,
        )
        if not self._lm_head.tied_embedding:
            assert lm_head is not None, (
                f"Target checkpoint {model_path} does not contain lm_head.weight"
            )
            self._lm_head.weight = lm_head.to(
                device=device,
                dtype=self._lm_head.weight.dtype,
            )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm.forward(self.fc.forward(target_hidden))

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_tokens.forward(input_ids)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        module = self._lm_head.tied_embedding or self._lm_head
        return F.linear(hidden_states, module.weight, self._lm_head.bias)


__all__ = ["DFlashDrafter"]
