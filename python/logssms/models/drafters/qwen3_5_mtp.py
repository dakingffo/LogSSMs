from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from logssms.layers import (
    LinearColumnParallel,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from logssms.kernel import sigmoid_and_mul
from logssms.models import load_weight
from logssms.models.qwen3_next import Qwen3NextDecoderLayer

from .base import BaseDrafterModel

if TYPE_CHECKING:
    from logssms.models.qwen3_next import Qwen3_5Config


class Qwen3_5MTPDrafter(BaseDrafterModel):

    def __init__(self, config: Qwen3_5Config):
        self._hidden_size = config.hidden_size
        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )
        self.pre_fc_norm_hidden = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )
        self.fc = LinearColumnParallel(
            config.hidden_size * 2, config.hidden_size, has_bias=False
        )
        self.layers = OPList(
            [
                Qwen3NextDecoderLayer(
                    config, 0, block_type="full_attention"
                )
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, zero_centered=True
        )


        self._embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self._lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=(
                self._embed_tokens if config.tie_word_embeddings else None
            ),
        )

    def load_weights(self, model_path: str, device: torch.device) -> None:
        mtp_sd: dict[str, torch.Tensor] = {}
        embed: torch.Tensor | None = None
        lm_head: torch.Tensor | None = None
        for key, value in load_weight(model_path, device):
            key = key.removeprefix("model.")
            if key.startswith("mtp."):
                mtp_sd[key] = value
            elif key == "embed_tokens.weight":
                embed = value
            elif key == "lm_head.weight":
                lm_head = value
        assert embed is not None, f"checkpoint {model_path} missing embed_tokens.weight"
        self.load_state_dict(mtp_sd, prefix="mtp")
        self._embed_tokens.weight = embed.to(self._embed_tokens.weight.dtype)
        if not self._lm_head.tied_embedding:
            assert lm_head is not None, (
                f"untied lm_head needs checkpoint lm_head.weight ({model_path})"
            )
            self._lm_head.weight = lm_head.to(self._lm_head.weight.dtype)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        module = self._lm_head.tied_embedding or self._lm_head
        logits = F.linear(hidden_states, module.weight, self._lm_head.bias)
        if self._lm_head.tp_size > 1:
            input_shape = logits.shape
            gathered = self._lm_head._comm.all_gather(logits)
            gathered = gathered.view((self._lm_head.tp_size,) + input_shape)
            gathered = gathered.permute(1, 0, 2).contiguous()
            logits = gathered.reshape(
                input_shape[:1] + (self._lm_head.tp_size * input_shape[1],)
            )[:, : module.num_embeddings]
        return logits

    def prepare_cache_rows(
        self,
        input_ids    : torch.Tensor,
        positions    : torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert input_ids.ndim == positions.ndim == 1
        assert hidden_states.shape == (input_ids.shape[0], self._hidden_size)
        emb = self._embed_tokens.forward(input_ids)
        emb = self.pre_fc_norm_embedding.forward(emb)
        h = self.pre_fc_norm_hidden.forward(hidden_states)
        h = torch.cat([emb, h], dim=-1)
        residual = self.fc.forward(h)

        layer = self.layers.op_list[0]
        h = layer.input_layernorm.forward(residual)[0]
        query, gate, key, value = layer.self_attn.proj(h, positions)
        return residual, query, gate, key, value

    def finish_cache_rows(
        self,
        residual        : torch.Tensor,
        gate            : torch.Tensor,
        attention_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers.op_list[0]
        h = sigmoid_and_mul(attention_output, gate, out=attention_output)
        h = layer.self_attn.o_proj.forward(
            h.reshape(-1, layer.self_attn.num_qo_heads * layer.self_attn.head_dim)
        )
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm.forward(h)[0]
        h = layer.mlp.forward(h)
        h = self.norm.forward(residual + h)
        return self.get_logits(h), h
