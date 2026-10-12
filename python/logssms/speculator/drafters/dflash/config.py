from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from logssms.utils import load_model_config

from ...base import BaseSpeculatorConfig, SpeculatorReserve

if TYPE_CHECKING:
    from logssms.engine.config import EngineConfig


@dataclass(frozen=True)
class DFlashSpeculatorConfig(BaseSpeculatorConfig):
    algorithm: ClassVar[str] = "DFLASH"

    block_size      : int = 16
    window_size     : int = 128
    target_layer_ids: tuple[int, ...] = (1, 6, 11, 16, 21)

    @property
    def num_draft_tokens(self) -> int:

        return self.block_size - 1

    @property
    def hidden_size_multiplier(self) -> int:
        return len(self.target_layer_ids)

    @property
    def max_init_hidden_rows(self) -> int:
        return self.window_size

    def make_reserve(self, max_running_req: int) -> SpeculatorReserve:
        return SpeculatorReserve(
            num_draft_slots=max_running_req * self.block_size,
            draft_slots_per_request=self.block_size,
        )

    def validate(self, config: EngineConfig) -> None:
        assert config.speculative_draft_model_path is not None
        draft_config = load_model_config(config.speculative_draft_model_path)
        assert getattr(draft_config, "architectures", None) == ["DFlashDraftModel"]
        dflash_config = draft_config.dflash_config
        checkpoint_layers = tuple(dflash_config["target_layer_ids"])
        assert self.target_layer_ids == checkpoint_layers, (
            f"DFLASH target layers {self.target_layer_ids} do not match checkpoint "
            f"layers {checkpoint_layers}"
        )
        assert draft_config.hidden_size == config.model_config.hidden_size
        assert draft_config.vocab_size == config.model_config.vocab_size
        assert draft_config.num_target_layers == config.model_config.num_layers
        assert 2 <= self.block_size <= int(draft_config.block_size)
        assert self.window_size >= 1
        assert len(set(self.target_layer_ids)) == len(self.target_layer_ids)
        assert min(self.target_layer_ids) >= 0

        assert max(self.target_layer_ids) + 1 < config.model_config.num_layers
        assert config.decode_batch_budget >= self.num_draft_tokens


__all__ = ["DFlashSpeculatorConfig"]
