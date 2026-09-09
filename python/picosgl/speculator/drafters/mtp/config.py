from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...base import BaseSpeculatorConfig, SpeculatorReserve

if TYPE_CHECKING:
    from picosgl.engine.config import EngineConfig


@dataclass(frozen=True)
class MTPSpeculatorConfig(BaseSpeculatorConfig):
    algorithm: ClassVar[str] = "MTP"

    num_draft_tokens: int = 3
    window_size     : int = 128

    @property
    def max_init_hidden_rows(self) -> int:
        return self.window_size

    def make_reserve(self, max_running_req: int) -> SpeculatorReserve:
        width = self.num_draft_tokens + 1
        return SpeculatorReserve(
            num_draft_slots=max_running_req * width,
            draft_slots_per_request=width,
        )

    def validate(self, config: EngineConfig) -> None:
        assert config.speculative_draft_model_path == config.model_path, (
            "--speculative-draft-model-path must equal --model-path under MTP "
            f"(got {config.speculative_draft_model_path!r} vs {config.model_path!r})."
        )
        assert getattr(config.model_config, "mtp_num_hidden_layers", 0) > 0, (
            "MTP speculative decoding requires a model with an MTP head "
            "(mtp_num_hidden_layers > 0)."
        )
        assert self.num_draft_tokens >= 1, (
            "--speculative-num-draft-tokens must be >= 1"
        )
        assert config.decode_batch_budget >= self.num_draft_tokens, (
            "--max-decode-tokens must be >= --speculative-num-draft-tokens, "
            "or the draft-state region is empty"
        )
