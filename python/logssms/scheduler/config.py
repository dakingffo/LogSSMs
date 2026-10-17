from __future__ import annotations

import os
from dataclasses import dataclass, field

from logssms.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_prefill_tokens: int = 8192
    max_decode_tokens : int | None = None
    _unique_suffix    : str  = field(default_factory=lambda: f".pid={os.getpid()}")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scheduling_policy == "mixed" and self.mixed_prefill_tokens > self.max_prefill_tokens:
            raise ValueError("mixed_prefill_tokens must be <= max_prefill_tokens")

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/logssms_0" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        return "ipc:///tmp/logssms_1" + self._unique_suffix
    
    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/logssms_2" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/logssms_3" + self._unique_suffix

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/logssms_4" + self._unique_suffix

    @property
    def zmq_drafter_addr(self) -> str:
        return "ipc:///tmp/logssms_6" + self._unique_suffix

    @property
    def zmq_drafter_reply_addr(self) -> str:
        return "ipc:///tmp/logssms_7" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_prefill_tokens

    @property
    def decode_batch_budget(self) -> int:
        if self.max_decode_tokens is not None:
            return self.max_decode_tokens
        base = max(1, self.max_running_req // 2)
        if self.enable_specualtive_decoding:
            assert self.speculator_config is not None
            return base * self.speculator_config.num_draft_tokens
        else:
            return base
