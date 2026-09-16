from __future__ import annotations

import os
from dataclasses import dataclass, field

from logssms.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_prefill_tokens: int = 8192  # chunk-prefill batch budget in tokens
    max_decode_tokens : int | None = None  # decode/verify batch budget in tokens; None => auto (decode_batch_budget)
    offline_mode      : bool = False
    _unique_suffix    : str  = field(default_factory=lambda: f".pid={os.getpid()}")

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
        """Resolved decode/verify batch budget in tokens.

        An explicit ``max_decode_tokens`` wins; otherwise ``max_running_req // 2``,
        scaled by the active speculator's draft length when speculative decoding is on. Each
        verify req occupies ``num_draft_tokens + 1`` positions, so scaling
        the budget keeps the req count at ~``max_running_req // 2`` while keeping the
        batch strictly smaller than the running-req count (no all-inflight empty
        iteration).
        """
        if self.max_decode_tokens is not None:
            return self.max_decode_tokens
        base = max(1, self.max_running_req // 2)
        if self.enable_specualtive_decoding:
            assert self.speculator_config is not None
            return base * self.speculator_config.num_draft_tokens
        else:
            return base
