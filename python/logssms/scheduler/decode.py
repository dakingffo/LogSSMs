from __future__ import annotations

from logssms.core import Batch

from .ar import ARManagerBase


class DecodeManager(ARManagerBase):
    def schedule_next_batch(self) -> Batch | None:
        self.inflight_uids[0] = self.inflight_uids[1]
        self.inflight_uids[1] = set()

        scheduled_token = 0
        reqs = []
        for uid, req in sorted(
            self.running_reqs.items(), 
            key=lambda x: x[0] in self.inflight_uids[0]

        ):
            if scheduled_token >= self.token_budget:
                break
            if req.can_decode:
                self.inflight_uids[1].add(uid)
                reqs.append(req)
                scheduled_token += 1

        if reqs:
            return Batch(reqs=reqs, phase="decode")
        else:
            return None

    def advance_for_overlap(self, batch: Batch) -> None:
        for req in batch.reqs:
            req.complete_n(1)
