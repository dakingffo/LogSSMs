from __future__ import annotations

from logssms.core import Batch

from .ar import ARManagerBase


class DecodeManager(ARManagerBase):
    @property
    def candidate_count(self) -> int:
        return sum(req.can_decode for req in self.running_reqs.values())

    def schedule_next_batch(self, max_requests: int | None = None) -> Batch | None:
        self.on_prefill_scheduled()
        budget = self.token_budget
        if max_requests is not None:
            budget = min(budget, max_requests)

        scheduled_token = 0
        reqs = []
        for uid, req in sorted(
            self.running_reqs.items(), 
            key=lambda x: x[0] in self.inflight_uids[0]

        ):
            if scheduled_token >= budget:
                break
            if req.can_decode:
                self.inflight_uids[1].add(uid)
                reqs.append(req)
                scheduled_token += 1

        if reqs:
            if max_requests is not None:
                # Bounded selections rotate for fairness. Unbounded selections
                # retain insertion order within each inflight priority group.
                for req in reqs:
                    self.running_reqs.pop(req.uid)
                    self.running_reqs[req.uid] = req
            return Batch(reqs=reqs, phase="decode")
        else:
            return None

    def advance_for_overlap(self, batch: Batch) -> None:
        for req in batch.decode_reqs:
            req.complete_n(1)
