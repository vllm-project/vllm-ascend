# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Host-only readiness certificates for steady Mooncake decode.

Completion notifications are observed at the runner's existing consumption
point. This module never polls a connector, synchronizes an event, or changes
transfer metadata. A notification alone does not certify a request: the first
stable batch must also consume an exact length snapshot after its prompt.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class _Request:
    generation: int
    pending: bool = False
    failed: bool = False
    ambiguous: bool = False
    slot: int | None = None
    prompt_len: int | None = None
    certified: bool = False


class PDDecodeQueueState:
    """One worker's proof, bound to request, slot, and transfer generation.

    The runner and its connector callbacks use this on the same host thread.
    Other requests may receive asynchronously without invalidating ready rows.
    Since completion records contain only request IDs, overlapping transfers
    for a reused ID cannot be disambiguated and keep that request on baseline.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _Request] = {}
        # A retired pending transfer keeps only an outstanding-receipt count.
        # Normal completion drains it; completed request IDs are not retained
        # for the lifetime of a long-running service.
        self._inflight_receives: Counter[str] = Counter()
        self._next_generation = 0
        self._last_output: Any = None
        self._metadata_known = False
        self._protocol_uncertain = False
        self._step_actions: set[str] = set()
        self._snapshot_generation = 0
        self._snapshot: dict[str, tuple[int, int]] = {}
        self.stats: Counter[str] = Counter()

    def _new_request(self, req_id: str) -> _Request:
        self._next_generation += 1
        pending = self._inflight_receives.get(req_id, 0) > 0
        request = _Request(self._next_generation, pending=pending, ambiguous=pending)
        self._requests[req_id] = request
        return request

    def _retire(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
        self._snapshot.pop(req_id, None)

    def invalidate(self) -> None:
        for req_id in tuple(self._requests):
            self._retire(req_id)
        self._last_output = None
        self._metadata_known = False
        self._snapshot.clear()
        self.stats["invalidations"] += 1

    def observe_step(self, output: Any, *, metadata_known: bool) -> None:
        if output is self._last_output:
            return
        self._last_output = output
        self._metadata_known = metadata_known and not self._protocol_uncertain
        self._step_actions = set()
        if not metadata_known:
            self._protocol_uncertain = True
            self.invalidate()
            self._last_output = output
            return
        metadata = output.kv_connector_metadata
        try:
            retired = set(output.finished_req_ids) | set(output.preempted_req_ids)
            retired.update(output.scheduled_cached_reqs.resumed_req_ids)
            for req_id in retired:
                self._retire(req_id)
            for new_req in output.scheduled_new_reqs:
                request = self._requests.get(new_req.req_id)
                # The transfer can complete before the scheduler first admits
                # a request to this worker. Preserve that unbound receipt.
                if request is not None and request.slot is not None:
                    self._retire(new_req.req_id)
            for req_id in metadata.requests:
                request = self._new_request(req_id)
                request.pending = True
                self._inflight_receives[req_id] += 1
                self._snapshot.pop(req_id, None)
                self.stats["receive_generations"] += 1
            self._step_actions = set(metadata.requests) | set(metadata.requests_to_send) | set(metadata.reqs_in_batch)
            for req_id in output.num_scheduled_tokens:
                if req_id not in self._requests:
                    self._new_request(req_id)
        except (AttributeError, KeyError, TypeError, ValueError):
            self._protocol_uncertain = True
            self.invalidate()
            self._last_output = output

    def observe_completion(self, output: Any) -> None:
        if output is None:
            return
        self.stats["completion_observations"] += 1
        try:
            failed = set(output.failed_recving or ())
            finished = set(output.finished_recving or ())
            invalid_blocks = bool(output.invalid_block_ids)
            if invalid_blocks:
                # The public output has no block-to-request ownership table.
                # Revoke proofs rather than infer which request was affected.
                for existing_request in self._requests.values():
                    existing_request.failed = True
                    existing_request.certified = False
                self.stats["invalid_block_batches"] += 1
            for req_id in finished | failed:
                outstanding = self._inflight_receives.get(req_id, 0)
                if outstanding == 1:
                    del self._inflight_receives[req_id]
                elif outstanding > 1:
                    self._inflight_receives[req_id] = outstanding - 1
                request = self._requests.get(req_id)
                if outstanding == 0 or request is None or not request.pending:
                    self.stats["unmatched_completions"] += 1
                    continue
                request.pending = outstanding > 1
                request.failed = request.failed or invalid_blocks or req_id in failed
                request.certified = False
                self.stats["failed_receives" if request.failed else "completed_receives"] += 1
        except (AttributeError, TypeError, ValueError):
            self._protocol_uncertain = True
            self.invalidate()

    def record_snapshot(self, indices: Mapping[str, int], generation: int) -> None:
        self._snapshot_generation = generation
        self._snapshot = {
            req_id: (int(slot), self._requests[req_id].generation)
            for req_id, slot in indices.items()
            if req_id in self._requests
        }

    def _rows(
        self,
        req_ids: Sequence[str],
        slots: Sequence[int],
        prompt_lens: Sequence[int],
        generation: int,
        *,
        require_binding: bool = True,
    ) -> list[tuple[_Request, int, int]] | None:
        if (
            not self._metadata_known
            or generation != self._snapshot_generation
            or not req_ids
            or len(req_ids) != len(slots)
            or len(req_ids) != len(prompt_lens)
            or self._step_actions.intersection(req_ids)
        ):
            return None
        rows = []
        for req_id, raw_slot, raw_prompt_len in zip(req_ids, slots, prompt_lens):
            slot, prompt_len = int(raw_slot), int(raw_prompt_len)
            request = self._requests.get(req_id)
            if (
                request is None
                or request.pending
                or request.failed
                or request.ambiguous
                or prompt_len <= 0
                or self._snapshot.get(req_id) != (slot, request.generation)
                or (require_binding and request.slot is not None and request.slot != slot)
                or (require_binding and request.prompt_len is not None and request.prompt_len != prompt_len)
            ):
                return None
            rows.append((request, slot, prompt_len))
        return rows

    def certify(
        self,
        req_ids: Sequence[str],
        slots: Sequence[int],
        prompt_lens: Sequence[int],
        exact_counts: Sequence[int],
        generation: int,
    ) -> bool:
        # A consumer can execute a local prefill or full prefix-cache hit
        # without a remote receive. Scheduler admission plus this exact prompt
        # boundary is sufficient; an observed receive stays pending until its
        # normal completion callback and cannot use this route prematurely.
        rows = self._rows(req_ids, slots, prompt_lens, generation, require_binding=False)
        if rows is None or len(rows) != len(exact_counts):
            return False
        if any(int(count) < prompt_len for (_, _, prompt_len), count in zip(rows, exact_counts)):
            return False
        for request, slot, prompt_len in rows:
            request.slot = slot
            request.prompt_len = prompt_len
            request.certified = True
        self.stats["exact_certified_batches"] += 1
        return True

    def can_queue(
        self, req_ids: Sequence[str], slots: Sequence[int], prompt_lens: Sequence[int], generation: int
    ) -> bool:
        rows = self._rows(req_ids, slots, prompt_lens, generation)
        ready = rows is not None and all(request.certified for request, _, _ in rows)
        self.stats["eligible_batches" if ready else "fallback_batches"] += 1
        return ready

    def snapshot(self) -> dict[str, object]:
        return {
            "stats": dict(self.stats),
            "requests": len(self._requests),
            "pending_requests": sum(request.pending for request in self._requests.values()),
            "certified_requests": sum(request.certified for request in self._requests.values()),
            "snapshot_generation": self._snapshot_generation,
            "outstanding_receive_ids": len(self._inflight_receives),
            "outstanding_receives": sum(self._inflight_receives.values()),
        }
