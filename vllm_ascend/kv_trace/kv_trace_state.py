# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local request bindings reconstructed from native scheduler messages."""

from __future__ import annotations

import threading
from typing import Any

BlockTable = tuple[tuple[int, ...], ...]


class KVCacheState:
    """Keep public binding deltas, without inventing allocation lifetimes."""

    def __init__(self, emit: Any):
        self.emit = emit
        self.requests: dict[str, BlockTable] = {}
        self.group_sizes: dict[int, int] = {}
        self.num_blocks = 0
        self._lock = threading.RLock()

    def configure(self, group_sizes: dict[int, int], num_blocks: int) -> None:
        self.group_sizes = group_sizes
        self.num_blocks = num_blocks

    def bind(self, output: Any) -> dict[str, tuple[int | None, int]]:
        """Apply native new/resumed replacements and cached block appends."""
        with self._lock:
            for request in output.finished_req_ids | set(output.preempted_req_ids or ()):
                self.requests.pop(request, None)
            computed = {}
            for request in output.scheduled_new_reqs:
                self.requests[request.req_id] = tuple(tuple(group) for group in request.block_ids)
                computed[request.req_id] = request.num_computed_tokens
            cached = output.scheduled_cached_reqs
            if not (len(cached.req_ids) == len(cached.new_block_ids) == len(cached.num_computed_tokens)):
                self.emit("kv_trace.violation", operation="bind", reason="cached_request_array_length_mismatch")
            for request, added, start in zip(cached.req_ids, cached.new_block_ids, cached.num_computed_tokens):
                computed[request] = start
                if request in cached.resumed_req_ids:
                    self.requests[request] = tuple(tuple(group) for group in added) if added is not None else ()
                elif added is not None:
                    old = self.requests.get(request)
                    if old is None:
                        self.emit(
                            "kv_trace.violation",
                            operation="bind",
                            reason="scheduler_binding_missing",
                            request_id=request,
                        )
                    elif len(old) != len(added):
                        self.emit(
                            "kv_trace.violation",
                            operation="bind",
                            reason="scheduler_group_count_mismatch",
                            request_id=request,
                        )
                    else:
                        self.requests[request] = tuple(prior + tuple(extra) for prior, extra in zip(old, added))
            if sum(output.num_scheduled_tokens.values()) != output.total_num_scheduled_tokens:
                self.emit("kv_trace.violation", operation="bind", reason="scheduled_token_total_mismatch")
            scheduled = {
                request: (computed.get(request), count) for request, count in output.num_scheduled_tokens.items()
            }
            for request, (start, count) in scheduled.items():
                if request not in self.requests or start is None or start < 0 or count <= 0:
                    self.emit(
                        "kv_trace.violation", operation="bind", reason="scheduler_range_missing", request_id=request
                    )
            self.emit(
                "kv_trace.bind",
                boundary="model_runner.execute_model.input",
                requests={request: self.requests.get(request) for request in dict.fromkeys((*computed, *scheduled))},
                scheduled_requests=scheduled,
                replay_starts={request.req_id: request.replay_start for request in output.scheduled_new_reqs},
                finished_request_ids=sorted(output.finished_req_ids),
                preempted_request_ids=sorted(output.preempted_req_ids or ()),
            )
            return scheduled
