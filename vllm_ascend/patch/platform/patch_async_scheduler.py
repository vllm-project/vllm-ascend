# SPDX-License-Identifier: Apache-2.0
"""Backport vLLM #48245's AsyncScheduler stale-output handling."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus


def _update_request_with_output(
    self: AsyncScheduler,
    request: Request,
    new_token_ids: list[int],
    is_stale: bool = False,
) -> tuple[list[int], bool]:
    status_before_update = request.status
    new_token_ids, stopped = super(AsyncScheduler, self)._update_request_with_output(request, new_token_ids)

    # Placeholders were zeroed at preemption; a stale delivery must not
    # decrement them (it would underflow).
    if not is_stale:
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

    # Cache the new tokens. Preempted requests should be skipped.
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request,
            request.num_computed_tokens - request.num_output_placeholders,
        )
    return new_token_ids, stopped


AsyncScheduler._update_request_with_output = _update_request_with_output
