# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from functools import wraps

from vllm.logger import logger
from vllm.v1.request import Request, RequestStatus


def _should_recompute_on_prefiller(request: Request, new_token_ids: list[int]) -> bool:
    params = request.kv_transfer_params
    return bool(
        request.status == RequestStatus.PREEMPTED
        and request.num_computed_tokens == 0
        and new_token_ids
        and params is not None
        and params.get("do_remote_decode")
        and request.num_output_tokens + len(new_token_ids) >= request.max_tokens
    )


def _patch_async_scheduler() -> None:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    original_update_request_with_output = AsyncScheduler._update_request_with_output
    if getattr(original_update_request_with_output, "_vllm_ascend_prefill_recompute_patched", False):
        return

    @wraps(original_update_request_with_output)
    def _patched_update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if _should_recompute_on_prefiller(request, new_token_ids):
            # Preemption has already freed the P-side block table and put the
            # request back in the waiting queue with num_computed_tokens=0.
            # Accepting this terminal in-flight output would finish the request
            # with no KV blocks to transfer. Consume its placeholders without
            # accepting its tokens so the prefiller recomputes the request.
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0
            logger.warning(
                "Dropping terminal in-flight output for preempted remote-decode "
                "request %s so the prefiller recomputes it.",
                request.request_id,
            )
            return [], False

        return original_update_request_with_output(self, request, new_token_ids)

    _patched_update_request_with_output._vllm_ascend_prefill_recompute_patched = True  # type: ignore[attr-defined]
    AsyncScheduler._update_request_with_output = _patched_update_request_with_output


_patch_async_scheduler()
