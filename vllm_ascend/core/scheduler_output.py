# SPDX-License-Identifier: Apache-2.0
"""vLLM release/main output construction shared by copied schedulers."""

from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.output import CachedRequestData

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.33.0"):
    from vllm.v1.core.sched.output import IterationDetails


def make_cached_request_data(
    scheduler: Any,
    running_reqs: list,
    resumed_reqs: list,
    num_scheduled_tokens: dict[str, int],
    spec_decode_tokens: dict[str, list[int]],
    req_to_new_blocks: dict,
    iteration_details: IterationDetails | None = None,
) -> CachedRequestData:
    """Construct cached-request data using the upstream version's signature."""
    if vllm_version_is("0.33.0"):
        return scheduler._make_cached_request_data(
            running_reqs, resumed_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks
        )

    return scheduler._make_cached_request_data(
        running_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks, iteration_details=iteration_details
    )
