# SPDX-License-Identifier: Apache-2.0
"""vLLM release/main output construction shared by copied schedulers."""

from typing import Any

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
    new_reqs_data: list,
    scheduled_encoder_inputs: dict[str, list[int]],
    encoder_compute_budget: int,
) -> tuple[Any, dict[str, Any]]:
    """Keep release payloads and populate main's scheduling-time statistics.

    Resumed requests are already included in new_reqs_data on main. Encoder
    counts use the final scheduled set, so preempted inputs are not counted.
    """
    if vllm_version_is("0.33.0"):
        return scheduler._make_cached_request_data(
            running_reqs, resumed_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks
        ), {}

    details = None
    if (
        scheduler.log_stats and scheduler.observability_config.enable_logging_iteration_details
    ) or scheduler.vllm_config.profiler_config.profiler is not None:
        details = IterationDetails()
    cached = scheduler._make_cached_request_data(
        running_reqs, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks, iteration_details=details
    )
    if details is not None:
        details.num_prefill_requests = len(num_scheduled_tokens) - details.num_decode_requests
        details.num_prefill_tokens = sum(num_scheduled_tokens.values()) - details.num_decode_tokens
        details.num_encoder_inputs = sum(len(inputs) for inputs in scheduled_encoder_inputs.values())
        details.num_encoder_output_tokens = scheduler.max_num_encoder_input_tokens - encoder_compute_budget
    return cached, {
        "iteration_details": details,
        "encoder_input_updates": scheduler._get_encoder_input_updates(
            num_scheduled_tokens, new_reqs_data, scheduled_encoder_inputs
        ),
    }
