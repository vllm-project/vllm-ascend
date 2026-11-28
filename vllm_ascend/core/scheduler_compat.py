# SPDX-License-Identifier: Apache-2.0
"""vLLM v0.34.0 compatibility helpers shared by copied Ascend schedulers."""

from __future__ import annotations

from typing import Any

from vllm_ascend.utils import vllm_version_is


def get_max_async_kv_loads_per_step(scheduler_config: Any) -> int | None:
    """Return max_async_kv_loads_per_step on main, or None on v0.34.0."""
    if vllm_version_is("0.34.0"):
        return None
    return scheduler_config.max_async_kv_loads_per_step


def scheduler_output_version_kwargs(
    scheduler: Any,
    iteration_details: Any,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> dict[str, Any]:
    """Build version-specific SchedulerOutput keyword arguments."""
    if vllm_version_is("0.34.0"):
        scheduled_encoder_input_stats = None
        if scheduler.log_stats and scheduler.observability_config.enable_logging_iteration_details:
            scheduled_encoder_input_stats = scheduler._make_scheduled_encoder_input_stats(scheduled_encoder_inputs)
        return {
            "scheduled_encoder_input_stats": scheduled_encoder_input_stats,
        }
    return {"iteration_details": iteration_details}
