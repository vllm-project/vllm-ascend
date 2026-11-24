# SPDX-License-Identifier: Apache-2.0
"""vLLM v0.33.0 compatibility helpers shared by copied Ascend schedulers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler as _SchedulerBase
else:
    _SchedulerBase = object


class _V033SchedulerMembers(_SchedulerBase):
    """Provide main-only Scheduler members missing in vLLM v0.33.0."""

    cache_aware_window: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.async_kv_waiting: RequestQueue = create_request_queue(self.policy)
        self._parked_reservations: dict[Request, int] = {}

    def _next_waiting_queue(self, take_async_kv: bool) -> RequestQueue:
        return self.waiting

    def _admits_inflight_kv_load(self) -> bool:
        return True

    def _reserve_parked(self, request: Request) -> None:
        self._inflight_prefills.add(request)

    def _unreserve_parked(self, request: Request) -> None:
        return


def with_v033_compat(base: type) -> type:
    """Add v0.33.0-only scheduler members; a no-op on vLLM main."""
    if not vllm_version_is("0.33.0"):
        return base
    if issubclass(base, _V033SchedulerMembers):
        return base
    return type(f"V033Compat{base.__name__}", (_V033SchedulerMembers, base), {"__module__": __name__})


def get_max_async_kv_loads_per_step(scheduler_config: Any) -> int | None:
    """Return max_async_kv_loads_per_step on main, or None on v0.33.0."""
    if vllm_version_is("0.33.0"):
        return None
    return scheduler_config.max_async_kv_loads_per_step


def get_max_num_inflight_kv_loads(scheduler_config: Any) -> int | None:
    """Return max_num_inflight_kv_loads on main, or None on v0.33.0."""
    if vllm_version_is("0.33.0"):
        return None
    return scheduler_config.max_num_inflight_kv_loads


def should_skip_external_kv(request: Any) -> bool:
    """Return whether external KV lookup should be skipped for this request."""
    if vllm_version_is("0.33.0"):
        return bool(request.skip_reading_prefix_cache)
    return bool(request.skip_reading_external_kv)


def get_new_request_prefill_token_ids(request: Any) -> Any:
    """Return V2 prefill token IDs for NewRequestData.from_request."""
    if vllm_version_is("0.33.0"):
        return request._all_token_ids
    return request.prefill_token_ids


def scheduler_output_version_kwargs(
    scheduler: Any,
    iteration_details: Any,
    scheduled_encoder_inputs: dict[str, list[int]],
) -> dict[str, Any]:
    """Build version-specific SchedulerOutput keyword arguments."""
    if vllm_version_is("0.33.0"):
        scheduled_encoder_input_stats = None
        if scheduler.log_stats and scheduler.observability_config.enable_logging_iteration_details:
            scheduled_encoder_input_stats = scheduler._make_scheduled_encoder_input_stats(scheduled_encoder_inputs)
        return {
            "scheduled_encoder_input_stats": scheduled_encoder_input_stats,
            "ec_manager_metadata": scheduler.encoder_cache_manager.get_manager_metadata(),
        }
    return {"iteration_details": iteration_details}
