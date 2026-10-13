# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-owned Scheduler wall-clock tracing for marked AscendStore requests.

The parent Scheduler is the alias already selected by Ascend's platform patches.
Schedule timings include connector lookup, allocation and metadata construction;
they are not pure CPU time or a residual inferred from model execution.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

STAGE_REQUEST_MARKER = "ascendstore-stage-"
NANOSECONDS_PER_MILLISECOND = 1_000_000


class AscendStoreTimingScheduler(Scheduler):
    """Trace only a marked benchmark round and flush after its successful cache reset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        transfer_config = self.vllm_config.kv_transfer_config
        assert transfer_config is not None
        self._stage_timing_path = Path(transfer_config.kv_connector_extra_config["benchmark_stage_timing_path"])
        self._stage_timing_enabled = False
        self._stage_records: list[dict[str, Any]] = []
        self._stage_step_id = 0
        self._stage_active_step_id: int | None = None
        self._stage_request_ids: set[str] = set()
        self._stage_last_request_ids: list[str] = []
        self._stage_schedule_start_ns: int | None = None
        self._stage_previous_update_end_ns: int | None = None
        self._stage_previous_update_request_ids: set[str] = set()
        self._stage_pid = os.getpid()

        assert self.connector is not None
        original_lookup = self.connector.get_num_new_matched_tokens

        def timed_external_lookup(*lookup_args: Any, **lookup_kwargs: Any) -> Any:
            if not self._stage_timing_enabled:
                return original_lookup(*lookup_args, **lookup_kwargs)
            request = lookup_args[0] if lookup_args else lookup_kwargs["request"]
            start_ns = time.perf_counter_ns()
            try:
                return original_lookup(*lookup_args, **lookup_kwargs)
            finally:
                end_ns = time.perf_counter_ns()
                self._record_stage(
                    "external_lookup_host", start_ns, end_ns, [request.request_id], {}, self._stage_active_step_id
                )

        # Only this Scheduler's connector is wrapped; the public connector class is unchanged.
        self.connector.get_num_new_matched_tokens = timed_external_lookup

    def add_request(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        if STAGE_REQUEST_MARKER in request.request_id:
            self._stage_timing_enabled = True
            self._stage_request_ids.add(request.request_id)
        return super().add_request(request, *args, **kwargs)

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        if not self._stage_timing_enabled:
            return super().schedule(*args, **kwargs)
        self._stage_step_id += 1
        self._stage_active_step_id = self._stage_step_id
        first_record = len(self._stage_records)
        scheduler_output = None
        start_ns = time.perf_counter_ns()
        self._stage_schedule_start_ns = start_ns
        try:
            scheduler_output = super().schedule(*args, **kwargs)
            return scheduler_output
        finally:
            end_ns = time.perf_counter_ns()
            scheduled_tokens = {} if scheduler_output is None else dict(scheduler_output.num_scheduled_tokens)
            for record in self._stage_records[first_record:]:
                if record["stage"] == "external_lookup_host" and record["step_id"] == self._stage_step_id:
                    record["scheduled_tokens"] = {
                        request_id: scheduled_tokens[request_id]
                        for request_id in record["request_ids"]
                        if request_id in scheduled_tokens
                    }
                    if not record["scheduled_tokens"]:
                        record["step_id"] = None
            request_ids = self._step_request_ids(scheduler_output, scheduled_tokens)
            shared_request_ids = self._stage_previous_update_request_ids.intersection(scheduled_tokens)
            if self._stage_previous_update_end_ns is not None and shared_request_ids:
                gap_tokens = {request_id: scheduled_tokens[request_id] for request_id in sorted(shared_request_ids)}
                self._record_stage(
                    "engine_between_steps_host",
                    self._stage_previous_update_end_ns,
                    start_ns,
                    list(gap_tokens),
                    gap_tokens,
                    self._stage_step_id,
                )
            # An update can pair only with the next schedule, not a later step across a cleanup or another request.
            self._stage_previous_update_end_ns = None
            self._stage_previous_update_request_ids.clear()
            if scheduler_output is None:
                self._stage_schedule_start_ns = None
            self._record_stage(
                "scheduler_schedule_host", start_ns, end_ns, request_ids, scheduled_tokens, self._stage_step_id
            )
            self._stage_active_step_id = None

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._stage_timing_enabled:
            return super().update_from_output(scheduler_output, model_runner_output, *args, **kwargs)
        scheduled_tokens = dict(scheduler_output.num_scheduled_tokens)
        request_ids = self._step_request_ids(scheduler_output, scheduled_tokens)
        start_ns = time.perf_counter_ns()
        try:
            return super().update_from_output(scheduler_output, model_runner_output, *args, **kwargs)
        finally:
            end_ns = time.perf_counter_ns()
            self._record_stage(
                "scheduler_update_from_output_host",
                start_ns,
                end_ns,
                request_ids,
                scheduled_tokens,
                self._stage_step_id,
            )
            if self._stage_schedule_start_ns is not None:
                # This parent range includes dispatch, future waiting and sampling, not just scheduling.
                self._record_stage(
                    "engine_schedule_to_update_host",
                    self._stage_schedule_start_ns,
                    end_ns,
                    request_ids,
                    scheduled_tokens,
                    self._stage_step_id,
                )
            self._stage_schedule_start_ns = None
            self._stage_previous_update_end_ns = end_ns
            self._stage_previous_update_request_ids = set(scheduled_tokens)

    def reset_prefix_cache(self, *args: Any, **kwargs: Any) -> Any:
        # A control RPC is not an adjacent request step, even if its reset is rejected.
        self._stage_previous_update_end_ns = None
        self._stage_previous_update_request_ids.clear()
        reset_successful = super().reset_prefix_cache(*args, **kwargs)
        if reset_successful:
            if self._stage_records:
                # The benchmark resets only after responses and Store drain, outside request timing.
                scheduler_class = type(self)
                report = {
                    "clock": "perf_counter_ns",
                    "pid": self._stage_pid,
                    "scheduler_class": f"{scheduler_class.__module__}.{scheduler_class.__qualname__}",
                    "scheduler_mro": [f"{base.__module__}.{base.__qualname__}" for base in scheduler_class.__mro__],
                    "records": self._stage_records,
                }
                self._stage_timing_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                self._stage_records.clear()
            self._stage_timing_enabled = False
            self._stage_step_id = 0
            self._stage_active_step_id = None
            self._stage_request_ids.clear()
            self._stage_last_request_ids.clear()
            self._stage_schedule_start_ns = None
            self._stage_previous_update_end_ns = None
            self._stage_previous_update_request_ids.clear()
        return reset_successful

    def _step_request_ids(self, scheduler_output: Any, scheduled_tokens: dict[str, int]) -> list[str]:
        if scheduled_tokens:
            self._stage_last_request_ids = list(scheduled_tokens)
            return self._stage_last_request_ids
        # These IDs describe cleanup ownership, never scheduled execution tokens.
        finished_ids = getattr(scheduler_output, "finished_req_ids", ())
        request_ids = sorted(self._stage_request_ids.intersection(finished_ids))
        return request_ids or self._stage_last_request_ids or sorted(self._stage_request_ids)

    def _record_stage(
        self,
        stage: str,
        start_ns: int,
        end_ns: int,
        request_ids: list[str],
        scheduled_tokens: dict[str, int],
        step_id: int | None,
    ) -> None:
        self._stage_records.append(
            {
                "stage": stage,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "host_ms": (end_ns - start_ns) / NANOSECONDS_PER_MILLISECOND,
                "pid": self._stage_pid,
                "thread_id": threading.get_ident(),
                "step_id": step_id,
                "request_ids": request_ids,
                "scheduled_tokens": scheduled_tokens,
            }
        )
