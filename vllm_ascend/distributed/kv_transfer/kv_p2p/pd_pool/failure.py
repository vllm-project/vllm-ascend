# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finish failed producer handoffs before vLLM builds the prefill response."""

from functools import wraps

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.pd_pool.metadata import PoolPDWorkerMetadata


def install_handoff_failure_handler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    update_from_output = Scheduler.update_from_output
    if getattr(update_from_output, "_pool_pd_handoff_failure_handler", False):
        return

    @wraps(update_from_output)
    def update(self, scheduler_output, model_runner_output):
        # update_connector_output runs after request_finished. Consume the
        # aggregated worker failures first so a failed P never emits a handoff.
        connector_output = model_runner_output.kv_connector_output
        meta = connector_output.kv_connector_worker_meta if connector_output is not None else None
        failed = ()
        if isinstance(meta, PoolPDWorkerMetadata) and meta.failed_handoffs:
            failed = self.finish_requests(meta.failed_handoffs, RequestStatus.FINISHED_ERROR)

        outputs = update_from_output(self, scheduler_output, model_runner_output)
        for request in failed:
            client = outputs.setdefault(request.client_index, EngineCoreOutputs())
            client.outputs.append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )
        return outputs

    update._pool_pd_handoff_failure_handler = True
    Scheduler.update_from_output = update
