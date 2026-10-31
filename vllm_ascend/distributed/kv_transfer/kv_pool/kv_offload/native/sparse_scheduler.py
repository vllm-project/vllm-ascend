# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Keep incomplete PD receives out of the native prefix store."""

from copy import copy

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus


class SparsePrefixOffloadingScheduler(OffloadingConnectorScheduler):
    """Retain upstream load/fence/cleanup logic, with conservative save admission.

    During remote prefill, core's computed-token count includes KV that PD has
    not delivered yet. An aborted request must not publish that range, even
    after its independent native prefix load completes. Failed transfers may
    also leave invalid main-KV or indexer pages. Neither is a store candidate.
    """

    def _store_ready_output(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        excluded = {
            req_id
            for req_id in scheduler_output.finished_req_ids or ()
            if (state := self._req_status.get(req_id)) is not None
            and state.req.status
            in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR, RequestStatus.FINISHED_IGNORED)
        }
        if not excluded:
            return scheduler_output
        output = copy(scheduler_output)
        output.finished_req_ids = set(scheduler_output.finished_req_ids) - excluded
        output.num_scheduled_tokens = {
            req_id: count for req_id, count in scheduler_output.num_scheduled_tokens.items() if req_id not in excluded
        }
        return output

    def _build_store_jobs(self, scheduler_output: SchedulerOutput):
        return super()._build_store_jobs(self._store_ready_output(scheduler_output))

    def _build_partial_tail_store_jobs(self, scheduler_output: SchedulerOutput):
        return super()._build_partial_tail_store_jobs(self._store_ready_output(scheduler_output))
