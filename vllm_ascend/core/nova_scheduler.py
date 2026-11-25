# SPDX-License-Identifier: Apache-2.0
"""Recover requests whose NOVA exact window is no longer available."""

import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


class NovaScheduler(Scheduler):
    def __init__(self, *args, nova_executor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nova_executor = nova_executor

    def schedule(self, throttle_prefills=False):
        if self.nova_executor is None:
            raise RuntimeError("NOVA scheduler has no model executor")
        running = [request for request in self.running if request.status == RequestStatus.RUNNING]
        if running:
            request_ids = [request.request_id for request in running]
            computed = [request.num_computed_tokens for request in running]
            missing_by_worker = self.nova_executor.collective_rpc(
                "get_missing_nova_windows", args=(request_ids, computed)
            )
            if not missing_by_worker:
                raise RuntimeError("NOVA scheduler received no worker window states")
            missing = set().union(*missing_by_worker)
            for request in running:
                if request.request_id in missing:
                    self.running.remove(request)
                    self._preempt_request(request, time.monotonic())
        return super().schedule(throttle_prefills)
