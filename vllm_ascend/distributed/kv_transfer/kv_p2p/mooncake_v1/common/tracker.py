# SPDX-License-Identifier: Apache-2.0
"""Request completion tracker shared by the mooncake v1 KV connectors.

The tracker is a pure in-memory state machine: it records requests that are
waiting for their peer transfer to finish, the finished set consumed by the
scheduler, and the delayed-free queue used to bound how long the prefill node
keeps KV blocks alive after a request is aborted.

``time`` and ``envs`` are imported at module scope on purpose: the unit tests
patch them through ``sys.modules[KVCacheTaskTracker.__module__]`` to control the
delayed-free timeout.
"""

import threading
import time
from collections import OrderedDict

from vllm import envs
from vllm.logger import logger


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str, *, report_completion: bool = True):
        with self.done_task_lock:
            if not report_completion:
                # A notification-only request never entered WAITING_FOR_REMOTE_KVS.
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
                return
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.warning(
                    "MooncakeConnector finish req not in reqs to process. "
                    "request_id=%s. "
                    "Possible cause: Request was already completed or not properly tracked. "
                    "Check: Verify request lifecycle and tracking logic.",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.error(
                    "Force freed expired request: %s. "
                    "Reason: Request exceeded timeout threshold (%s seconds). "
                    "Action: Resources have been forcibly released to prevent memory leak.",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests
