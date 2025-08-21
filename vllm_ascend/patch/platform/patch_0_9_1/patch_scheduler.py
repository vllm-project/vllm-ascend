from collections.abc import Iterable
from typing import Union

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

def finish_requests(
    self,
    request_ids: Union[str, Iterable[str]],
    finished_status: RequestStatus,
) -> None:
    """Handles the finish signal from outside the scheduler.

    For example, the API server can abort a request when the client
    disconnects.
    """
    assert RequestStatus.is_finished(finished_status)
    if isinstance(request_ids, str):
        request_ids = (request_ids, )
    else:
        request_ids = set(request_ids)

    for req_id in request_ids:
        request = self.requests.get(req_id)
        if request is None:
            # Invalid request ID.
            continue
        if request in self.waiting or request in self.running:
            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
        request.status = finished_status
        self._free_request(request)


Scheduler.finish_requests = finish_requests
