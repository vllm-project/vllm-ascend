"""Communication channels for the AscendStore v1 classic path."""

from __future__ import annotations

from .protocol import LookupRequest, LookupResponse
from .worker import WorkerService


class LookupChannel:
    """Communication boundary between scheduler-side and worker-side lookup components."""

    def __init__(self, worker_service: WorkerService) -> None:
        self._worker_service = worker_service

    def lookup(self, request: LookupRequest) -> LookupResponse:
        return self._worker_service.lookup(request)
