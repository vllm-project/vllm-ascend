"""Mark the first formal AISBench request for selected profiling benchmarks.

This module stays separate because importing the adapter requires the optional
AISBench package, while the rest of the profiling workflow does not.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from tools.profiling.capture import POLL_INTERVAL_SECONDS, ProfileSpec, record_first_request


class TimedGenInferencer(GenInferencer):
    """Keep warmup outside the profiling timeline; ordinary runs use GenInferencer."""

    async def warmup(self, *args, **kwargs):
        # AISBench routes warmup through do_request too, but does not include
        # those requests in its performance details or benchmark duration.
        self._warming_up = True
        try:
            return await super().warmup(*args, **kwargs)
        finally:
            self._warming_up = False

    def _mark_request(self) -> None:
        """Mark once per adapter and propagate errors that prevent capture."""
        if getattr(self, "_warming_up", False) or getattr(self, "_profile_marker_attempted", False):
            return
        record_first_request(Path(os.environ["ASCEND_PROFILE_REQUEST_MARKER"]))
        self._profile_marker_attempted = True

    async def do_request(self, data, token_bucket, session):
        """Called after request-rate/concurrency waits, before model inference."""
        self._mark_request()
        return await super().do_request(data, token_bucket, session)

    def inference_with_shm(self, *args, **kwargs):
        """Publish each worker's formal success counter outside the request path.

        AISBench resets and starts StatusCounter here, after warmup. Its get_req
        counter advances only for output.success, including with multiple workers.
        """
        spec = ProfileSpec.from_env()
        if spec.start_after_requests is None:
            return super().inference_with_shm(*args, **kwargs)
        progress = Path(os.environ["ASCEND_PROFILE_REQUEST_MARKER"]).parent / "completed_requests"
        path = progress / f"{os.getpid()}.count"
        stopped = Event()

        def publish_count():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(str(self.status_counter.get_req))
            temporary.replace(path)

        def monitor():
            while not stopped.wait(POLL_INTERVAL_SECONDS):
                publish_count()

        with ThreadPoolExecutor(max_workers=1) as pool:
            reporter = pool.submit(monitor)
            try:
                return super().inference_with_shm(*args, **kwargs)
            finally:
                stopped.set()
                reporter.result()
                # The base method drains StatusCounter before returning.
                publish_count()
