# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle scaffold for the mooncake V1 workers.

The pull and layerwise workers used to duplicate their shutdown and
startup-readiness handling with only the log labels, thread attribute names and
the resource-release step differing. Those four methods live here as template
methods; a subclass supplies the family strings and three hooks.

Contract for subclasses:
  * _thread_label / _shutdown_log_name / _ready_thread_failed_message /
    _ready_timeout_message must be set (missing attributes raise AttributeError,
    never a silent no-op);
  * _transfer_threads() returns the (recv, send) thread attributes, either may
    be None;
  * _rollback_registration() releases this startup's native registration and
    must be idempotent;
  * _close_shutdown_resources() releases executor/socket resources AFTER the
    threads have been stopped and joined, and must not raise.
"""

import time
from typing import Any

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.constants import (
    KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS,
    SHUTDOWN_JOIN_TIMEOUT_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te


class MooncakeConnectorWorkerBase:
    """Idempotent shutdown, bounded startup wait and buffer-claim accounting."""

    _thread_label: str
    _shutdown_lock: Any
    _shutdown_complete: bool
    engine_id: str
    _shutdown_log_name: str
    _ready_thread_failed_message: str
    _ready_timeout_message: str

    def _transfer_threads(self):
        raise NotImplementedError

    def _rollback_registration(self) -> None:
        raise NotImplementedError

    def _close_shutdown_resources(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        """Idempotent best-effort shutdown of worker-owned threads and resources."""
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_complete = True
        label_lower = self._thread_label.lower()
        threads = [t for t in self._transfer_threads() if t is not None]
        for thread in threads:
            try:
                thread.stop()
            except Exception as error:
                logger.warning(
                    "Failed to signal %s thread stop. thread=%s. error=%s. ",
                    label_lower,
                    thread.name,
                    error,
                )
        for thread in threads:
            try:
                thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.warning(
                        "%s thread did not stop within %.1f s. thread=%s. ",
                        self._thread_label,
                        SHUTDOWN_JOIN_TIMEOUT_SECONDS,
                        thread.name,
                    )
            except Exception as error:
                logger.warning(
                    "Failed to join %s thread. thread=%s. error=%s. ",
                    label_lower,
                    thread.name,
                    error,
                )
        fatal_error = self.get_thread_fatal_error()
        if fatal_error is not None:
            logger.error(
                "%s transfer thread reported a fatal error. engine_id=%s. error=%s. ",
                self._thread_label,
                self.engine_id,
                fatal_error,
            )
        self._close_shutdown_resources()
        logger.info("%s shutdown complete. engine_id=%s", self._shutdown_log_name, self.engine_id)

    def _wait_until_ready(self, thread, ready_event) -> None:
        """Wait for a transfer thread to become ready; roll back on failure.

        The thread is stopped and joined before the native registrations are
        released. unregister_memory also disconnects peers, so releasing it
        while a thread is still alive would race with an in-flight transfer.
        """
        start_wait_time = time.time()
        while not ready_event.is_set():
            if not thread.is_alive():
                self.shutdown()
                self._rollback_registration()
                raise RuntimeError(self._ready_thread_failed_message)
            if time.time() - start_wait_time > KV_CACHE_THREAD_STARTUP_TIMEOUT_SECONDS:
                self.shutdown()
                self._rollback_registration()
                raise RuntimeError(self._ready_timeout_message)
            time.sleep(3)

    def _claim_kv_buffer_registration(self, register_regions) -> None:
        """Record whether this startup owns the process-wide KV buffers.

        A repeated call returns False once the buffers are already registered;
        the original owner keeps the claim, so a later rollback still releases
        the registration made by the first call.
        """
        if global_te.register_buffer(register_regions.ptrs, register_regions.lengths):
            self._registered_kv_buffer = True

    def get_thread_fatal_error(self) -> BaseException | None:
        """Return the first fatal error reported by a transfer thread, if any."""
        for thread in self._transfer_threads():
            error = getattr(thread, "fatal_error", None) if thread is not None else None
            if error is not None:
                return error
        return None
