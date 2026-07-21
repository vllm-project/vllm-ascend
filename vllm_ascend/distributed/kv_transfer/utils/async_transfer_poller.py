# SPDX-License-Identifier: Apache-2.0
"""Generic async transfer poller for mooncake connector classes.

Extracts duplicated polling and completion logic shared across:
- mooncake_connector.KVCacheRecvingThread
- mooncake_hybrid_connector.KVCacheRecvingThread
- mooncake_layerwise_connector.KVCacheSendingLayerThread
"""

import logging
import threading
import time
from typing import Any, Callable

from vllm.logger import logger


class AsyncTransferPoller:
    """Generic poller for async mooncake transfers.

    Handles the common pattern of polling TransferEngine for batch completion,
    logging elapsed time, and dispatching callbacks on completion or failure.

    The class is parameterized with:
        engine: The TransferEngine instance used for polling.
        pending_async_transfers: Dict mapping batch_id -> info.
            Info can be either a tuple (request_id, remote_request_id, group_name, start_time)
            for the receiving-side connectors, or a dict {"req_ids": ..., "session_id": ...,
            "layer_idx": ..., "req_start_time": ...} for the layerwise sending connector.
        pending_async_lock: threading.Lock protecting pending_async_transfers.
        callback_map: Dict of callbacks:
            "on_completed"(batch_id, info): called when a transfer completes successfully.
            "on_failed"(batch_id, info): called when a transfer fails.
            "on_poll_error"(batch_id, info): called when polling raises an exception.
    """

    def __init__(
        self,
        engine: Any,
        pending_async_transfers: dict[int, Any],
        pending_async_lock: threading.Lock,
        callback_map: dict[str, Callable[[int, Any], None]],
    ):
        self.engine = engine
        self._pending_async_transfers = pending_async_transfers
        self._pending_async_lock = pending_async_lock
        self._on_completed = callback_map.get("on_completed")
        self._on_failed = callback_map.get("on_failed")
        self._on_poll_error = callback_map.get("on_poll_error")

    def poll_async_transfers(self) -> list[int]:
        """Poll all pending async transfers for completion.

        Returns:
            List of batch_ids that have completed successfully.
        """
        with self._pending_async_lock:
            batch_ids = list(self._pending_async_transfers.keys())
            if not batch_ids:
                return []

        completed = []
        failed = []
        for batch_id in batch_ids:
            try:
                result = self.engine.get_batch_transfer_status([batch_id])
                if result == 0:
                    completed.append(batch_id)
                elif result < 0:
                    failed.append(batch_id)
                    logger.error("Async transfer failed for batch %s, result=%d", batch_id, result)
            except Exception as e:
                logger.error("Error polling async transfer status for batch %s: %s", batch_id, e)
                failed.append(batch_id)

        # Handle failed transfers: pop from pending and dispatch on_failed callback
        for batch_id in failed:
            with self._pending_async_lock:
                if batch_id in self._pending_async_transfers:
                    info = self._pending_async_transfers.pop(batch_id)
                    self._log_transfer_warning(batch_id, info)
            if self._on_poll_error:
                self._on_poll_error(batch_id, info)
            if self._on_failed:
                self._on_failed(batch_id, info)

        if not completed:
            return []

        # Pop completed transfers and log elapsed time
        completed_transfers = []
        with self._pending_async_lock:
            for batch_id in completed:
                if batch_id in self._pending_async_transfers:
                    info = self._pending_async_transfers.pop(batch_id)
                    req_end_time = time.perf_counter()
                    req_transfer_elapsed = self._get_elapsed_ms(info, req_end_time)
                    completed_transfers.append((batch_id, info, req_transfer_elapsed))

        for batch_id, info, elapsed in completed_transfers:
            self._log_transfer_completed(batch_id, info, elapsed)
            if self._on_completed:
                self._on_completed(batch_id, info)

        return completed

    def _get_elapsed_ms(self, info: Any, end_time: float) -> float:
        """Extract start_time from info (tuple or dict) and compute elapsed ms."""
        if isinstance(info, dict):
            start_time = info.get("req_start_time", end_time)
        else:
            # Tuple format: (request_id, remote_request_id, group_name, start_time)
            start_time = info[3] if len(info) > 3 else end_time
        return (end_time - start_time) * 1000

    def _log_transfer_warning(self, batch_id: int, info: Any) -> None:
        """Log a warning for a failed transfer."""
        if isinstance(info, dict):
            logger.warning("Marked async transfer batch_id %s as failed", batch_id)
        else:
            remote_req_id = info[1] if len(info) > 1 else str(batch_id)
            logger.warning("Marked async transfer request %s as failed due to poll error", remote_req_id)

    def _log_transfer_completed(self, batch_id: int, info: Any, elapsed: float) -> None:
        """Log completion of a transfer."""
        if isinstance(info, dict):
            logger.debug(
                "Async write batch_id=%s (layer%d) completed, took %.2f ms",
                batch_id, info.get("layer_idx", -1), elapsed,
            )
        else:
            remote_req_id = info[1] if len(info) > 1 else str(batch_id)
            logger.debug(
                "Async KV cache transfer completed for request %s, batch_id=%s, took %.2f ms",
                remote_req_id, batch_id, elapsed,
            )

    def poll_and_complete_async(self) -> None:
        """Poll all pending async transfers and dispatch callbacks for completed ones.

        This is the combined entry point that calls poll_async_transfers()
        and relies on the callbacks registered in the callback_map to handle
        completion/failure logic.
        """
        self.poll_async_transfers()