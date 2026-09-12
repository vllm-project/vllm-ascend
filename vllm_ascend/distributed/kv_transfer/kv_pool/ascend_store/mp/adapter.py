"""Parent-side state adapters for subprocess-backed KV transfers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgspec
import torch
from vllm.distributed.kv_events import BlockStored

from ..kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
)
from ..metadata import LayerLoadTask, LayerTransferTask, ReqMeta


class _ProcessTransferAdapterMixin(KVTransferThread):
    """Keep request bookkeeping local while the child executes the handler."""

    _operation: str

    def __init__(self, *args, process, **kwargs):
        super().__init__(*args, **kwargs)
        self._process = process
        self._pending: set[Future] = set()
        self._pending_condition = threading.Condition()
        self._generations: dict[str, object] = {}

    def add_request(self, request: ReqMeta) -> None:
        self.raise_if_failed()
        req_id, event_id = request.req_id, request.event_id
        with self.done_task_lock:
            generation = self._generations.setdefault(req_id, object())
        future = self._submit_request(request)
        with self._pending_condition:
            self._pending.add(future)
        future.add_done_callback(lambda f: self._complete(f, req_id, event_id, generation))

    def _submit_request(self, request: ReqMeta) -> Future:
        return self._process.submit_request(self._operation, request)

    def _complete(self, future: Future, req_id: str, event_id: int | None, generation: object) -> None:
        try:
            result = future.result()
            if isinstance(self, KVCacheStoreRecvingThread) and self._record_operation_cb is not None:
                for operation, duration_seconds, num_keys in result.get("operations", ()):
                    self._record_operation_cb(operation, duration_seconds, num_keys)
            with self.done_task_lock:
                if generation is self._generations.get(req_id):
                    if isinstance(self, KVCacheStoreRecvingThread):
                        with self._invalid_block_ids_lock:
                            self._invalid_block_ids.update(result["invalid_blocks"])
                    if self._operation == "store":
                        if req_id in self.stored_requests:
                            self.stored_requests[req_id] -= 1
                            if self.stored_requests[req_id] == 0:
                                del self.stored_requests[req_id]
                                self.finished_requests.add(req_id)
                                self._generations.pop(req_id, None)
                    elif result["finished"]:
                        self.finished_requests.add(req_id)
                        self._generations.pop(req_id, None)
            self.update_kv_event([msgspec.convert(item, BlockStored) for item in result["events"]])
            if isinstance(self, KVCacheStoreSendingThread) and event_id is not None:
                with self.completed_events_lock:
                    self.completed_events[event_id] = 1
        except Exception as exc:
            self._fatal_error = exc
        finally:
            with self._pending_condition:
                self._pending.discard(future)
                self._pending_condition.notify_all()

    def discard_finished_requests(self, req_ids: set[str]) -> None:
        with self.done_task_lock:
            self.finished_requests -= req_ids
            for req_id in req_ids:
                self._generations.pop(req_id, None)

    def raise_if_failed(self) -> None:
        self._process.client.raise_if_failed()
        super().raise_if_failed()

    def wait_for_pending(self) -> None:
        deadline = time.monotonic() + self._process.client.timeout
        with self._pending_condition:
            while self._pending:
                self.raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for KV transfer callbacks")
                self._pending_condition.wait(remaining)
        self.raise_if_failed()

    def close(self) -> None:
        self.wait_for_pending()


class KVCacheStoreSendingProcessAdapter(_ProcessTransferAdapterMixin, KVCacheStoreSendingThread):
    """Wait for source readiness locally before starting a child store."""

    _operation = "store"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coordinator = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="KVBlockSave",
            initializer=torch.npu.set_device,
            initargs=(torch.npu.current_device(),),
        )

    def _submit_request(self, request: ReqMeta) -> Future:
        return self._coordinator.submit(self._coordinate_store, request, request.current_event)

    def _coordinate_store(self, request: ReqMeta, event: torch.npu.Event | None) -> dict[str, Any]:
        if event is not None:
            event.synchronize()
        future = self._process.submit_request(self._operation, request)
        return self._process.client.wait(future)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._coordinator.shutdown(wait=True, cancel_futures=True)


class KVCacheStoreRecvingProcessAdapter(_ProcessTransferAdapterMixin, KVCacheStoreRecvingThread):
    """Parent-side receiving state and failed-block reporting."""

    _operation = "load"


class _LayerProcessTransferAdapterMixin(KVTransferThread):
    """Keep layer scheduling state local while the child executes transfers."""

    def __init__(self, *args, process, **kwargs):
        super().__init__(*args, **kwargs)
        self._process = process
        self._pending: set[Future] = set()
        self._pending_condition = threading.Condition()
        self._closing = threading.Event()

    def _track(self, future: Future, complete: Callable[[Any], None]) -> None:
        with self._pending_condition:
            self._pending.add(future)
        future.add_done_callback(lambda item: self._complete(item, complete))

    def _complete(self, future: Future, complete: Callable[[Any], None]) -> None:
        try:
            complete(future.result())
        except BaseException as exc:
            if not self._closing.is_set():
                self._fatal_error = exc
        finally:
            with self._pending_condition:
                self._pending.discard(future)
                self._pending_condition.notify_all()

    def raise_if_failed(self) -> None:
        self._process.client.raise_if_failed()
        super().raise_if_failed()

    def wait_for_pending(self) -> None:
        deadline = time.monotonic() + self._process.client.timeout
        with self._pending_condition:
            while self._pending:
                self.raise_if_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for layerwise transfer callbacks")
                self._pending_condition.wait(remaining)
        self.raise_if_failed()

    def close(self) -> None:
        self.wait_for_pending()
        self._closing.set()


class _LayerSendingProcessAdapterMixin(_LayerProcessTransferAdapterMixin):
    sync_save_events: list[torch.npu.Event]
    layer_save_finished_events: list[threading.Event]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coordinator = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="KVLayerSave",
            initializer=torch.npu.set_device,
            initargs=(torch.npu.current_device(),),
        )

    def add_request(self, transfer_tasks: list[LayerTransferTask]) -> None:
        self.raise_if_failed()
        if not transfer_tasks:
            return
        layer_id = transfer_tasks[0].layer_id
        event = self.sync_save_events[layer_id]
        future = self._coordinator.submit(self._coordinate_store, transfer_tasks, layer_id, event)
        self._track(future, lambda result: self._complete_store(result, transfer_tasks, layer_id))

    def _coordinate_store(
        self,
        transfer_tasks: list[LayerTransferTask],
        layer_id: int,
        event: torch.npu.Event,
    ) -> dict[str, Any]:
        # The event belongs to this Worker process. Waiting here keeps device
        # ordering local while the model thread continues with later layers.
        event.synchronize()
        future = self._process.submit_layer_request("store", transfer_tasks, layer_id)
        return self._process.client.wait(future)

    def _complete_store(self, result: dict[str, Any], transfer_tasks: list[LayerTransferTask], layer_id: int) -> None:
        committed_keys = result.get("committed_keys", ())
        revoked_keys = result.get("revoked_keys", ())
        session_tracker = getattr(self, "_session_tracker", None)
        if session_tracker is not None:
            session_tracker.commit_put_keys(committed_keys)
            session_tracker.revoke_put_keys(revoked_keys)
        remove_started_keys = getattr(self, "_remove_started_keys", None)
        if remove_started_keys is not None:
            remove_started_keys([*committed_keys, *revoked_keys])
        finished_req_ids = set(result["finished_req_ids"])
        for req_id in result["completed_req_ids"]:
            remaining = self.dec_stored_request(req_id)
            if req_id in finished_req_ids and remaining == 0 and self.try_finish_and_delete_stored_request(req_id):
                self.set_finished_request(req_id)
        self.update_kv_event([msgspec.convert(item, BlockStored) for item in result["events"]])
        assert not self.layer_save_finished_events[layer_id].is_set(), f"process: {layer_id} save failed "
        self.layer_save_finished_events[layer_id].set()
        transfer_tasks.clear()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._closing.set()
            self._coordinator.shutdown(wait=True, cancel_futures=True)


class _LayerRecvingProcessAdapterMixin(_LayerProcessTransferAdapterMixin):
    get_event: threading.Event
    layer_load_finished_events: list[threading.Event]
    layer_save_finished_events: list[threading.Event]
    _invalid_block_ids: set[int]
    _invalid_block_ids_lock: threading.Lock
    _load_abort_event: threading.Event

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coordinator = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="KVLayerLoad",
            initializer=torch.npu.set_device,
            initargs=(torch.npu.current_device(),),
        )

    def add_request(self, data: LayerLoadTask) -> None:
        self.raise_if_failed()
        future = self._coordinator.submit(self._coordinate_load, data)
        self._track(future, lambda result: self._complete_load(result, data.layer_id))

    def _coordinate_load(self, data: LayerLoadTask) -> dict[str, Any]:
        wait_for_save = data.wait_for_save_layer
        if wait_for_save is not None:
            while not self.layer_save_finished_events[wait_for_save].wait(timeout=0.1):
                self._raise_if_stopped()
                save_failure_checker = getattr(self, "save_failure_checker", None)
                if save_failure_checker is not None:
                    save_failure_checker()
            save_failure_checker = getattr(self, "save_failure_checker", None)
            if save_failure_checker is not None:
                save_failure_checker()
            sync_save_events = getattr(self, "sync_save_events", None)
            if sync_save_events is not None:
                save_event = sync_save_events[wait_for_save]
                if save_event is not None:
                    save_event.synchronize()
            self.layer_save_finished_events[wait_for_save].clear()

        if data.transfer_tasks:
            attention_start_gate = data.attention_start_gate
            if attention_start_gate is not None:
                while not attention_start_gate.wait(timeout=0.1):
                    self._raise_if_stopped()
            stagger_h2d_submit = getattr(self, "_stagger_h2d_submit", None)
            if stagger_h2d_submit is not None:
                stagger_h2d_submit(data.layer_id)

        external_slot_release_waiter = getattr(self, "external_slot_release_waiter", None)
        if external_slot_release_waiter is not None:
            external_slot_release_waiter(data.layer_id)
        if not data.transfer_tasks:
            return {"completed_req_ids": [], "finished_req_ids": [], "events": []}

        future = self._process.submit_layer_request("load", data.transfer_tasks, data.layer_id)
        return self._process.client.wait(future)

    def _raise_if_stopped(self) -> None:
        if self._closing.is_set():
            raise RuntimeError("Layerwise receive process is closing")
        self.raise_if_failed()

    def _complete_load(self, result: dict[str, Any], layer_id: int) -> None:
        invalid_blocks = result.get("invalid_blocks", ())
        if invalid_blocks:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(invalid_blocks)
        if result.get("load_aborted", False):
            self._load_abort_event.set()
        for req_id in result["finished_req_ids"]:
            self.set_finished_request(req_id)
        self.update_kv_event([msgspec.convert(item, BlockStored) for item in result["events"]])
        assert not self.layer_load_finished_events[layer_id].is_set(), f"process: {layer_id} load failed "
        self.layer_load_finished_events[layer_id].set()
        self.get_event.set()

    def close(self) -> None:
        self._closing.set()
        self._coordinator.shutdown(wait=True, cancel_futures=True)


class KVCacheStoreKeyLayerSendingProcessAdapter(_LayerSendingProcessAdapterMixin, KVCacheStoreKeyLayerSendingThread):
    """Parent-side state for key-based layer saves executed by the child."""

    def build_cached_process_tokens(self, task: LayerTransferTask):
        # Key generation belongs in the transfer process when this mode is on.
        return None


class KVCacheStoreKeyLayerRecvingProcessAdapter(_LayerRecvingProcessAdapterMixin, KVCacheStoreKeyLayerRecvingThread):
    """Parent-side state for key-based layer loads executed by the child."""


class KVCacheStoreLayerSendingProcessAdapter(_LayerSendingProcessAdapterMixin, KVCacheStoreLayerSendingThread):
    """Parent-side state for GVA layer saves executed by the child."""


class KVCacheStoreLayerRecvingProcessAdapter(_LayerRecvingProcessAdapterMixin, KVCacheStoreLayerRecvingThread):
    """Parent-side state for GVA layer loads executed by the child."""


__all__ = [
    "KVCacheStoreKeyLayerRecvingProcessAdapter",
    "KVCacheStoreKeyLayerSendingProcessAdapter",
    "KVCacheStoreLayerRecvingProcessAdapter",
    "KVCacheStoreLayerSendingProcessAdapter",
    "KVCacheStoreRecvingProcessAdapter",
    "KVCacheStoreSendingProcessAdapter",
]
