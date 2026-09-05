"""Private interpreter entry point for an engine-owned KV transfer process."""

from __future__ import annotations

import os
import queue
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
from zmq import DONTWAIT, LINGER, POLLIN, ROUTER, ROUTER_MANDATORY, Context, Poller  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from ..kv_transfer import KVCacheStoreRecvingThread, KVCacheStoreSendingThread
    from .npu_ipc import ImportedKVCache


def run_worker(endpoint: str, parent_fd: int, runtime_factory=None) -> int:
    """Keep control and parent-death detection responsive during backend IO."""
    context = Context()
    socket = context.socket(ROUTER)
    socket.setsockopt(LINGER, 0)
    socket.setsockopt(ROUTER_MANDATORY, 1)
    completions: queue.Queue = queue.Queue()
    control = ThreadPoolExecutor(max_workers=1, thread_name_prefix="KVTransferBackend")
    runtime = None
    closing = False
    closed_exit_code = None

    def initialize(config):
        nonlocal runtime
        factory = TransferRuntime if runtime_factory is None else runtime_factory
        runtime = factory(config)

    def close_runtime():
        if runtime is not None:
            runtime.close()

    def completed(future, identity, operation_id, operation):
        exit_code = 0
        try:
            response = (operation_id, future.result(), None)
            encoded = msgspec.msgpack.encode(response)
        except BaseException as exc:
            exit_code = 1
            encoded = msgspec.msgpack.encode((operation_id, None, f"{type(exc).__name__}: {exc}"))
        completions.put((identity, encoded, operation == "close", exit_code))

    try:
        socket.bind(endpoint)
        poller = Poller()
        poller.register(socket, POLLIN)
        poller.register(parent_fd, POLLIN)
        while True:
            events = dict(poller.poll(10))
            if parent_fd in events and not os.read(parent_fd, 1):
                return 1 if closed_exit_code is None else closed_exit_code
            if socket in events:
                identity, encoded = socket.recv_multipart()
                operation_id, operation, payload = msgspec.msgpack.decode(encoded)
                try:
                    if closing:
                        raise RuntimeError("KV transfer process is closing")
                    if operation == "init":
                        if runtime is not None:
                            raise RuntimeError("KV transfer process is already initialized")
                        future = control.submit(initialize, payload)
                    elif operation == "close":
                        closing = True
                        future = control.submit(close_runtime)
                    elif runtime is None:
                        raise RuntimeError("KV transfer process is not initialized")
                    elif operation in ("store", "load"):
                        future = runtime.submit(operation, payload)
                    else:
                        future = control.submit(runtime.execute, operation, payload)
                except Exception as exc:
                    future = Future()
                    future.set_exception(exc)
                future.add_done_callback(
                    partial(completed, identity=identity, operation_id=operation_id, operation=operation)
                )
            while not completions.empty():
                identity, encoded, is_close, exit_code = completions.get_nowait()
                socket.send_multipart((identity, encoded), DONTWAIT)
                if is_close:
                    # send() only queues a reply in ZMQ. Keep the socket alive
                    # until the parent consumes CLOSED and releases its pipe;
                    # exiting here can lose that final completion record.
                    closed_exit_code = exit_code
    finally:
        # On parent death a device operation may never finish. The private
        # entry point exits the interpreter directly after this loop returns.
        control.shutdown(wait=False, cancel_futures=True)
        socket.close()
        context.term()


class TransferRuntime:
    """Own the backend and imported buffers; reuse the existing transfer handlers."""

    def __init__(self, config: dict[str, Any]):
        # Import device code only inside the fresh child, never at entry-point
        # discovery time (CPU lifecycle tests provide their own runtime).
        import torch

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.transfer_backend import (
            create_transfer_backend,
        )

        self.config = config
        self.device_index = config["device_index"]
        torch.npu.set_device(self.device_index)
        self.backend = create_transfer_backend(config["backend"], self.device_index)
        self.cache: ImportedKVCache | None = None
        self.sender: KVCacheStoreSendingThread | None = None
        self.receiver: KVCacheStoreRecvingThread | None = None
        self._send = ThreadPoolExecutor(max_workers=1, initializer=self.backend.set_device)
        self._recv = ThreadPoolExecutor(max_workers=1, initializer=self.backend.set_device)

    def execute(self, operation: str, payload: Any) -> Any:
        self.backend.set_device()
        if operation == "register":
            return self.register(payload)
        if operation == "exists":
            return self.backend.exists(payload)
        if operation == "ensure_ready":
            ensure = getattr(self.backend, "ensure_initialized", None)
            if ensure is not None:
                ensure()
            return None
        if operation == "get_ranges":
            if self.cache is None:
                raise RuntimeError("KV caches are not registered")
            keys, ranges = payload
            addresses, sizes = [], []
            for row in ranges:
                addresses.append([self.cache.resolve_range(*item) for item in row])
                sizes.append([item[2] for item in row])
            return self.backend.get(keys, addresses, sizes)
        raise ValueError(f"Unknown KV transfer operation: {operation}")

    def register(self, payload: dict[str, Any]) -> None:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
            KVCacheStoreRecvingThread,
            KVCacheStoreSendingThread,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase, KeyMetadata
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
            WorkerKVCacheSpec,
            import_worker_kv_caches,
        )

        if self.cache is not None:
            raise RuntimeError("KV caches are already registered")
        cache = import_worker_kv_caches(msgspec.convert(payload["cache"], WorkerKVCacheSpec))
        self.cache = cache
        if cache.device_index != self.device_index:
            raise ValueError("Imported KV caches do not match the transfer backend device")
        addresses = {
            int(group): [cache.resolve_range(*item) for item in ranges]
            for group, ranges in payload["group_ranges"].items()
        }
        database = ChunkedTokenDatabase(
            [KeyMetadata(**item) for item in payload["metadata"]],
            payload["block_sizes"],
            payload["partitions"],
            hash_block_size=payload["hash_block_size"],
        )
        database.set_group_buffers(
            addresses,
            payload["block_lengths"],
            payload["block_strides"],
            group_cache_families=payload["families"],
            group_num_layers=payload["num_layers"],
            group_layer_cache_entry_offsets=payload["entry_offsets"],
        )
        ranges = payload["registered_ranges"]
        self.backend.register_buffer([cache.resolve_range(*item) for item in ranges], [item[2] for item in ranges])
        common = dict(
            m_store=self.backend,
            token_database=database,
            block_size=payload["block_sizes"],
            tp_rank=self.config["tp_rank"],
            tp_size=self.config["tp_size"],
            dcp_size=self.config["dcp_size"],
        )
        self.sender = KVCacheStoreSendingThread(
            **common,
            put_step=self.config["put_step"],
            kv_role=self.config["kv_role"],
            group_uses_align_state=payload["align_state"],
            enable_kv_event=self.config["enable_kv_events"],
        )
        self.receiver = KVCacheStoreRecvingThread(**common)

    def submit(self, operation: str, payload: Any) -> Future:
        if self.cache is None or self.sender is None or self.receiver is None:
            raise RuntimeError("KV caches are not registered")
        executor = self._send if operation == "store" else self._recv
        return executor.submit(self._transfer, operation, payload)

    def _transfer(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.npu_ipc import (
            NPUEventSpec,
            import_npu_event,
        )

        event = payload.pop("current_event")
        if payload["load_spec"] is not None:
            payload["load_spec"] = LoadSpec(**payload["load_spec"])
        request = ReqMeta(**payload)
        if event is not None:
            request.current_event = import_npu_event(msgspec.convert(event, NPUEventSpec))
        worker = self.sender if operation == "store" else self.receiver
        assert worker is not None
        if operation == "store":
            worker.add_stored_request(request.req_id)
        # The original handlers own task_done and all key/address/IO logic.
        # Their thread objects are local state holders; these executors are
        # their execution threads in this interpreter.
        worker.request_queue.put(request)
        worker.request_queue.get_nowait()
        worker._handle_request(request)
        finished = worker.get_and_clear_finished_requests()
        result = {"finished": request.req_id in finished, "events": worker.get_kv_events(), "invalid_blocks": []}
        if operation == "load":
            receiver = self.receiver
            assert receiver is not None
            with receiver._invalid_block_ids_lock:
                result["invalid_blocks"] = list(receiver._invalid_block_ids)
                receiver._invalid_block_ids.clear()
        return result

    def close(self) -> None:
        self._send.shutdown(wait=True)
        self._recv.shutdown(wait=True)
        self.backend.set_device()
        # Unregister before releasing imported allocations. If cleanup fails,
        # keep those references until the interpreter exits.
        self.backend.close()
        if self.cache is not None:
            self.cache.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    exit_code = run_worker(sys.argv[1], int(sys.argv[2]))
    os._exit(exit_code)
