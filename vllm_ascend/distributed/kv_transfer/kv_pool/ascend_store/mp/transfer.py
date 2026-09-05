"""A worker-owned transfer process with one owner for its ZMQ socket.

The socket carries commands and completion records only. Allocation references
stay in the model worker until the child has finished using their IPC handles.
An operation timeout makes the entire channel unusable: a timed-out write may
already have executed, so it must never be transparently replayed.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msgspec
from zmq import DEALER, DONTWAIT, IMMEDIATE, LINGER, SNDHWM, Again, Context  # type: ignore[attr-defined]

POLL_INTERVAL_MS = 10
MAX_PENDING_TRANSFERS = 256
TRANSFER_TIMEOUT_SECONDS = 120.0
PROCESS_EXIT_TIMEOUT_SECONDS = 5.0


@dataclass
class _Command:
    operation_id: int
    operation: str
    payload: bytes
    future: Future
    deadline: float


class TransferChannel:
    """Own a child interpreter and fail all waiters if it dies or times out."""

    def __init__(self, *, timeout: float = TRANSFER_TIMEOUT_SECONDS, command: list[str] | None = None):
        if timeout <= 0:
            raise ValueError("Transfer timeout must be positive")
        self.timeout = timeout
        self._commands: queue.Queue[_Command] = queue.Queue(MAX_PENDING_TRANSFERS)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._error: BaseException | None = None
        self._next_id = 0
        self._stop = threading.Event()
        self._ready: Future = Future()
        self._directory = tempfile.TemporaryDirectory(prefix="kv-")
        endpoint = f"ipc://{self._directory.name}/worker"
        parent_read, parent_write = os.pipe()
        self._parent_write: int | None = parent_write
        if command is None:
            command = [sys.executable, str(Path(__file__).with_name("worker.py"))]
        try:
            # vLLM model workers can be multiprocessing daemons. Popen starts
            # a fresh interpreter without multiprocessing's nested-child ban.
            self.process = subprocess.Popen(
                [*command, endpoint, str(parent_read)], pass_fds=(parent_read,), close_fds=True
            )
        except BaseException:
            os.close(parent_write)
            self._directory.cleanup()

            raise
        finally:
            os.close(parent_read)
        self._io = threading.Thread(target=self._run, args=(endpoint,), name="KVTransferControl", daemon=True)
        self._io.start()
        try:
            self._ready.result(timeout)
        except BaseException:
            self.close()
            raise

    def submit(self, operation: str, payload: Any = None) -> Future:
        with self._lock:
            self.raise_if_failed()
            if self._closed:
                raise RuntimeError("KV transfer process is closed")
            return self._enqueue(operation, payload)

    def _enqueue(self, operation: str, payload: Any = None) -> Future:
        self._next_id += 1
        future: Future = Future()
        command = _Command(
            self._next_id,
            operation,
            msgspec.msgpack.encode((self._next_id, operation, payload)),
            future,
            time.monotonic() + self.timeout,
        )
        try:
            self._commands.put_nowait(command)
        except queue.Full as exc:
            raise RuntimeError("KV transfer command queue is full") from exc
        return future

    def call(self, operation: str, payload: Any = None) -> Any:
        return self.wait(self.submit(operation, payload))

    def wait(self, future: Future) -> Any:
        try:
            return future.result(self.timeout)
        except TimeoutError as exc:
            self._fail(exc)
            raise RuntimeError("KV transfer operation timed out; the process must be closed") from exc

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("KV transfer process failed") from self._error

    def _fail(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error
        self._stop.set()

    def _run(self, endpoint: str) -> None:
        context = None
        socket = None
        pending: dict[int, _Command] = {}
        outgoing: _Command | None = None
        try:
            context = Context()
            socket = context.socket(DEALER)
            socket.setsockopt(LINGER, 0)
            socket.setsockopt(SNDHWM, MAX_PENDING_TRANSFERS)
            socket.setsockopt(IMMEDIATE, 1)
            socket.connect(endpoint)
            self._ready.set_result(None)
            while not self._stop.is_set():
                while len(pending) < MAX_PENDING_TRANSFERS:
                    if outgoing is None:
                        try:
                            outgoing = self._commands.get_nowait()
                        except queue.Empty:
                            break
                    if time.monotonic() >= outgoing.deadline:
                        raise TimeoutError("Timed out submitting a KV transfer command")
                    try:
                        socket.send(outgoing.payload, DONTWAIT)
                    except Again:
                        break
                    else:
                        pending[outgoing.operation_id] = outgoing
                        outgoing = None
                readable = socket.poll(POLL_INTERVAL_MS)
                while readable:
                    operation_id, result, error = msgspec.msgpack.decode(socket.recv())
                    completed = pending.pop(operation_id, None)
                    if completed is None:
                        raise RuntimeError(f"Received completion for unknown KV transfer operation {operation_id}")
                    if completed.operation == "close":
                        self._stop.set()
                    if error is not None:
                        completed.future.set_exception(RuntimeError(error))
                    else:
                        completed.future.set_result(result)
                    readable = socket.poll(0)
                if not self._stop.is_set() and self.process.poll() is not None:
                    # Read queued replies before interpreting process exit;
                    # the final CLOSED reply may already be in the socket.
                    if not socket.poll(0):
                        raise RuntimeError(f"KV transfer subprocess exited with code {self.process.returncode}")
                if any(time.monotonic() >= item.deadline for item in pending.values()):
                    raise TimeoutError("Timed out waiting for KV transfer completion")
        except BaseException as exc:
            self._fail(exc)
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            with self._lock:
                self._closed = True
                if outgoing is not None:
                    pending[outgoing.operation_id] = outgoing
                while not self._commands.empty():
                    item = self._commands.get_nowait()
                    pending[item.operation_id] = item
            for item in pending.values():
                if not item.future.done():
                    item.future.set_exception(RuntimeError(f"KV transfer channel stopped: {self._error}"))
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()

    def close(self) -> None:
        with self._close_lock:
            self._close()

    def _close(self) -> None:
        with self._lock:
            closed = None
            if not self._closed and self._error is None:
                try:
                    closed = self._enqueue("close")
                except RuntimeError as exc:
                    self._fail(exc)
            self._closed = True
        try:
            if closed is not None:
                self.wait(closed)
            elif self._error is not None:
                self.raise_if_failed()
        finally:
            self._stop.set()
            self._io.join(PROCESS_EXIT_TIMEOUT_SECONDS)
            if self._parent_write is not None:
                os.close(self._parent_write)
                self._parent_write = None
            try:
                self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(PROCESS_EXIT_TIMEOUT_SECONDS)
            self._directory.cleanup()


class KVTransferProcess:
    """Expose the few parent-side operations that need the child-owned backend."""

    def __init__(self, config: dict[str, Any]):
        self.channel = TransferChannel()
        self.cache: Any = None
        self._events: dict[Future, Any] = {}
        self._event_lock = threading.Lock()
        self._device_uuid: str | None = None
        try:
            self.channel.call("init", config)
        except BaseException:
            self.channel.close()
            raise

    def register_kv_caches(self, worker, kv_caches, pointers, lengths) -> None:
        from .npu_ipc import export_worker_kv_caches

        if self.cache is not None:
            raise RuntimeError("KV caches are already registered")
        self.cache = export_worker_kv_caches(kv_caches)
        self._device_uuid = self.cache.spec.storages[0].device_uuid
        database = worker.token_database
        payload = dict(
            cache=self.cache.spec,
            registered_ranges=[self.cache.describe_range(p, n) for p, n in zip(pointers, lengths)],
            group_ranges={
                group: [self.cache.describe_range(address, 0) for address in addresses]
                for group, addresses in worker.group_kv_caches_base_addr.items()
            },
            metadata=[asdict(item) for item in database.metadata],
            block_sizes=database.block_size,
            partitions=database.partitions,
            hash_block_size=database.hash_block_size,
            block_lengths=worker.group_block_len,
            block_strides=worker.group_block_stride,
            families=worker.group_kv_cache_families,
            num_layers=worker.group_num_layers,
            entry_offsets=worker.group_layer_cache_entry_offsets,
            align_state=worker.group_uses_align_state,
        )
        try:
            self.channel.call("register", payload)
        except BaseException:
            self.close()
            raise

    def submit_request(self, operation: str, request) -> Future:
        from .npu_ipc import NPUEventSpec

        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        event = request.current_event
        event_spec = None
        if event is not None:
            assert self._device_uuid is not None
            event_spec = NPUEventSpec(self._device_uuid, event.ipc_handle())
        # Snapshot only fields consumed by the ordinary transfer handlers.
        # ReqMeta also contains layerwise arrays/GVAs and a live device event;
        # serializing the whole object would leak unrelated process state.
        payload = dict(
            req_id=request.req_id,
            save_end_token=request.save_end_token,
            target_token_len=request.target_token_len,
            save_start_token=request.save_start_token,
            block_ids_by_group=request.block_ids_by_group,
            block_hashes=request.block_hashes,
            can_save=request.can_save,
            load_spec=None if request.load_spec is None else asdict(request.load_spec),
            is_last_chunk=request.is_last_chunk,
            current_event=event_spec,
            kv_cache_group_ids=request.kv_cache_group_ids,
            skip_null_blocks_by_group=request.skip_null_blocks_by_group,
            num_prompt_tokens=request.num_prompt_tokens,
            token_ids=request.token_ids,
            original_block_size=request.original_block_size,
            event_id=request.event_id,
        )
        future = self.channel.submit(operation, payload)
        if event is not None:
            with self._event_lock:
                self._events[future] = event
            future.add_done_callback(self._release_event)
        return future

    def _release_event(self, future: Future) -> None:
        # An error/timeout is not proof that device IO stopped. In that case
        # keep the event until close has reaped the child interpreter.
        if future.exception() is None:
            with self._event_lock:
                self._events.pop(future, None)

    def exists(self, keys):
        return self.channel.call("exists", keys)

    def ensure_initialized(self) -> None:
        self.channel.call("ensure_ready")

    def get(self, keys, addresses, sizes):
        if self.cache is None:
            raise RuntimeError("KV caches are not registered")
        if len(keys) != len(addresses) or len(keys) != len(sizes):
            raise ValueError("Keys, addresses and sizes must have equal lengths")
        ranges = []
        for row, row_sizes in zip(addresses, sizes):
            if len(row) != len(row_sizes):
                raise ValueError("Each buffer address needs one size")
            ranges.append([self.cache.describe_range(pointer, size) for pointer, size in zip(row, row_sizes)])
        return self.channel.call("get_ranges", (keys, ranges))

    def close(self) -> None:
        try:
            self.channel.close()
        finally:
            # TransferChannel.close reaps the process before returning, even
            # after a failed drain. No child can access these handles now.
            if self.channel.process.poll() is not None:
                with self._event_lock:
                    self._events.clear()
                if self.cache is not None:
                    self.cache.close()
