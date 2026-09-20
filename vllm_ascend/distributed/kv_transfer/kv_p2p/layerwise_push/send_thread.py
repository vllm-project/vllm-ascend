# SPDX-License-Identifier: Apache-2.0
"""P-side control thread for backend-independent layerwise push."""

from __future__ import annotations

import queue
import socket
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.protocol import (
    LAYOUT_META,
    REQUEST_DONE,
    TARGET_BLOCKS,
    WRITE_FAILED,
    ComponentLayout,
    SendTask,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread import (
    ConsumerDestinationState,
    WriteBackend,
    WritePlanner,
)

THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0
WRITE_WORKERS = 2
MAX_PENDING_BATCHES = 64
SOURCE_EVENT_POLL_MS = 1
ENQUEUE_RETRY_SECONDS = 0.1
HANDSHAKE_TIMEOUT_SECONDS = 10.0
TransferId = tuple[int, str]


class SlotReuseTracker:
    """Track in-flight transfers for source slots reused by another layer."""

    def __init__(self, layer_slots: dict[int, tuple[int, ...]]) -> None:
        use_count = Counter(slot for slots in layer_slots.values() for slot in set(slots))
        self.reused_slots = frozenset(slot for slot, count in use_count.items() if count > 1)
        slot_count = max(use_count, default=-1) + 1
        self.events = [threading.Event() for _ in range(slot_count)]
        for event in self.events:
            event.set()
        self.errors: dict[int, str] = {}
        self._transfers: dict[int, set[TransferId]] = {slot: set() for slot in self.reused_slots}
        self._transfer_slots: dict[TransferId, tuple[int, ...]] = {}
        self._lock = threading.Lock()

    def begin(self, reader: TransferId, touched_slots: Iterable[int]) -> None:
        slots = tuple(dict.fromkeys(slot for slot in touched_slots if slot in self.reused_slots))
        if not slots:
            return
        with self._lock:
            self._transfer_slots[reader] = slots
            for slot in slots:
                readers = self._transfers[slot]
                if not readers:
                    self.errors.pop(slot, None)
                    self.events[slot].clear()
                readers.add(reader)

    def complete(self, reader: TransferId, error: str | None = None) -> None:
        with self._lock:
            slots = self._transfer_slots.pop(reader, ())
            for slot in slots:
                readers = self._transfers[slot]
                readers.discard(reader)
                if error is not None:
                    self.errors[slot] = error
                if not readers:
                    self.events[slot].set()

    def fail_all(self, error: str) -> None:
        with self._lock:
            readers = tuple(self._transfer_slots)
        for reader in readers:
            self.complete(reader, error)

    def event(self, slot: int) -> threading.Event | None:
        return self.events[slot] if 0 <= slot < len(self.events) else None

    def error(self, slot: int) -> str | None:
        return self.errors.get(slot)


@dataclass
class ProducerSendState:
    last_layer_idx: int
    layer_layouts: dict[int, tuple[ComponentLayout, ...]]
    p_session: str
    block_sizes: tuple[int, ...]
    layer_storage_slots: dict[int, tuple[int, ...]]
    num_blocks: int = 0
    pp_rank: int = 0
    tp_rank: int = 0
    tp_size: int = 1
    pp_layers: tuple[tuple[int, ...], ...] = ()
    write_mode: str = "async"


class LayerwisePushSendingThread(threading.Thread):
    """P control thread with bounded, per-endpoint ordered WRITE submission."""

    def __init__(
        self,
        *,
        ready_event: threading.Event,
        state: ProducerSendState,
        backend: WriteBackend,
    ) -> None:
        super().__init__(daemon=True, name="LayerwisePushSendingThread")
        self._layout_meta_sent_paths: set[str] = set()
        self._state = state
        self.last_layer_idx = state.last_layer_idx
        self.ready_event = ready_event
        self.send_queue: queue.Queue[SendTask] = queue.Queue(maxsize=MAX_PENDING_BATCHES)
        self.backend = backend
        self._pending: dict[str, deque] = {}
        self._planners: dict[str, WritePlanner] = {}
        self._requested_targets: dict[str, set[str]] = {}
        self._active: dict[str, tuple[Any, tuple]] = {}
        self._handshake_deadlines: dict[str, float] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._push_stream: Any | None = None
        self._waiting_for_event = False
        self.fatal_error: BaseException | None = None
        # A native socket pair lets queue producers wake the same poller that
        # handles D replies, without sharing a ZeroMQ socket across threads.
        self._task_reader, self._task_writer = socket.socketpair()
        self._task_reader.setblocking(False)
        self._task_writer.setblocking(False)
        self._poller = zmq.Poller()
        self._poller.register(self._task_reader, zmq.POLLIN)
        self._persist_ctx = zmq.Context()  # type: ignore[attr-defined]
        self._dealers: dict[str, Any] = {}
        self._stopped = False
        self.startup_error: BaseException | None = None
        self._next_transfer_id = 0
        self._next_reservation_id = -1
        self._reservation_lock = threading.Lock()
        self._reuse_tracker = SlotReuseTracker(state.layer_storage_slots)
        # Keep these attributes for the worker-facing compatibility methods.
        self.storage_send_done_events = self._reuse_tracker.events
        self._storage_write_errors = self._reuse_tracker.errors
        self._request_completion_lock = threading.Lock()
        self._pending_completion_requests: set[str] = set()
        self._scheduler_finished_requests: set[str] = set()
        self._completed_requests: set[str] = set()
        self._completion_requests_by_transfer: dict[tuple[int, str], set[str]] = {}
        self._pending_completion_paths: dict[str, set[str]] = {}

    def enqueue(self, task: SendTask) -> None:
        if self.fatal_error is not None or self._stopped:
            raise RuntimeError("Push sender is unavailable") from self.fatal_error
        with self._reservation_lock:
            reservation_id = (self._next_reservation_id, "queued")
            self._next_reservation_id -= 1
        task.reservation_id = reservation_id
        self._ensure_reuse_tracker().begin(
            reservation_id,
            self._state.layer_storage_slots.get(task.layer_idx, ()),
        )
        try:
            while True:
                if self.fatal_error is not None or self._stopped:
                    raise RuntimeError("Push sender is unavailable") from self.fatal_error
                try:
                    self.send_queue.put(task, timeout=ENQUEUE_RETRY_SECONDS)
                    break
                except queue.Full:
                    continue
        except BaseException as error:
            self._ensure_reuse_tracker().complete(reservation_id, str(error))
            raise
        # A full socket already has an unread wakeup; the task stays queued.
        with suppress(BlockingIOError):
            self._task_writer.send(b"\x00")

    def track_requests(self, request_ids: set[str], endpoints: set[tuple[str, int]]) -> None:
        with self._request_completion_lock:
            self._pending_completion_requests.update(request_ids)
            for request_id in request_ids:
                self._pending_completion_paths[request_id] = {
                    make_zmq_path("tcp", host, port) for host, port in endpoints
                }

    def get_and_clear_finished_requests(self, scheduler_finished_req_ids: set[str]) -> set[str]:
        if self.fatal_error is not None:
            raise RuntimeError("Push sender failed; source blocks remain pinned") from self.fatal_error
        with self._request_completion_lock:
            self._scheduler_finished_requests.update(scheduler_finished_req_ids & self._pending_completion_requests)
            finished = self._completed_requests & self._scheduler_finished_requests
            self._completed_requests.difference_update(finished)
            self._scheduler_finished_requests.difference_update(finished)
            self._pending_completion_requests.difference_update(finished)
            return finished

    def _complete_transfer(self, reader: tuple[int, str], error: str | None = None) -> None:
        self._ensure_reuse_tracker().complete(reader, error)
        request_ids = self._completion_requests_by_transfer.pop(reader, ())
        if error is not None or not request_ids:
            return
        with self._request_completion_lock:
            for request_id in request_ids:
                remaining = self._pending_completion_paths.get(request_id)
                if remaining is not None:
                    remaining.discard(reader[1])
                    if remaining:
                        continue
                    self._pending_completion_paths.pop(request_id)
                    self._completed_requests.add(request_id)

    def _ensure_dealer(self, path: str):
        if path not in self._dealers:
            dealer = self._persist_ctx.socket(zmq.DEALER)  # type: ignore[attr-defined]
            dealer.setsockopt(zmq.LINGER, 0)  # type: ignore[attr-defined]
            dealer.setsockopt(zmq.SNDHWM, 0)  # type: ignore[attr-defined]
            dealer.setsockopt(zmq.RCVHWM, 0)  # type: ignore[attr-defined]
            dealer.connect(path)
            self._dealers[path] = dealer
            self._poller.register(dealer, zmq.POLLIN)
        return self._dealers[path]

    def _wake(self, _future=None) -> None:
        with suppress(OSError):
            self._task_writer.send(b"\x00")

    def run(self) -> None:
        try:
            from vllm.distributed import get_world_group

            device = torch.device(f"npu:{get_world_group().local_rank}")
            torch.npu.set_device(device)
            if self._state.write_mode == "sync":
                self._executor = ThreadPoolExecutor(
                    max_workers=WRITE_WORKERS,
                    thread_name_prefix="LayerwiseWrite",
                    initializer=torch.npu.set_device,
                    initargs=(device,),
                )
            self.ready_event.set()
            encoder = msgspec.msgpack.Encoder()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while not self._stopped:
                self._drain_replies(decoder)
                self._check_handshake_deadlines()
                self._dispatch_writes(encoder)
                pending_count = sum(len(tasks) for tasks in self._pending.values()) + len(self._active)
                if pending_count < MAX_PENDING_BATCHES:
                    try:
                        self._process_send_task(self.send_queue.get_nowait(), encoder)
                        continue
                    except queue.Empty:
                        pass
                self._poller.poll(self._poll_timeout_ms())
                with suppress(BlockingIOError):
                    self._task_reader.recv(4096)
        except BaseException as error:
            self.fatal_error = error
            if not self.ready_event.is_set():
                self.startup_error = error
            logger.exception("Push sender failed; buffers must not be recycled")
            for dealer in self._dealers.values():
                with suppress(Exception):
                    dealer.send(msgspec.msgpack.encode((WRITE_FAILED, str(error))))
        finally:
            self.ready_event.set()
            self._ensure_reuse_tracker().fail_all("Push sender stopped")
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            for dealer in self._dealers.values():
                dealer.close(linger=0)
            self._task_reader.close()
            self._task_writer.close()
            self._persist_ctx.destroy(linger=0)

    def _dispatch_writes(self, encoder) -> None:
        # Only this control thread owns ZeroMQ sockets and completion maps.
        self._waiting_for_event = False
        for path, (completion, batch) in list(self._active.items()):
            if self._state.write_mode == "async":
                if not completion.query():
                    self._waiting_for_event = True
                    continue
            elif not completion.done():
                continue
            else:
                completion.result()  # A timeout does not prove DMA stopped.
            del self._active[path]
            transfer_id, _, _, done_ext_ids, _, _, _ = batch
            if done_ext_ids:
                # This endpoint is ordered: a request's terminal batch finishes
                # only after all its earlier layers/chunks have been written.
                # D needs this notification to start decode, but P already
                # knows the synchronous WRITEs finished and needs no ACK.
                self._dealers[path].send(encoder.encode((REQUEST_DONE, done_ext_ids)))
                for ext_id in done_ext_ids:
                    self._planners[path]._state.dest_blocks_by_req.pop(ext_id, None)
                    self._requested_targets[path].discard(ext_id)
            self._complete_transfer((transfer_id, path))
            # Put this endpoint behind peers when it has more queued batches.
            if path in self._pending:
                self._pending[path] = self._pending.pop(path)

        active_limit = MAX_PENDING_BATCHES if self._state.write_mode == "async" else WRITE_WORKERS
        for path, tasks in list(self._pending.items()):
            if len(self._active) >= active_limit:
                break
            if not tasks or path in self._active or path not in self._planners:
                continue
            planner = self._planners[path]
            batch = tasks[0]
            transfer_id, layer_idx, write_reqs, done_ext_ids, member, ratio, event = batch
            if event is not None and not event.query():
                # No device event FD exists; use a bounded poll while waiting.
                self._waiting_for_event = True
                continue
            if any(entry[0] not in planner._state.dest_blocks_by_req for entry in write_reqs):
                continue
            if any(ext_id not in planner._state.dest_blocks_by_req for ext_id in done_ext_ids):
                continue
            tasks.popleft()
            if not tasks:
                del self._pending[path]
            if self._state.write_mode == "async":
                source_addrs, destination_addrs, lengths = planner.plan_batch(
                    path,
                    layer_idx,
                    write_reqs,
                    self._state.layer_layouts,
                    member,
                    ratio,
                )
                stream = self._ensure_push_stream()
                planner.backend.write_async(
                    planner._state.session,
                    source_addrs,
                    destination_addrs,
                    lengths,
                    stream.npu_stream,
                )
                # Record even for an empty terminal batch: the event remains
                # ordered behind earlier writes submitted on this stream.
                event = torch.npu.Event()
                event.record(stream)
                self._active[path] = (event, batch)
                self._waiting_for_event = True
            else:
                assert self._executor is not None
                future: Future = self._executor.submit(
                    planner.write_batch,
                    path,
                    layer_idx,
                    write_reqs,
                    planner._state.session,
                    self._state.layer_layouts,
                    member,
                    ratio,
                )
                self._active[path] = (future, batch)
                future.add_done_callback(self._wake)

    def _ensure_push_stream(self) -> Any:
        if self._push_stream is None:
            self._push_stream = torch.npu.Stream()
        return self._push_stream

    def _check_handshake_deadlines(self) -> None:
        now = time.monotonic()
        expired = [path for path, deadline in self._handshake_deadlines.items() if deadline <= now]
        if expired:
            raise TimeoutError(f"Timed out waiting for layerwise push layout handshake from {expired[0]}")

    def _poll_timeout_ms(self) -> int | None:
        timeout = SOURCE_EVENT_POLL_MS if self._waiting_for_event else None
        if self._handshake_deadlines:
            remaining = min(self._handshake_deadlines.values()) - time.monotonic()
            handshake_timeout = max(0, int(remaining * 1000))
            timeout = handshake_timeout if timeout is None else min(timeout, handshake_timeout)
        return timeout

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        self._stopped = True
        # The poller may already be awake, or the thread may have exited.
        with suppress(OSError):
            self._task_writer.send(b"\x00")
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("Layerwise push send thread did not stop within %.1f seconds", timeout)

    def _ensure_reuse_tracker(self) -> SlotReuseTracker:
        tracker = getattr(self, "_reuse_tracker", None)
        if tracker is None:
            tracker = SlotReuseTracker(self._state.layer_storage_slots)
            existing_events = getattr(self, "storage_send_done_events", None)
            if existing_events is not None and len(existing_events) == len(tracker.events):
                tracker.events = existing_events
            tracker.errors = getattr(self, "_storage_write_errors", {})
            self._reuse_tracker = tracker
            self.storage_send_done_events = tracker.events
            self._storage_write_errors = tracker.errors
        return tracker

    def _process_send_task(self, send_task: SendTask, encoder: msgspec.msgpack.Encoder) -> None:
        transfer_id = getattr(self, "_next_transfer_id", 0)
        self._next_transfer_id = transfer_id + 1
        layer_idx = send_task.layer_idx
        wait_event = send_task.wait_event
        layer_name = send_task.layer_name

        # Requests for the same D endpoint share one notification. Each request
        # carries source block ids and destination offsets for every KV group;
        # components select their group through ComponentLayout.group_index.
        endpoint_payloads: dict[
            tuple[str, int],
            tuple[list[tuple[str, list[list[int]], list[int]]], list[str], list[str]],
        ] = {}
        endpoint_contributors: dict[tuple[str, int], tuple[int, int]] = {}
        layer_layouts = self._state.layer_layouts[layer_idx]
        layer_group_indices = {layout.group_index for layout in layer_layouts}

        for req_id, rm in send_task.send_request.items():
            if rm.chunk_start_blocks is None:
                raise RuntimeError(f"Layerwise push request {req_id} is missing its precomputed chunk block range")
            source_blocks = [[] for _ in self._state.block_sizes]
            start_blocks = [0 for _ in self._state.block_sizes]
            for group_idx in layer_group_indices:
                source_blocks[group_idx] = rm.local_block_ids[group_idx]
                start_blocks[group_idx] = rm.chunk_start_blocks[group_idx]
            ext_id = get_external_request_id(req_id)
            endpoint = rm.layer_endpoints.get(layer_idx)
            chunk_done = layer_idx in rm.terminal_layers and rm.chunk_finish and endpoint is not None
            has_blocks = any(source_blocks[group_idx] for group_idx in layer_group_indices)
            if (has_blocks or chunk_done) and endpoint is not None:
                endpoint_contributors[endpoint] = (rm.group_member_idx, rm.tp_ratio)
                write_reqs, done_ext_ids, done_req_ids = endpoint_payloads.setdefault(endpoint, ([], [], []))
                if has_blocks:
                    write_reqs.append((ext_id, source_blocks, start_blocks))
                if chunk_done:
                    done_ext_ids.append(ext_id)
                    done_req_ids.append(req_id)
            logger.debug(
                "Layerwise push P prepared layer=%d (%s), req=%s, blocks_by_group=%s, done=%s",
                layer_idx,
                layer_name,
                ext_id,
                [len(group) for group in source_blocks],
                chunk_done,
            )

        if endpoint_payloads:
            registered_transfers: list[tuple[int, str]] = []
            sent_transfers: set[tuple[int, str]] = set()
            try:
                for (remote_host, remote_port), (write_reqs, done_ext_ids, done_req_ids) in endpoint_payloads.items():
                    group_member_idx, tp_ratio = endpoint_contributors[remote_host, remote_port]
                    path = make_zmq_path("tcp", remote_host, remote_port)
                    layer_slots = self._state.layer_storage_slots.get(layer_idx, ())
                    touched_slots = [
                        slot_id
                        for layout, slot_id in zip(layer_layouts, layer_slots, strict=True)
                        if any(req[1][layout.group_index] for req in write_reqs)
                    ]
                    reader = (transfer_id, path)
                    self._ensure_reuse_tracker().begin(reader, touched_slots)
                    if done_req_ids:
                        with self._request_completion_lock:
                            self._completion_requests_by_transfer[reader] = set(done_req_ids)
                    registered_transfers.append(reader)
                    dealer = self._ensure_dealer(path)
                    if path not in self._layout_meta_sent_paths:
                        self._send_layout_meta(path, dealer, encoder)
                    ext_ids = {entry[0] for entry in write_reqs} | set(done_ext_ids)
                    requested = self._requested_targets.setdefault(path, set())
                    missing = ext_ids - requested
                    if missing:
                        dealer.send(encoder.encode((TARGET_BLOCKS, tuple(missing))))
                        requested.update(missing)
                    self._pending.setdefault(path, deque()).append(
                        (
                            transfer_id,
                            layer_idx,
                            write_reqs,
                            done_ext_ids,
                            group_member_idx,
                            tp_ratio,
                            wait_event,
                        )
                    )
                    sent_transfers.add(reader)
                    logger.debug(
                        "Layerwise push P sent WRITE batch: transfer=%d, layer=%d (%s), "
                        "endpoint=%s:%d, reqs=%d, done_reqs=%d",
                        transfer_id,
                        layer_idx,
                        layer_name,
                        remote_host,
                        remote_port,
                        len(write_reqs),
                        len(done_ext_ids),
                    )
                # Replace the enqueue-time safety reservation with the exact
                # per-endpoint readers registered above.
                self._release_task_reservation(send_task)
            except Exception as error:
                for reader in registered_transfers:
                    if reader not in sent_transfers:
                        self._complete_transfer(reader, str(error))
                self._release_task_reservation(send_task, str(error))
                raise
        else:
            self._release_task_reservation(send_task)

    def _send_layout_meta(self, path: str, dealer, encoder: msgspec.msgpack.Encoder) -> None:
        layouts = {
            layer_idx: {
                "num_blocks": self._state.num_blocks,
                "producer_pp_rank": self._state.pp_rank,
                "producer_tp_rank": self._state.tp_rank,
                "components": [
                    {
                        "name": component.name,
                        "group_index": component.group_index,
                        "block_size": component.block_size,
                        "dtypes": list(component.dtypes),
                        "base_addrs": list(component.base_addrs),
                        "block_strides": list(component.block_strides),
                        "block_lengths": list(component.block_lengths),
                        "block_shapes": [list(shape) for shape in component.block_shapes],
                        "block_size_scales": list(component.block_size_scales),
                    }
                    for component in components
                ],
            }
            for layer_idx, components in self._state.layer_layouts.items()
        }
        dealer.send(
            encoder.encode(
                (
                    LAYOUT_META,
                    self._state.p_session,
                    encoder.encode(layouts),
                    self._state.pp_layers,
                    self._state.tp_size,
                    self._state.pp_rank,
                    self._state.tp_rank,
                )
            )
        )
        # Reply processing stays in the event loop so one cold D cannot block
        # handshakes or completed writes for other endpoints.
        self._layout_meta_sent_paths.add(path)
        self._handshake_deadlines[path] = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS

    def _drain_replies(self, decoder: msgspec.msgpack.Decoder) -> None:
        for path, dealer in self._dealers.items():
            while dealer.poll(timeout=0):
                frames = dealer.recv_multipart(flags=zmq.NOBLOCK)
                payload = [frame for frame in frames if frame]
                if len(payload) != 1:
                    raise ValueError(f"Invalid push reply frame count: frames={len(frames)}, non_empty={len(payload)}")
                msg = decoder.decode(payload[0])
                if msg[0] == LAYOUT_META:
                    _, session, raw_layouts, rank, size, shared = msg
                    if path in self._planners:
                        raise RuntimeError("Duplicate destination layout handshake")
                    layouts = msgspec.convert(raw_layouts, type=dict[int, tuple[ComponentLayout, ...]])
                    self._planners[path] = WritePlanner(
                        self.backend,
                        ConsumerDestinationState(
                            tp_size=size,
                            layer_layouts=layouts,
                            dest_blocks_by_req={},
                            tp_shared_components=frozenset(shared),
                            session=session,
                        ),
                        rank,
                    )
                    self._handshake_deadlines.pop(path, None)
                elif msg[0] == TARGET_BLOCKS:
                    planner = self._planners[path]
                    planner._state.dest_blocks_by_req.update(
                        {ext_id: [list(group) for group in groups] for ext_id, groups in msg[1].items()}
                    )
                else:
                    raise RuntimeError(f"Unexpected push reply: {msg!r}")

    def _release_task_reservation(self, task: SendTask, error: str | None = None) -> None:
        if task.reservation_id is not None:
            self._ensure_reuse_tracker().complete(task.reservation_id, error)
            task.reservation_id = None
        logger.debug("Layerwise push P layer send complete: layer=%d", task.layer_idx)

    def _record_layer_error(self, layer_idx: int, error: str) -> None:
        for slot_id in self._state.layer_storage_slots.get(layer_idx, ()):
            if slot_id in self._ensure_reuse_tracker().reused_slots:
                self._storage_write_errors[slot_id] = error

    def get_storage_send_event(self, slot_id: int) -> threading.Event | None:
        return self._ensure_reuse_tracker().event(slot_id)

    def get_storage_error(self, slot_id: int) -> str | None:
        return self._ensure_reuse_tracker().error(slot_id)
