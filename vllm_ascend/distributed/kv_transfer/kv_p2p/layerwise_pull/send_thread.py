# SPDX-License-Identifier: Apache-2.0
"""P-side control thread for backend-independent layerwise pull."""

from __future__ import annotations

import queue
import socket
import threading
import time
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    DEST_BLOCKS,
    DEST_BLOCKS_REQUEST,
    DEST_LAYOUT_META,
    LAYOUT_META,
    PUSH_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    WRITE_DONE,
    WRITE_FAILED,
    ComponentLayout,
    SendTask,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread import _tp_block_range, plan_block_reads

THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0
PUSH_DEST_BLOCKS_TIMEOUT_SECONDS = 30.0
ReaderId = tuple[int, str]


class SlotReuseTracker:
    """Track active D readers only for source slots reused by another layer."""

    def __init__(self, layer_slots: dict[int, tuple[int, ...]]) -> None:
        use_count = Counter(slot for slots in layer_slots.values() for slot in set(slots))
        self.reused_slots = frozenset(slot for slot, count in use_count.items() if count > 1)
        slot_count = max(use_count, default=-1) + 1
        self.events = [threading.Event() for _ in range(slot_count)]
        for event in self.events:
            event.set()
        self.errors: dict[int, str] = {}
        self._readers: dict[int, set[ReaderId]] = {slot: set() for slot in self.reused_slots}
        self._reader_slots: dict[ReaderId, tuple[int, ...]] = {}
        self._lock = threading.Lock()

    def begin(self, reader: ReaderId, touched_slots: Iterable[int]) -> None:
        slots = tuple(dict.fromkeys(slot for slot in touched_slots if slot in self.reused_slots))
        if not slots:
            return
        with self._lock:
            self._reader_slots[reader] = slots
            for slot in slots:
                readers = self._readers[slot]
                if not readers:
                    self.errors.pop(slot, None)
                    self.events[slot].clear()
                readers.add(reader)

    def complete(self, reader: ReaderId, error: str | None = None) -> None:
        with self._lock:
            slots = self._reader_slots.pop(reader, ())
            for slot in slots:
                readers = self._readers[slot]
                readers.discard(reader)
                if error is not None:
                    self.errors[slot] = error
                if not readers:
                    self.events[slot].set()

    def fail_all(self, error: str) -> None:
        with self._lock:
            readers = tuple(self._reader_slots)
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
    # Push mode (transfer_mode="push"): backend used by this thread to write
    # into D-advertised destinations. None in pull mode (P never transfers).
    transfer_mode: str = "pull"
    backend: Any | None = None


class LayerwisePullSendingThread(threading.Thread):
    """Publish ready layer layouts and source block ids to Decode.

    This thread never transfers payload bytes itself. After a layer is ready it
    sends ``READ_READY_BATCH`` and waits for Decode's ``READ_DONE`` before a
    reused source slot may be overwritten.
    """

    def __init__(
        self,
        *,
        ready_event: threading.Event,
        state: ProducerSendState,
    ) -> None:
        super().__init__(daemon=True, name="LayerwisePullSendingThread")
        self.timeout = 10.0
        self._layout_meta_sent_paths: set[str] = set()
        self._state = state
        self.last_layer_idx = state.last_layer_idx
        self.ready_event = ready_event
        self.send_queue: queue.Queue[SendTask] = queue.Queue()
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
        self._reuse_tracker = SlotReuseTracker(state.layer_storage_slots)
        # Keep these attributes for the worker-facing compatibility methods.
        self.storage_send_done_events = self._reuse_tracker.events
        self._storage_read_errors = self._reuse_tracker.errors
        self._legacy_readers: dict[int, tuple[int, str]] = {}
        self._request_completion_lock = threading.Lock()
        self._pending_completion_requests: set[str] = set()
        self._scheduler_finished_requests: set[str] = set()
        self._completed_requests: set[str] = set()
        self._completion_requests_by_reader: dict[tuple[int, str], set[str]] = {}
        self._pending_completion_paths: dict[str, set[str]] = {}
        # Push mode state
        self._transfer_mode = state.transfer_mode
        self._backend = state.backend
        self._push_ready_paths: set[str] = set()
        self._d_layouts_by_path: dict[str, dict[int, tuple[ComponentLayout, ...]]] = {}
        self._d_sessions: dict[str, str] = {}
        self._push_pairs_cache: dict[tuple[str, int], tuple[tuple[ComponentLayout, ComponentLayout], ...]] = {}
        self._dest_blocks_by_req: dict[str, list[list[int]]] = {}
        self._dest_block_queries_sent: set[str] = set()
        # Push requests whose write failed; never re-processed.
        self._push_failed_reqs: set[str] = set()
        # Deferred (layer task, deadline) pairs waiting for DEST_BLOCKS.
        self._deferred_tasks: list[tuple[SendTask, float]] = []
        # Per-layer fresh compute-stream events recorded by the producer in
        # save_kv_layer right after KV scatter.
        self._p_save_events: dict[int, Any] = {}

    def enqueue(self, task: SendTask) -> None:
        self.send_queue.put(task)
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
        with self._request_completion_lock:
            self._scheduler_finished_requests.update(scheduler_finished_req_ids & self._pending_completion_requests)
            finished = self._completed_requests & self._scheduler_finished_requests
            self._completed_requests.difference_update(finished)
            self._scheduler_finished_requests.difference_update(finished)
            self._pending_completion_requests.difference_update(finished)
            return finished

    def _complete_reader(self, reader: tuple[int, str], error: str | None = None) -> None:
        self._ensure_reuse_tracker().complete(reader, error)
        with self._request_completion_lock:
            for request_id in self._completion_requests_by_reader.pop(reader, ()):
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

    def run(self) -> None:
        try:
            from vllm.distributed import get_world_group

            local_rank = get_world_group().local_rank
            torch.npu.set_device(torch.device(f"npu:{local_rank}"))
        except BaseException as error:
            self.startup_error = error
            logger.error("Layerwise pull send thread failed to initialize its NPU device: %s", error)
            self.ready_event.set()
            self._task_reader.close()
            self._task_writer.close()
            self._persist_ctx.destroy(linger=0)
            return
        self.ready_event.set()

        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=tuple)
        thread_error = None
        try:
            while not self._stopped:
                try:
                    if self._dealers:
                        self._drain_read_replies(decoder)
                except Exception as e:
                    logger.error("Layerwise pull reply drain error: %s: %s", type(e).__name__, e)
                try:
                    send_task = self.send_queue.get_nowait()
                    try:
                        self._process_send_task(send_task, encoder)
                    except Exception as e:
                        layer_idx = getattr(send_task, "layer_idx", -1)
                        logger.error(
                            "Layerwise pull send task failed (layer=%s): %s: %s",
                            layer_idx,
                            type(e).__name__,
                            e,
                        )
                        self._fail_layer(layer_idx, str(e))
                except queue.Empty:
                    # Push mode needs a periodic wakeup to retry deferred tasks
                    # and expire their deadlines even when the wire is silent.
                    if getattr(self, "_transfer_mode", "pull") == "push":
                        self._poller.poll(timeout=100)
                    else:
                        self._poller.poll()
                    with suppress(BlockingIOError):
                        self._task_reader.recv(4096)
                    if getattr(self, "_transfer_mode", "pull") == "push":
                        self._retry_deferred(encoder)
        except BaseException as error:
            thread_error = error
            logger.error("Layerwise pull send thread crashed: %s: %s", type(error).__name__, error)
        finally:
            pending_error = (
                str(thread_error)
                if thread_error is not None
                else "send thread stopped before D completed all pending reads"
            )
            if self._dealers:
                try:
                    self._drain_read_replies(decoder)
                except Exception as error:
                    logger.warning("Layerwise pull final reply drain failed: %s", error)
            self._ensure_reuse_tracker().fail_all(pending_error)
            if self._dealers:
                for dealer in self._dealers.values():
                    dealer.close(linger=0)
                self._dealers.clear()
            self._task_reader.close()
            self._task_writer.close()
            self._persist_ctx.destroy(linger=0)

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        self._stopped = True
        # The poller may already be awake, or the thread may have exited.
        with suppress(OSError):
            self._task_writer.send(b"\x00")
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("Layerwise pull send thread did not stop within %.1f seconds", timeout)

    def record_p_save_event(self, layer_idx: int) -> None:
        evt = torch.npu.Event()
        evt.record()
        self._p_save_events[layer_idx] = evt

    def mark_layer_pending(self, layer_idx: int) -> None:
        """Compatibility helper for callers that do not provide transfer IDs."""
        reader = (-layer_idx - 1, "legacy")
        legacy_readers = getattr(self, "_legacy_readers", None)
        if legacy_readers is None:
            legacy_readers = {}
            self._legacy_readers = legacy_readers
        legacy_readers[layer_idx] = reader
        self._ensure_reuse_tracker().begin(reader, self._state.layer_storage_slots.get(layer_idx, ()))

    def _ensure_reuse_tracker(self) -> SlotReuseTracker:
        tracker = getattr(self, "_reuse_tracker", None)
        if tracker is None:
            tracker = SlotReuseTracker(self._state.layer_storage_slots)
            existing_events = getattr(self, "storage_send_done_events", None)
            if existing_events is not None and len(existing_events) == len(tracker.events):
                tracker.events = existing_events
            tracker.errors = getattr(self, "_storage_read_errors", {})
            self._reuse_tracker = tracker
            self.storage_send_done_events = tracker.events
            self._storage_read_errors = tracker.errors
        return tracker

    def _process_send_task(self, send_task: SendTask, encoder: msgspec.msgpack.Encoder) -> None:
        # getattr: unit tests instantiate this thread without running __init__.
        if getattr(self, "_transfer_mode", "pull") == "push":
            self._process_push_task(send_task, encoder)
            return
        transfer_id = getattr(self, "_next_transfer_id", 0)
        self._next_transfer_id = transfer_id + 1
        layer_idx = send_task.layer_idx
        p_save_event = self._p_save_events.pop(layer_idx, None)
        if p_save_event is not None:
            p_save_event.synchronize()
        elif send_task.wait_event is not None:
            send_task.wait_event.synchronize()
        layer_name = send_task.layer_name

        endpoint_payloads, endpoint_contributors, layer_layouts, layer_group_indices = self._group_by_endpoint(
            send_task
        )

        if endpoint_payloads:
            registered_readers: list[tuple[int, str]] = []
            sent_readers: set[tuple[int, str]] = set()
            try:
                for (remote_host, remote_port), (read_reqs, done_ext_ids, done_req_ids) in endpoint_payloads.items():
                    group_member_idx, tp_ratio = endpoint_contributors[remote_host, remote_port]
                    path = make_zmq_path("tcp", remote_host, remote_port)
                    layer_slots = self._state.layer_storage_slots.get(layer_idx, ())
                    touched_slots = [
                        slot_id
                        for layout, slot_id in zip(layer_layouts, layer_slots, strict=True)
                        if any(req[1][layout.group_index] for req in read_reqs)
                    ]
                    reader = (transfer_id, path)
                    self._ensure_reuse_tracker().begin(reader, touched_slots)
                    if done_req_ids:
                        with self._request_completion_lock:
                            self._completion_requests_by_reader[reader] = set(done_req_ids)
                    registered_readers.append(reader)
                    dealer = self._ensure_dealer(path)
                    if path not in self._layout_meta_sent_paths:
                        self._send_layout_meta(path, dealer, encoder)
                    dealer.send(
                        encoder.encode(
                            (
                                READ_READY_BATCH,
                                layer_idx,
                                layer_name,
                                read_reqs,
                                done_ext_ids,
                                group_member_idx,
                                tp_ratio,
                                transfer_id,
                            )
                        )
                    )
                    sent_readers.add(reader)
                    logger.debug(
                        "Layerwise pull P sent READ_READY_BATCH: transfer=%d, layer=%d (%s), "
                        "endpoint=%s:%d, reqs=%d, done_reqs=%d",
                        transfer_id,
                        layer_idx,
                        layer_name,
                        remote_host,
                        remote_port,
                        len(read_reqs),
                        len(done_ext_ids),
                    )
                # Replace the enqueue-time safety reservation with the exact
                # per-endpoint readers registered above.
                self._signal_layer_done(layer_idx)
            except Exception as error:
                for reader in registered_readers:
                    if reader not in sent_readers:
                        self._complete_reader(reader, str(error))
                self._signal_layer_done(layer_idx)
                raise
        else:
            self._signal_layer_done(layer_idx)

    def _group_by_endpoint(self, send_task: SendTask):
        """Group per-request source block ids by D endpoint (shared by both modes).

        Returns (endpoint_payloads, endpoint_contributors, layer_layouts,
        layer_group_indices) where endpoint_payloads maps (host, port) to
        (read_reqs, done_ext_ids, done_req_ids); read_reqs entries are
        (ext_id, source_blocks_by_group, start_blocks_by_group).
        """
        layer_idx = send_task.layer_idx
        layer_name = send_task.layer_name
        endpoint_payloads: dict[
            tuple[str, int],
            tuple[list[tuple[str, list[list[int]], list[int]]], list[str], list[str]],
        ] = {}
        endpoint_contributors: dict[tuple[str, int], tuple[int, int]] = {}
        layer_layouts = self._state.layer_layouts[layer_idx]
        layer_group_indices = {layout.group_index for layout in layer_layouts}

        for req_id, rm in send_task.send_request.items():
            if rm.chunk_start_blocks is None:
                raise RuntimeError(f"Layerwise pull request {req_id} is missing its precomputed chunk block range")
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
                read_reqs, done_ext_ids, done_req_ids = endpoint_payloads.setdefault(endpoint, ([], [], []))
                if has_blocks:
                    read_reqs.append((ext_id, source_blocks, start_blocks))
                if chunk_done:
                    done_ext_ids.append(ext_id)
                    done_req_ids.append(req_id)
            logger.debug(
                "Layerwise pull P prepared layer=%d (%s), req=%s, blocks_by_group=%s, done=%s",
                layer_idx,
                layer_name,
                ext_id,
                [len(group) for group in source_blocks],
                chunk_done,
            )
        return endpoint_payloads, endpoint_contributors, layer_layouts, layer_group_indices

    # ------------------------------------------------------------------
    # Push mode (transfer_mode="push"): P plans and writes into D-advertised
    # destinations; completion is detected locally, D only counts WRITE_DONE.
    # ------------------------------------------------------------------
    def _push_handshake(self, path: str, dealer: Any, encoder: msgspec.msgpack.Encoder) -> None:
        """Send PUSH_META and wait for D's DEST_LAYOUT_META (destination layouts)."""
        if path in self._push_ready_paths:
            return
        dealer.send(
            encoder.encode(
                (
                    PUSH_META,
                    self._state.pp_layers,
                    self._state.tp_size,
                    self._state.pp_rank,
                    self._state.tp_rank,
                )
            )
        )
        if not dealer.poll(timeout=int(self.timeout * 1000)):
            raise RuntimeError("Layerwise push PUSH_META timed out waiting for DEST_LAYOUT_META")
        frames = dealer.recv_multipart()
        payload = [f for f in frames if f != b""]
        if len(payload) != 1:
            raise RuntimeError(f"Layerwise push PUSH_META got malformed reply: {payload!r}")
        msg = msgspec.msgpack.decode(payload[0])
        if msg[0] == WRITE_FAILED:
            raise RuntimeError(f"Layerwise push rejected by D: {msg[2] if len(msg) > 2 else 'unknown'}")
        if msg[0] != DEST_LAYOUT_META:
            raise RuntimeError(f"Layerwise push PUSH_META got unexpected reply: {msg[0]!r}")
        raw_layers = msgspec.msgpack.decode(msg[1])
        layouts = {
            int(layer_idx): tuple(
                ComponentLayout(
                    name=component["name"],
                    group_index=int(component["group_index"]),
                    block_size=int(component["block_size"]),
                    dtypes=tuple(component["dtypes"]),
                    base_addrs=tuple(component["base_addrs"]),
                    block_strides=tuple(component["block_strides"]),
                    block_lengths=tuple(component["block_lengths"]),
                    block_shapes=tuple(tuple(shape) for shape in component["block_shapes"]),
                    block_size_scales=tuple(component["block_size_scales"]),
                )
                for component in raw_layer["components"]
            )
            for layer_idx, raw_layer in raw_layers.items()
        }
        self._d_layouts_by_path[path] = layouts
        self._d_sessions[path] = msg[2]
        self._push_ready_paths.add(path)
        logger.info("Layerwise push P handshaked with %s: d_session=%s, layers=%d", path, msg[2], len(layouts))

    def _match_push_components(
        self,
        path: str,
        layer_idx: int,
    ) -> tuple[tuple[ComponentLayout, ComponentLayout], ...]:
        """Pair P source components with D destination components by name."""
        cached = self._push_pairs_cache.get((path, layer_idx))
        if cached is not None:
            return cached
        source_components = self._state.layer_layouts.get(layer_idx)
        destination_components = self._d_layouts_by_path[path].get(layer_idx)
        if source_components is None or destination_components is None:
            raise RuntimeError(f"Layerwise push layout is missing layer {layer_idx}")
        destination_by_name = {component.name: component for component in destination_components}
        if len(destination_by_name) != len(destination_components):
            raise RuntimeError(f"Layerwise push destination has duplicate component names for layer {layer_idx}")
        source_names = {component.name for component in source_components}
        if source_names != set(destination_by_name):
            raise RuntimeError(
                f"Layerwise push components differ for layer {layer_idx}: "
                f"source={sorted(source_names)}, destination={sorted(destination_by_name)}"
            )
        for source in source_components:
            destination = destination_by_name[source.name]
            if source.block_size != destination.block_size:
                raise RuntimeError(f"Layerwise push block size differs for {source.name}")
            if len(source.base_addrs) != len(destination.base_addrs):
                raise RuntimeError(f"Layerwise push tensor count differs for {source.name}")
            if source.dtypes != destination.dtypes:
                raise RuntimeError(f"Layerwise push dtype differs for {source.name}")
            if source.block_shapes != destination.block_shapes:
                raise RuntimeError(f"Layerwise push shape differs for {source.name}")
            if source.block_size_scales != destination.block_size_scales:
                raise RuntimeError(f"Layerwise push block scale differs for {source.name}")
            for tensor_idx, (source_length, destination_length) in enumerate(
                zip(source.block_lengths, destination.block_lengths, strict=True)
            ):
                if source_length != destination_length:
                    raise RuntimeError(
                        f"Layerwise push tensor size differs for {source.name}[{tensor_idx}]: "
                        f"source={source_length}, destination={destination_length}"
                    )
        pairs = tuple((source, destination_by_name[source.name]) for source in source_components)
        self._push_pairs_cache[(path, layer_idx)] = pairs
        return pairs

    def _query_dest_blocks(self, dealer: Any, encoder: msgspec.msgpack.Encoder, path: str, ext_id: str) -> None:
        if (path, ext_id) in self._dest_block_queries_sent:
            return
        self._dest_block_queries_sent.add((path, ext_id))
        dealer.send(encoder.encode((DEST_BLOCKS_REQUEST, ext_id)))

    def _defer_task(self, send_task: SendTask, deadline: float) -> None:
        if any(task is send_task for task, _ in self._deferred_tasks):
            return
        self._deferred_tasks.append((send_task, deadline))

    def _process_push_task(
        self,
        send_task: SendTask,
        encoder: msgspec.msgpack.Encoder,
        deadline: float | None = None,
    ) -> None:
        if self._backend is None:
            raise RuntimeError("Layerwise push requires a transfer backend on the P side")
        transfer_id = self._next_transfer_id
        self._next_transfer_id = transfer_id + 1
        layer_idx = send_task.layer_idx
        p_save_event = self._p_save_events.pop(layer_idx, None)
        if p_save_event is not None:
            p_save_event.synchronize()
        elif send_task.wait_event is not None:
            send_task.wait_event.synchronize()

        endpoint_payloads, endpoint_contributors, layer_layouts, _ = self._group_by_endpoint(send_task)
        if not endpoint_payloads:
            self._signal_layer_done(layer_idx)
            return

        deferred = False
        for (remote_host, remote_port), (read_reqs, done_ext_ids, done_req_ids) in endpoint_payloads.items():
            path = make_zmq_path("tcp", remote_host, remote_port)
            dealer = self._ensure_dealer(path)
            self._push_handshake(path, dealer, encoder)
            group_member_idx, tp_ratio = endpoint_contributors[(remote_host, remote_port)]

            # Split requests by dest-block availability; missing ones defer.
            ready: list[tuple[str, list[list[int]], list[int]]] = []
            done_ready_ids: list[str] = []
            done_ready_req_ids: list[str] = []
            for ext_id, source_blocks, start_blocks in read_reqs:
                if ext_id in self._push_failed_reqs:
                    continue
                if ext_id in self._dest_blocks_by_req:
                    ready.append((ext_id, source_blocks, start_blocks))
                else:
                    self._query_dest_blocks(dealer, encoder, path, ext_id)
                    deferred = True
            for ext_id, req_id in zip(done_ext_ids, done_req_ids):
                if ext_id in self._push_failed_reqs:
                    continue
                if ext_id in self._dest_blocks_by_req:
                    done_ready_ids.append(ext_id)
                    done_ready_req_ids.append(req_id)
                else:
                    # Terminal-layer request with no block payload at this layer
                    # still must deliver WRITE_DONE after its blocks arrive.
                    self._query_dest_blocks(dealer, encoder, path, ext_id)
                    deferred = True

            if not ready:
                continue
            layer_slots = self._state.layer_storage_slots.get(layer_idx, ())
            touched_slots = [
                slot_id
                for layout, slot_id in zip(layer_layouts, layer_slots, strict=True)
                if any(req[1][layout.group_index] for req in ready)
            ]
            reader = (transfer_id, path)
            self._ensure_reuse_tracker().begin(reader, touched_slots)
            try:
                local_addrs: list[int] = []
                remote_addrs: list[int] = []
                lengths: list[int] = []
                pairs = self._match_push_components(path, layer_idx)
                for ext_id, source_blocks, start_blocks in ready:
                    dest_blocks_by_group = self._dest_blocks_by_req[ext_id]
                    for source, destination in pairs:
                        group_idx = source.group_index
                        source_ids = source_blocks[group_idx]
                        start_block = int(start_blocks[group_idx])
                        all_dest = dest_blocks_by_group[destination.group_index]
                        end_block = start_block + len(source_ids)
                        if end_block > len(all_dest):
                            raise RuntimeError(
                                f"Layerwise push D block range is incomplete for {source.name}: "
                                f"range=[{start_block}, {end_block}), allocated={len(all_dest)}"
                            )
                        dest_ids = all_dest[start_block:end_block]
                        # Unequal TP: this P rank writes only its own slice of the
                        # block range, mirroring pull's contributor slicing.
                        if tp_ratio > 1:
                            owned_start, owned_end = _tp_block_range(
                                len(source_ids), group_member_idx, tp_ratio, start_block
                            )
                            source_ids = source_ids[owned_start:owned_end]
                            dest_ids = dest_ids[owned_start:owned_end]
                        if len(source_ids) != len(dest_ids):
                            raise ValueError(
                                f"Layerwise push block counts differ for request {ext_id}: "
                                f"source={len(source_ids)}, dest={len(dest_ids)}"
                            )
                        for tensor_idx, (src_base, dst_base) in enumerate(
                            zip(source.base_addrs, destination.base_addrs, strict=True)
                        ):
                            remote, local, planned_lengths = plan_block_reads(
                                remote_base=dst_base,
                                local_base=src_base,
                                remote_block_ids=dest_ids,
                                local_block_ids=source_ids,
                                remote_stride=destination.block_strides[tensor_idx],
                                local_stride=source.block_strides[tensor_idx],
                                length=source.block_lengths[tensor_idx],
                            )
                            remote_addrs.extend(remote.tolist())
                            local_addrs.extend(local.tolist())
                            lengths.extend(planned_lengths.tolist())
                # The write IS the transfer: a successful sync return means the
                # payload has left the source buffers (bare-link validated), so
                # the source slots are immediately reusable.
                self._backend.write(self._d_sessions[path], local_addrs, remote_addrs, lengths)
                if done_ready_req_ids:
                    with self._request_completion_lock:
                        self._completion_requests_by_reader[reader] = set(done_ready_req_ids)
                self._complete_reader(reader)
                if done_ready_ids:
                    dealer.send(encoder.encode((WRITE_DONE, layer_idx, done_ready_ids, transfer_id)))
                logger.debug(
                    "Layerwise push P wrote layer=%d to %s: reqs=%d, descriptors=%d, done=%d",
                    layer_idx,
                    path,
                    len(ready),
                    len(lengths),
                    len(done_ready_ids),
                )
            except Exception as error:
                failed_ext_ids = [ext_id for ext_id, _, _ in ready] + done_ready_ids
                self._push_failed_reqs.update(failed_ext_ids)
                self._complete_reader(reader, str(error))
                with suppress(Exception):
                    dealer.send(encoder.encode((WRITE_FAILED, layer_idx, str(error), failed_ext_ids)))
                raise
        if deferred:
            self._defer_task(
                send_task, deadline if deadline is not None else time.monotonic() + PUSH_DEST_BLOCKS_TIMEOUT_SECONDS
            )
        self._signal_layer_done(layer_idx)

    def _retry_deferred(self, encoder: msgspec.msgpack.Encoder) -> None:
        if not self._deferred_tasks:
            return
        now = time.monotonic()
        tasks, self._deferred_tasks = self._deferred_tasks, []
        for task, deadline in tasks:
            pending = [
                req_id
                for req_id in task.send_request
                if get_external_request_id(req_id) not in self._dest_blocks_by_req
                and get_external_request_id(req_id) not in self._push_failed_reqs
            ]
            if not pending:
                continue  # everything in this task was already written
            if now >= deadline:
                # D never allocated (cancelled/lost): fail the request on D and
                # let P free it locally.
                logger.error("Layerwise push timed out waiting for D blocks, failing requests: %s", pending)
                for req_id in pending:
                    rm = task.send_request[req_id]
                    endpoint = rm.layer_endpoints.get(task.layer_idx)
                    if endpoint is None:
                        continue
                    path = make_zmq_path("tcp", *endpoint)
                    dealer = self._ensure_dealer(path)
                    with suppress(Exception):
                        dealer.send(
                            encoder.encode(
                                (WRITE_FAILED, task.layer_idx, "dest blocks timeout", [get_external_request_id(req_id)])
                            )
                        )
                continue
            subset = SendTask(
                send_request={req_id: task.send_request[req_id] for req_id in pending},
                layer_idx=task.layer_idx,
                layer_name=task.layer_name,
            )
            try:
                # May re-defer the subset with the SAME original deadline.
                self._process_push_task(subset, encoder, deadline=deadline)
            except Exception as error:
                logger.error("Layerwise push deferred write failed (layer=%d): %s", task.layer_idx, error)
                self._defer_task(subset, deadline)

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
        if dealer.poll(timeout=int(self.timeout * 1000)):
            frames = dealer.recv_multipart()
            payload = [f for f in frames if f != b""]
            if payload != [b"ACK"]:
                raise RuntimeError(f"Layerwise pull LAYOUT_META got unexpected reply: {payload!r}")
            self._layout_meta_sent_paths.add(path)
            logger.info(
                "Layerwise pull P sent LAYOUT_META: session=%s, layers=%d",
                self._state.p_session,
                len(layouts),
            )
        else:
            raise RuntimeError("Layerwise pull LAYOUT_META timed out")

    def _drain_read_replies(self, decoder: msgspec.msgpack.Decoder) -> None:
        for path, dealer in self._dealers.items():
            while True:
                try:
                    if not dealer.poll(timeout=0):
                        break
                    frames = dealer.recv_multipart(flags=zmq.NOBLOCK)  # type: ignore[attr-defined]
                except zmq.Again:  # type: ignore[attr-defined]
                    break
                payload = [f for f in frames if f != b""]
                if len(payload) != 1:
                    continue
                try:
                    msg = decoder.decode(payload[0])
                except Exception:
                    continue
                if msg[0] == DEST_BLOCKS:
                    # Push mode: D answered a DEST_BLOCKS_REQUEST.
                    ext_id = str(msg[1])
                    self._dest_blocks_by_req[ext_id] = [list(group) for group in msg[2]]
                    continue
                if len(msg) >= 2 and msg[0] == READ_DONE:
                    if len(msg) > 2:
                        self._complete_reader((int(msg[2]), path))
                    else:
                        self._signal_layer_done(msg[1])
                elif len(msg) >= 2 and msg[0] == READ_FAILED:
                    layer_idx = msg[1]
                    error = msg[2] if len(msg) > 2 else ""
                    logger.error(
                        "Layerwise pull P received READ_FAILED: layer=%s, error=%s",
                        layer_idx,
                        error,
                    )
                    if len(msg) > 3:
                        self._complete_reader((int(msg[3]), path), str(error))
                    else:
                        self._record_layer_error(layer_idx, str(error))
                        self._signal_layer_done(layer_idx)

    def _signal_layer_done(self, layer_idx: int) -> None:
        reader = getattr(self, "_legacy_readers", {}).pop(layer_idx, None)
        if reader is not None:
            self._ensure_reuse_tracker().complete(reader)
        logger.debug("Layerwise pull P layer send complete: layer=%d", layer_idx)

    def _fail_layer(self, layer_idx: int, error: str) -> None:
        if layer_idx in getattr(self, "_legacy_readers", {}):
            self._record_layer_error(layer_idx, error)
            self._signal_layer_done(layer_idx)

    def _record_layer_error(self, layer_idx: int, error: str) -> None:
        for slot_id in self._state.layer_storage_slots.get(layer_idx, ()):
            if slot_id in self._ensure_reuse_tracker().reused_slots:
                self._storage_read_errors[slot_id] = error

    def get_storage_send_event(self, slot_id: int) -> threading.Event | None:
        return self._ensure_reuse_tracker().event(slot_id)

    def get_storage_error(self, slot_id: int) -> str | None:
        return self._ensure_reuse_tracker().error(slot_id)
