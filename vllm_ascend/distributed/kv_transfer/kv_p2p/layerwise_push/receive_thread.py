# SPDX-License-Identifier: Apache-2.0
"""Batched P-side WRITE planning and the D-side control receiver."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.protocol import (
    LAYOUT_META,
    REQUEST_DONE,
    TARGET_BLOCKS,
    WRITE_FAILED,
    ComponentLayout,
)

THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class WriteBackend:
    """Expose one WRITE-only interface over a registered transfer engine."""

    def __init__(
        self,
        engine: Any,
        name: str,
        is_error: Callable[[int], bool],
    ) -> None:
        self._engine = engine
        self._name = name
        self._is_error = is_error

    @classmethod
    def memfabric(cls, engine: Any) -> WriteBackend:
        return cls(engine, "MemFabric", lambda ret: ret != 0)

    @classmethod
    def mooncake(cls, engine: Any) -> WriteBackend:
        return cls(engine, "Mooncake", lambda ret: ret < 0)

    def write(
        self,
        session_id: str,
        local_addrs: Sequence[int],
        remote_addrs: Sequence[int],
        lengths: Sequence[int],
    ) -> None:
        if not (len(local_addrs) == len(remote_addrs) == len(lengths)):
            raise ValueError(
                "Layerwise push descriptor counts differ: "
                f"local={len(local_addrs)}, remote={len(remote_addrs)}, lengths={len(lengths)}"
            )
        if not local_addrs:
            return
        ret = self._engine.batch_transfer_sync_write(
            session_id,
            local_addrs if isinstance(local_addrs, list) else list(local_addrs),
            remote_addrs if isinstance(remote_addrs, list) else list(remote_addrs),
            lengths if isinstance(lengths, list) else list(lengths),
        )
        if self._is_error(ret):
            raise RuntimeError(f"{self._name} WRITE failed for session {session_id}, ret={ret}")

    def write_async(
        self,
        session_id: str,
        local_addrs: Sequence[int],
        remote_addrs: Sequence[int],
        lengths: Sequence[int],
        stream: int,
    ) -> None:
        """Submit a WRITE on ``stream``; completion is tracked by its NPU event."""
        if not (len(local_addrs) == len(remote_addrs) == len(lengths)):
            raise ValueError(
                "Layerwise push descriptor counts differ: "
                f"local={len(local_addrs)}, remote={len(remote_addrs)}, lengths={len(lengths)}"
            )
        if not local_addrs:
            return
        submit = getattr(self._engine, "batch_transfer_async_write_submit", None)
        if submit is None:
            raise RuntimeError(
                f"{self._name} does not provide batch_transfer_async_write_submit; "
                'use transfer_backend="memfabric" or push_write_mode="sync"'
            )
        ret = submit(
            session_id,
            local_addrs if isinstance(local_addrs, list) else list(local_addrs),
            remote_addrs if isinstance(remote_addrs, list) else list(remote_addrs),
            lengths if isinstance(lengths, list) else list(lengths),
            stream,
        )
        if self._is_error(ret):
            raise RuntimeError(f"{self._name} async WRITE submit failed for session {session_id}, ret={ret}")


def _coalesce_descriptors(
    source_addrs: np.ndarray,
    destination_addrs: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge runs only when both local and remote byte ranges are contiguous."""
    count = source_addrs.shape[0]
    if count <= 1:
        return source_addrs, destination_addrs, lengths
    contiguous = (source_addrs[1:] == source_addrs[:-1] + lengths[:-1]) & (
        destination_addrs[1:] == destination_addrs[:-1] + lengths[:-1]
    )
    run_start = np.concatenate(([0], np.nonzero(~contiguous)[0] + 1))
    run_end = np.append(run_start[1:] - 1, count - 1)
    cumulative = np.cumsum(lengths)
    merged_lengths = cumulative[run_end] - cumulative[run_start] + lengths[run_start]
    return source_addrs[run_start], destination_addrs[run_start], merged_lengths


def plan_block_transfers(
    *,
    source_base: int,
    destination_base: int,
    source_block_ids: list[int] | np.ndarray,
    destination_block_ids: list[int] | np.ndarray,
    source_stride: int,
    destination_stride: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a block mapping in NumPy and coalesce adjacent ranges."""
    if len(source_block_ids) != len(destination_block_ids):
        raise ValueError(
            "Layerwise push block counts differ: "
            f"source={len(source_block_ids)}, destination={len(destination_block_ids)}"
        )
    if len(source_block_ids) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    source_ids = np.asarray(source_block_ids, dtype=np.int64)
    destination_ids = np.asarray(destination_block_ids, dtype=np.int64)
    lengths = np.full(source_ids.shape[0], length, dtype=np.int64)
    return _coalesce_descriptors(
        source_base + source_ids * source_stride,
        destination_base + destination_ids * destination_stride,
        lengths,
    )


@dataclass
class ConsumerDestinationState:
    """Local destination layouts and request-owned block ids."""

    tp_size: int
    layer_layouts: dict[int, tuple[ComponentLayout, ...]]
    dest_blocks_by_req: dict[str, list[list[int]]]
    # Sparse offload's main cache is one TP-shared CPU destination. This small
    # adapter hint prevents duplicate writes; ordinary HBM layouts leave it empty.
    tp_shared_components: frozenset[str] = frozenset()
    dest_blocks_condition: threading.Condition = field(default_factory=threading.Condition)
    session: str = ""


def _tp_block_range(
    num_blocks: int,
    tp_rank: int,
    tp_size: int,
    start_block: int = 0,
) -> tuple[int, int]:
    logical_rank = (tp_rank - start_block % tp_size) % tp_size
    blocks_per_rank, remainder = divmod(num_blocks, tp_size)
    start = logical_rank * blocks_per_rank + min(logical_rank, remainder)
    count = blocks_per_rank + int(logical_rank < remainder)
    return start, start + count


class WritePlanner:
    """P-side batched address planning for one D endpoint; one active writer."""

    def __init__(self, backend: WriteBackend, state: ConsumerDestinationState, tp_rank: int):
        self.backend = backend
        self._state = state
        self.tp_rank = tp_rank
        self._component_pairs_by_source = {}

    def _match_components(
        self,
        layer_idx: int,
        source_layouts: dict[int, tuple[ComponentLayout, ...]],
    ) -> tuple[tuple[ComponentLayout, ComponentLayout], ...]:
        """Validate fixed layouts once before caching their component pairs."""
        source_components = source_layouts.get(layer_idx)
        destination_components = self._state.layer_layouts.get(layer_idx)
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

        # Layouts are shared by all requests in this batch.
        for source in source_components:
            destination = destination_by_name[source.name]
            if source.block_size != destination.block_size:
                raise RuntimeError(
                    f"Layerwise push block size differs for {source.name}: "
                    f"source={source.block_size}, destination={destination.block_size}"
                )
            if len(source.base_addrs) != len(destination.base_addrs):
                raise RuntimeError(
                    f"Layerwise push tensor count differs for {source.name}: "
                    f"source={len(source.base_addrs)}, destination={len(destination.base_addrs)}"
                )
            if source.dtypes != destination.dtypes:
                raise RuntimeError(
                    f"Layerwise push dtype differs for {source.name}: "
                    f"source={source.dtypes}, destination={destination.dtypes}"
                )
            if source.block_shapes != destination.block_shapes:
                raise RuntimeError(
                    f"Layerwise push shape differs for {source.name}: "
                    f"source={source.block_shapes}, destination={destination.block_shapes}"
                )
            if source.block_size_scales != destination.block_size_scales:
                raise RuntimeError(
                    f"Layerwise push block scale differs for {source.name}: "
                    f"source={source.block_size_scales}, destination={destination.block_size_scales}"
                )
            for tensor_idx, (source_length, destination_length) in enumerate(
                zip(source.block_lengths, destination.block_lengths, strict=True)
            ):
                if source_length != destination_length:
                    raise RuntimeError(
                        f"Layerwise push tensor size differs for {source.name}[{tensor_idx}]: "
                        f"source={source_length}, destination={destination_length}"
                    )
        return tuple((source, destination_by_name[source.name]) for source in source_components)

    def plan_batch(
        self,
        identity: bytes,
        layer_idx: int,
        write_reqs: list[tuple[str, list[list[int]], list[int]]],
        source_layouts: dict[int, tuple[ComponentLayout, ...]],
        group_member_idx: int = 0,
        ratio: int = 1,
    ) -> tuple[list[int], list[int], list[int]]:
        pairs_by_layer = self._component_pairs_by_source.setdefault(identity, {})
        component_pairs = pairs_by_layer.get(layer_idx)
        if component_pairs is None:
            component_pairs = self._match_components(layer_idx, source_layouts)
            pairs_by_layer[layer_idx] = component_pairs

        # Gather paired IDs across requests before doing NumPy work per tensor.
        blocks_by_component: dict[str, tuple[list[int], list[int]]] = {
            source.name: ([], []) for source, _ in component_pairs
        }
        for ext_req_id, source_blocks_by_group, start_blocks_by_group in write_reqs:
            destination_blocks_by_group = self._state.dest_blocks_by_req.get(ext_req_id)
            if destination_blocks_by_group is None:
                raise RuntimeError(f"Missing push destination blocks for {ext_req_id}")

            for source, destination in component_pairs:
                try:
                    source_block_ids = source_blocks_by_group[source.group_index]
                    start_block = int(start_blocks_by_group[source.group_index])
                    all_destination_block_ids = destination_blocks_by_group[destination.group_index]
                except IndexError as error:
                    raise RuntimeError(f"Layerwise push block group is missing for {source.name}") from error
                end_block = start_block + len(source_block_ids)
                if end_block > len(all_destination_block_ids):
                    raise RuntimeError(
                        f"Layerwise push D block range is incomplete for {source.name}: "
                        f"range=[{start_block}, {end_block}), allocated={len(all_destination_block_ids)}"
                    )
                destination_block_ids = all_destination_block_ids[start_block:end_block]

                if source.name in self._state.tp_shared_components:
                    if group_member_idx != 0:
                        continue
                    owned_start, owned_end = _tp_block_range(
                        len(source_block_ids), self.tp_rank, self._state.tp_size, start_block
                    )
                    source_block_ids = source_block_ids[owned_start:owned_end]
                    destination_block_ids = destination_block_ids[owned_start:owned_end]
                elif ratio > 1:
                    owned_start, owned_end = _tp_block_range(
                        len(source_block_ids), group_member_idx, ratio, start_block
                    )
                    source_block_ids = source_block_ids[owned_start:owned_end]
                    destination_block_ids = destination_block_ids[owned_start:owned_end]

                if len(source_block_ids) != len(destination_block_ids):
                    raise ValueError(
                        f"Layerwise push block counts differ for request {ext_req_id}: "
                        f"source={len(source_block_ids)}, destination={len(destination_block_ids)}"
                    )
                batch_source_blocks, batch_destination_blocks = blocks_by_component[source.name]
                batch_source_blocks.extend(source_block_ids)
                batch_destination_blocks.extend(destination_block_ids)

        destination_addrs: list[int] = []
        source_addrs: list[int] = []
        lengths: list[int] = []
        for source, destination in component_pairs:
            batch_source_blocks, batch_destination_blocks = blocks_by_component[source.name]
            if not batch_source_blocks:
                continue
            source_ids = np.asarray(batch_source_blocks, dtype=np.int64)
            destination_ids = np.asarray(batch_destination_blocks, dtype=np.int64)
            for tensor_idx, (source_base, destination_base) in enumerate(
                zip(source.base_addrs, destination.base_addrs, strict=True)
            ):
                planned_source, planned_destination, planned_lengths = plan_block_transfers(
                    source_base=source_base,
                    destination_base=destination_base,
                    source_block_ids=source_ids,
                    destination_block_ids=destination_ids,
                    source_stride=source.block_strides[tensor_idx],
                    destination_stride=destination.block_strides[tensor_idx],
                    length=source.block_lengths[tensor_idx],
                )
                source_addrs.extend(planned_source.tolist())
                destination_addrs.extend(planned_destination.tolist())
                lengths.extend(planned_lengths.tolist())

        return source_addrs, destination_addrs, lengths

    def write_batch(
        self,
        identity: bytes,
        layer_idx: int,
        write_reqs: list[tuple[str, list[list[int]], list[int]]],
        session: str,
        source_layouts: dict[int, tuple[ComponentLayout, ...]],
        group_member_idx: int = 0,
        ratio: int = 1,
    ) -> None:
        source_addrs, destination_addrs, lengths = self.plan_batch(
            identity,
            layer_idx,
            write_reqs,
            source_layouts,
            group_member_idx,
            ratio,
        )
        self.backend.write(session, source_addrs, destination_addrs, lengths)


class LayerwisePushReceiveThread(threading.Thread):
    """Publish D destinations and account for P-side completed writes."""

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        state: ConsumerDestinationState,
    ) -> None:
        super().__init__(daemon=True, name=f"LayerwisePushReceiveThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self._state = state
        self.ready_event = threading.Event()
        self._p_sessions: dict[bytes, str] = {}
        self._p_layer_layouts: dict[bytes, dict[int, tuple[ComponentLayout, ...]]] = {}
        self._done_requests: set[str] = set()
        self._done_contributors: dict[str, set[tuple[int, int]]] = {}
        self._expected_sources: dict[str, frozenset[tuple[int, int]]] = {}
        self._p_completion_sources: dict[bytes, tuple[tuple[int, int], int, frozenset[tuple[int, int]]]] = {}
        self._terminal_requests: set[str] = set()
        self._lock = threading.Lock()
        self._host = get_ip()
        self._stop_event = threading.Event()
        self.startup_error: BaseException | None = None
        self.fatal_error: BaseException | None = None
        self._target_waiters: dict[bytes, set[str]] = {}
        self._wake_reader, self._wake_writer = socket.socketpair()
        self._wake_reader.setblocking(False)
        self._wake_writer.setblocking(False)

    def _record_chunk_done(
        self, done_ext_ids: list[str], contributor: tuple[int, int], expected: frozenset[tuple[int, int]]
    ) -> None:
        with self._lock:
            for ext_id in done_ext_ids:
                if ext_id in self._terminal_requests:
                    continue
                if contributor not in expected or self._expected_sources.setdefault(ext_id, expected) != expected:
                    raise ValueError("Layerwise push received inconsistent request completion topology")
                contributors = self._done_contributors.setdefault(ext_id, set())
                contributors.add(contributor)
                if contributors == expected:
                    self._done_requests.add(ext_id)
                    self._terminal_requests.add(ext_id)
                    self._done_contributors.pop(ext_id, None)
                    self._expected_sources.pop(ext_id, None)

    def discard_requests(self, ext_ids: set[str]) -> None:
        with self._lock:
            for ext_id in ext_ids:
                self._done_contributors.pop(ext_id, None)
                self._expected_sources.pop(ext_id, None)
                self._terminal_requests.discard(ext_id)

    def _register_remote_layout(self, identity: bytes, msg: tuple) -> None:
        """Establish all expected PP/TP sources before accepting completion."""
        _, session, encoded_layers, pp_layers, tp_size, pp_rank, tp_rank = msg
        if not isinstance(session, str):
            raise ValueError("LAYOUT_META session must be a string")
        if tp_size < self._state.tp_size or tp_size % self._state.tp_size:
            raise ValueError("Layerwise push requires P TP size divisible by D TP size")
        if not 0 <= pp_rank < len(pp_layers) or not 0 <= tp_rank < tp_size:
            raise ValueError("Layerwise push received an invalid producer rank")
        ratio = tp_size // self._state.tp_size
        if tp_rank // ratio != self.tp_rank:
            raise ValueError("Layerwise push producer connected to the wrong D TP rank")
        all_layers = [layer for layers in pp_layers for layer in layers]
        if len(all_layers) != len(set(all_layers)):
            raise ValueError("Layerwise push producer PP stages have overlapping layers")
        local_layers = set(self._state.layer_layouts)
        if not local_layers.issubset(all_layers):
            raise ValueError("Layerwise push producer topology is missing local destination layers")
        expected = frozenset(
            (pp, member)
            for pp, layers in enumerate(pp_layers)
            if local_layers.intersection(layers)
            for member in range(ratio)
        )
        contributor = (pp_rank, tp_rank % ratio)
        if contributor not in expected:
            raise ValueError("Layerwise push producer has no layers for this D stage")
        raw_layers = msgspec.msgpack.decode(encoded_layers)
        if set(raw_layers) != set(pp_layers[pp_rank]):
            raise ValueError("Layerwise push producer layout does not match its advertised PP ownership")
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
        self._p_sessions[identity] = session
        self._p_layer_layouts[identity] = layouts
        self._p_completion_sources[identity] = (contributor, ratio, expected)

    def get_and_clear_done(self) -> set[str]:
        with self._lock:
            done = self._done_requests
            self._done_requests = set()
            return done

    def get_and_clear_failed(self) -> set[str]:
        # A WRITE timeout cannot safely trigger block reuse/local recompute.
        # Failures stop the worker via fatal_error instead.
        return set()

    def notify_targets(self) -> None:
        with suppress(OSError):
            self._wake_writer.send(b"\x00")

    def _send_targets(self, sock, encoder) -> None:
        # D keeps these blocks until every expected source sends its terminal
        # REQUEST_DONE, including when the scheduler has cancelled the request.
        for identity, waiting in list(self._target_waiters.items()):
            with self._state.dest_blocks_condition:
                targets = {
                    ext_id: self._state.dest_blocks_by_req[ext_id]
                    for ext_id in waiting
                    if ext_id in self._state.dest_blocks_by_req
                }
            if targets:
                sock.send_multipart((identity, b"", encoder.encode((TARGET_BLOCKS, targets))))
                waiting.difference_update(targets)
            if not waiting:
                del self._target_waiters[identity]

    def run(self) -> None:
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        path = make_zmq_path("tcp", self._host, self.side_channel_port + self.tp_rank)
        ctx = zmq.Context()
        sock = None
        try:
            sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.ROUTER, bind=True)
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            poller.register(self._wake_reader, zmq.POLLIN)
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            encoder = msgspec.msgpack.Encoder()
            while not self._stop_event.is_set():
                events = dict(poller.poll())
                if self._wake_reader.fileno() in events:
                    with suppress(BlockingIOError):
                        self._wake_reader.recv(4096)
                if sock in events:
                    frames = sock.recv_multipart()
                    identity = frames[0]
                    payload = [frame for frame in frames[1:] if frame]
                    if len(payload) != 1:
                        raise ValueError("Invalid push control message")
                    msg = decoder.decode(payload[0])
                    if msg[0] == LAYOUT_META:
                        self._register_remote_layout(identity, msg)
                        layouts = {
                            idx: [asdict(component) for component in components]
                            for idx, components in self._state.layer_layouts.items()
                        }
                        sock.send_multipart(
                            (
                                identity,
                                b"",
                                encoder.encode(
                                    (
                                        LAYOUT_META,
                                        self._state.session,
                                        layouts,
                                        self.tp_rank,
                                        self._state.tp_size,
                                        tuple(self._state.tp_shared_components),
                                    )
                                ),
                            )
                        )
                    elif msg[0] == TARGET_BLOCKS:
                        if identity not in self._p_completion_sources:
                            raise RuntimeError("Push targets requested before layout handshake")
                        self._target_waiters.setdefault(identity, set()).update(msg[1])
                    elif msg[0] == REQUEST_DONE:
                        _, done_ext_ids = msg
                        # The connection handshake already fixes this P
                        # source's PP/TP identity and expected contributors.
                        contributor, _, expected = self._p_completion_sources[identity]
                        self._record_chunk_done(list(done_ext_ids), contributor, expected)
                    elif msg[0] == WRITE_FAILED:
                        raise RuntimeError(f"Remote WRITE failed; keeping destination blocks pinned: {msg[1]}")
                    else:
                        raise ValueError(f"Unexpected push message {msg[0]!r}")
                self._send_targets(sock, encoder)
        except BaseException as error:
            self.fatal_error = error
            if not self.ready_event.is_set():
                self.startup_error = error
            logger.exception("Layerwise push receiver stopped; destination blocks must not be recycled")
        finally:
            self.ready_event.set()
            if sock is not None:
                sock.close(linger=0)
            ctx.destroy(linger=0)
            self._wake_reader.close()
            self._wake_writer.close()

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        self._stop_event.set()
        self.notify_targets()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
