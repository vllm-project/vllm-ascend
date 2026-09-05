# SPDX-License-Identifier: Apache-2.0
"""D-side reader for backend-independent layerwise pull."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import msgspec
import numpy as np
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    LAYOUT_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    ComponentLayout,
)

READ_THREAD_POLL_TIMEOUT_MS = 100
THREAD_SHUTDOWN_TIMEOUT_SECONDS = 5.0
DEST_BLOCK_WAIT_TIMEOUT_SECONDS = 2.0


class PullBackend:
    """Expose one READ-only interface over a registered transfer engine."""

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
    def memfabric(cls, engine: Any) -> PullBackend:
        return cls(engine, "MemFabric", lambda ret: ret != 0)

    @classmethod
    def mooncake(cls, engine: Any) -> PullBackend:
        return cls(engine, "Mooncake", lambda ret: ret < 0)

    def read(
        self,
        session_id: str,
        local_addrs: Sequence[int],
        remote_addrs: Sequence[int],
        lengths: Sequence[int],
    ) -> None:
        if not (len(local_addrs) == len(remote_addrs) == len(lengths)):
            raise ValueError(
                "Layerwise pull descriptor counts differ: "
                f"local={len(local_addrs)}, remote={len(remote_addrs)}, lengths={len(lengths)}"
            )
        if not local_addrs:
            return
        ret = self._engine.batch_transfer_sync_read(
            session_id,
            local_addrs if isinstance(local_addrs, list) else list(local_addrs),
            remote_addrs if isinstance(remote_addrs, list) else list(remote_addrs),
            lengths if isinstance(lengths, list) else list(lengths),
        )
        if self._is_error(ret):
            raise RuntimeError(f"{self._name} READ failed for session {session_id}, ret={ret}")


def _coalesce_read_descriptors(
    remote_addrs: np.ndarray,
    local_addrs: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge runs only when both local and remote byte ranges are contiguous."""
    count = remote_addrs.shape[0]
    if count <= 1:
        return remote_addrs, local_addrs, lengths
    contiguous = (remote_addrs[1:] == remote_addrs[:-1] + lengths[:-1]) & (
        local_addrs[1:] == local_addrs[:-1] + lengths[:-1]
    )
    run_start = np.concatenate(([0], np.nonzero(~contiguous)[0] + 1))
    run_end = np.append(run_start[1:] - 1, count - 1)
    cumulative = np.cumsum(lengths)
    merged_lengths = cumulative[run_end] - cumulative[run_start] + lengths[run_start]
    return remote_addrs[run_start], local_addrs[run_start], merged_lengths


def plan_block_reads(
    *,
    remote_base: int,
    local_base: int,
    remote_block_ids: list[int] | np.ndarray,
    local_block_ids: list[int] | np.ndarray,
    remote_stride: int,
    local_stride: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a block mapping in NumPy and coalesce adjacent ranges."""
    if len(remote_block_ids) != len(local_block_ids):
        raise ValueError(
            f"Layerwise pull block counts differ: remote={len(remote_block_ids)}, local={len(local_block_ids)}"
        )
    if len(remote_block_ids) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    remote_ids = np.asarray(remote_block_ids, dtype=np.int64)
    local_ids = np.asarray(local_block_ids, dtype=np.int64)
    lengths = np.full(remote_ids.shape[0], length, dtype=np.int64)
    return _coalesce_read_descriptors(
        remote_base + remote_ids * remote_stride,
        local_base + local_ids * local_stride,
        lengths,
    )


@dataclass
class ConsumerReadState:
    """Local destination layouts and request-owned block ids."""

    tp_size: int
    layer_layouts: dict[int, tuple[ComponentLayout, ...]]
    dest_blocks_by_req: dict[str, list[list[int]]]
    # Sparse offload's main cache is one TP-shared CPU destination. This small
    # adapter hint prevents duplicate reads; ordinary HBM layouts leave it empty.
    tp_shared_components: frozenset[str] = frozenset()
    dest_blocks_condition: threading.Condition = field(default_factory=threading.Condition)


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


class LayerwisePullReadThread(threading.Thread):
    """Receive ready notifications and pull matching cache components."""

    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        backend: PullBackend,
        state: ConsumerReadState,
    ) -> None:
        super().__init__(daemon=True, name=f"LayerwisePullReadThread-TP{tp_rank}")
        self.tp_rank = tp_rank
        self.side_channel_port = side_channel_port
        self.backend = backend
        self._state = state
        self.ready_event = threading.Event()
        self._p_sessions: dict[bytes, str] = {}
        self._p_layer_layouts: dict[bytes, dict[int, tuple[ComponentLayout, ...]]] = {}
        self._done_requests: set[str] = set()
        self._failed_requests: set[str] = set()
        # Unlike the failure notification queue, these IDs survive request
        # cleanup: another P stage may still send READ_READY after D retries
        # locally or reuses the blocks. Retain them for this reader's lifetime;
        # there is no cross-source cancellation/drain handshake.
        self._failed_request_ids: set[str] = set()
        self._done_contributors: dict[str, set[tuple[int, int]]] = {}
        self._expected_sources: dict[str, frozenset[tuple[int, int]]] = {}
        self._p_completion_sources: dict[bytes, tuple[tuple[int, int], int, frozenset[tuple[int, int]]]] = {}
        self._terminal_requests: set[str] = set()
        self._lock = threading.Lock()
        self._host = get_ip()
        self._stop_event = threading.Event()
        self.startup_error: BaseException | None = None

    def _record_chunk_done(
        self, done_ext_ids: list[str], contributor: tuple[int, int], expected: frozenset[tuple[int, int]]
    ) -> None:
        with self._lock:
            for ext_id in done_ext_ids:
                if ext_id in self._terminal_requests or ext_id in self._failed_request_ids:
                    continue
                if contributor not in expected or self._expected_sources.setdefault(ext_id, expected) != expected:
                    raise ValueError("Layerwise pull received inconsistent request completion topology")
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
            raise ValueError("Layerwise pull requires P TP size divisible by D TP size")
        if not 0 <= pp_rank < len(pp_layers) or not 0 <= tp_rank < tp_size:
            raise ValueError("Layerwise pull received an invalid producer rank")
        ratio = tp_size // self._state.tp_size
        if tp_rank // ratio != self.tp_rank:
            raise ValueError("Layerwise pull producer connected to the wrong D TP rank")
        all_layers = [layer for layers in pp_layers for layer in layers]
        if len(all_layers) != len(set(all_layers)):
            raise ValueError("Layerwise pull producer PP stages have overlapping layers")
        local_layers = set(self._state.layer_layouts)
        if not local_layers.issubset(all_layers):
            raise ValueError("Layerwise pull producer topology is missing local destination layers")
        expected = frozenset(
            (pp, member)
            for pp, layers in enumerate(pp_layers)
            if local_layers.intersection(layers)
            for member in range(ratio)
        )
        contributor = (pp_rank, tp_rank % ratio)
        if contributor not in expected:
            raise ValueError("Layerwise pull producer has no layers for this D stage")
        raw_layers = msgspec.msgpack.decode(encoded_layers)
        if set(raw_layers) != set(pp_layers[pp_rank]):
            raise ValueError("Layerwise pull producer layout does not match its advertised PP ownership")
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
        with self._lock:
            failed = self._failed_requests
            self._failed_requests = set()
            return failed

    def run(self) -> None:
        from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

        path = make_zmq_path("tcp", self._host, self.side_channel_port + self.tp_rank)
        logger.info("Layerwise pull read thread listening on %s", path)
        ctx = zmq.Context()  # type: ignore[attr-defined]
        sock = None
        try:
            try:
                sock = make_zmq_socket(
                    ctx=ctx,
                    path=path,
                    socket_type=zmq.ROUTER,  # type: ignore[attr-defined]
                    bind=True,
                )
                sock.setsockopt(zmq.RCVTIMEO, READ_THREAD_POLL_TIMEOUT_MS)  # type: ignore[attr-defined]
            except BaseException as error:
                self.startup_error = error
                logger.error("Layerwise pull read thread failed to start on %s: %s", path, error)
                return
            finally:
                self.ready_event.set()

            decoder = msgspec.msgpack.Decoder(type=tuple)
            encoder = msgspec.msgpack.Encoder()
            while not self._stop_event.is_set():
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        continue
                    identity = frames[0]
                    payload = [frame for frame in frames[1:] if frame]
                    if len(payload) != 1:
                        continue
                    msg = decoder.decode(payload[0])
                    if msg[0] == LAYOUT_META:
                        self._register_remote_layout(identity, msg)
                        sock.send_multipart((identity, b"", b"ACK"))
                        continue

                    if msg[0] != READ_READY_BATCH:
                        logger.error("Layerwise pull received unexpected message %s", msg)
                        continue

                    layer_idx = int(msg[1])
                    read_reqs = [(entry[0], [list(group) for group in entry[1]], list(entry[2])) for entry in msg[3]]
                    done_ext_ids = list(msg[4]) if len(msg) > 4 else []
                    group_member_idx = int(msg[5]) if len(msg) > 5 else 0
                    ratio = max(int(msg[6]), 1) if len(msg) > 6 else 1
                    transfer_id = int(msg[7]) if len(msg) > 7 else None
                    try:
                        session = self._p_sessions.get(identity)
                        source_layouts = self._p_layer_layouts.get(identity)
                        if session is None or source_layouts is None:
                            raise RuntimeError("LAYOUT_META was not received before READ_READY_BATCH")
                        contributor, expected_ratio, expected = self._p_completion_sources[identity]
                        if contributor[1] != group_member_idx or expected_ratio != ratio:
                            raise ValueError("READ_READY_BATCH TP contributor differs from its layout handshake")
                        if read_reqs:
                            self._do_read_batch(
                                layer_idx,
                                read_reqs,
                                session,
                                source_layouts,
                                group_member_idx,
                                ratio,
                            )
                        reply = (
                            (READ_DONE, layer_idx, transfer_id) if transfer_id is not None else (READ_DONE, layer_idx)
                        )
                        if done_ext_ids:
                            self._record_chunk_done(done_ext_ids, contributor, expected)
                        sock.send_multipart((identity, b"", encoder.encode(reply)))
                    except Exception as error:
                        logger.error("Layerwise pull read failed for layer %d: %s", layer_idx, error)
                        failed = {entry[0] for entry in read_reqs}
                        failed.update(done_ext_ids)
                        with self._lock:
                            # The synchronous read has returned. Block all later
                            # writes before making failure visible to the worker,
                            # even if sending the error reply itself fails.
                            self._failed_requests.update(failed - self._failed_request_ids)
                            self._failed_request_ids.update(failed)
                            self._done_requests.difference_update(failed)
                            for ext_id in failed:
                                self._done_contributors.pop(ext_id, None)
                                self._expected_sources.pop(ext_id, None)
                        reply = (
                            (READ_FAILED, layer_idx, str(error), transfer_id)
                            if transfer_id is not None
                            else (READ_FAILED, layer_idx, str(error))
                        )
                        sock.send_multipart((identity, b"", encoder.encode(reply)))
                except zmq.Again:  # type: ignore[attr-defined]
                    continue
                except Exception as error:
                    logger.error("Layerwise pull read thread error: %s: %s", type(error).__name__, error)
        finally:
            self.ready_event.set()
            if sock is not None:
                sock.close(linger=0)
            ctx.destroy(linger=0)

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        with self._state.dest_blocks_condition:
            self._stop_event.set()
            self._state.dest_blocks_condition.notify_all()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)
        if self.is_alive():
            logger.warning("Layerwise pull read thread did not stop within %.1f seconds", timeout)

    def _do_read_batch(
        self,
        layer_idx: int,
        read_reqs: list[tuple[str, list[list[int]], list[int]]],
        session: str,
        source_layouts: dict[int, tuple[ComponentLayout, ...]],
        group_member_idx: int = 0,
        ratio: int = 1,
    ) -> None:
        if self._failed_request_ids:
            # READ_DONE for skipped requests only releases the P-side source
            # buffers. _record_chunk_done must not turn them into D successes.
            read_reqs = [entry for entry in read_reqs if entry[0] not in self._failed_request_ids]
            if not read_reqs:
                return
        source_components = source_layouts.get(layer_idx)
        destination_components = self._state.layer_layouts.get(layer_idx)
        if source_components is None or destination_components is None:
            raise RuntimeError(f"Layerwise pull layout is missing layer {layer_idx}")

        destination_by_name = {component.name: component for component in destination_components}
        if len(destination_by_name) != len(destination_components):
            raise RuntimeError(f"Layerwise pull destination has duplicate component names for layer {layer_idx}")
        source_names = {component.name for component in source_components}
        if source_names != set(destination_by_name):
            raise RuntimeError(
                f"Layerwise pull components differ for layer {layer_idx}: "
                f"source={sorted(source_names)}, destination={sorted(destination_by_name)}"
            )

        # Layouts are shared by all requests in this batch.
        for source in source_components:
            destination = destination_by_name[source.name]
            if source.block_size != destination.block_size:
                raise RuntimeError(
                    f"Layerwise pull block size differs for {source.name}: "
                    f"source={source.block_size}, destination={destination.block_size}"
                )
            if len(source.base_addrs) != len(destination.base_addrs):
                raise RuntimeError(
                    f"Layerwise pull tensor count differs for {source.name}: "
                    f"source={len(source.base_addrs)}, destination={len(destination.base_addrs)}"
                )
            if source.dtypes != destination.dtypes:
                raise RuntimeError(
                    f"Layerwise pull dtype differs for {source.name}: "
                    f"source={source.dtypes}, destination={destination.dtypes}"
                )
            if source.block_shapes != destination.block_shapes:
                raise RuntimeError(
                    f"Layerwise pull shape differs for {source.name}: "
                    f"source={source.block_shapes}, destination={destination.block_shapes}"
                )
            if source.block_size_scales != destination.block_size_scales:
                raise RuntimeError(
                    f"Layerwise pull block scale differs for {source.name}: "
                    f"source={source.block_size_scales}, destination={destination.block_size_scales}"
                )
            for tensor_idx, (source_length, destination_length) in enumerate(
                zip(source.block_lengths, destination.block_lengths, strict=True)
            ):
                if source_length != destination_length:
                    raise RuntimeError(
                        f"Layerwise pull tensor size differs for {source.name}[{tensor_idx}]: "
                        f"source={source_length}, destination={destination_length}"
                    )

        # Gather paired IDs across requests before doing NumPy work per tensor.
        blocks_by_component: dict[str, tuple[list[int], list[int]]] = {
            source.name: ([], []) for source in source_components
        }
        for ext_req_id, source_blocks_by_group, start_blocks_by_group in read_reqs:
            destination_blocks_by_group = self._state.dest_blocks_by_req.get(ext_req_id)
            if destination_blocks_by_group is None:
                with self._state.dest_blocks_condition:
                    self._state.dest_blocks_condition.wait_for(
                        lambda req_id=ext_req_id: req_id in self._state.dest_blocks_by_req or self._stop_event.is_set(),
                        timeout=DEST_BLOCK_WAIT_TIMEOUT_SECONDS,
                    )
                    if self._stop_event.is_set():
                        raise RuntimeError(f"Layerwise pull stopped waiting for D blocks for request {ext_req_id}")
                    destination_blocks_by_group = self._state.dest_blocks_by_req.get(ext_req_id)
                    if destination_blocks_by_group is None:
                        raise RuntimeError(f"Layerwise pull has no D blocks for request {ext_req_id}")

            for source in source_components:
                destination = destination_by_name[source.name]
                try:
                    source_block_ids = source_blocks_by_group[source.group_index]
                    start_block = int(start_blocks_by_group[source.group_index])
                    all_destination_block_ids = destination_blocks_by_group[destination.group_index]
                except IndexError as error:
                    raise RuntimeError(f"Layerwise pull block group is missing for {source.name}") from error
                end_block = start_block + len(source_block_ids)
                if end_block > len(all_destination_block_ids):
                    raise RuntimeError(
                        f"Layerwise pull D block range is incomplete for {source.name}: "
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
                        f"Layerwise pull block counts differ for request {ext_req_id}: "
                        f"remote={len(source_block_ids)}, local={len(destination_block_ids)}"
                    )
                batch_source_blocks, batch_destination_blocks = blocks_by_component[source.name]
                batch_source_blocks.extend(source_block_ids)
                batch_destination_blocks.extend(destination_block_ids)

        local_addrs: list[int] = []
        remote_addrs: list[int] = []
        lengths: list[int] = []
        for source in source_components:
            batch_source_blocks, batch_destination_blocks = blocks_by_component[source.name]
            if not batch_source_blocks:
                continue
            destination = destination_by_name[source.name]
            source_ids = np.asarray(batch_source_blocks, dtype=np.int64)
            destination_ids = np.asarray(batch_destination_blocks, dtype=np.int64)
            for tensor_idx, (remote_base, local_base) in enumerate(
                zip(source.base_addrs, destination.base_addrs, strict=True)
            ):
                remote, local, planned_lengths = plan_block_reads(
                    remote_base=remote_base,
                    local_base=local_base,
                    remote_block_ids=source_ids,
                    local_block_ids=destination_ids,
                    remote_stride=source.block_strides[tensor_idx],
                    local_stride=destination.block_strides[tensor_idx],
                    length=source.block_lengths[tensor_idx],
                )
                remote_addrs.extend(remote.tolist())
                local_addrs.extend(local.tolist())
                lengths.extend(planned_lengths.tolist())

        self.backend.read(session, local_addrs, remote_addrs, lengths)
