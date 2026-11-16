# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Scheduler-side logic for Mooncake pull transfers."""

import hashlib
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
ACK_MSG = b"ACK"

# Prefix length (hex chars) of each content-hash in the bidirectional KV
# block-hash chain. 16 hex chars (64 bits) per block is collision-safe for
# alignment verification while keeping the per-request JSON payload compact.
BIDIR_BLOCK_HASH_HEX_LEN = 16

# Hash granularity for the bidirectional KV alignment chain, in tokens. This
# is deliberately independent of the scheduler-level block size (which for
# hybrid-attention models under HMA is the lcm of per-group block sizes,
# e.g. 896 for Qwen3.5 — far too coarse: a 4527-token prompt + 300-token
# decode output both span exactly 5 x 896-token blocks, so no decode-side KV
# would ever be recovered). 128 matches the physical attention-group block
# size (the unit of KV transfer) and keeps ~40 hashes per 5k-token request.
BIDIR_HASH_BLOCK_SIZE = 128


def _bidir_block_hash(parent: str, block_tokens: Sequence[int]) -> str:
    """Hash one chain step: ``sha256(parent + "," + tokens)`` truncated.

    Shared by the full-block chain and the sequence tail so both sides
    compute identical values for the same token span and parent.
    """
    digest = hashlib.sha256()
    digest.update(parent.encode("ascii"))
    digest.update(b",")
    digest.update(",".join(str(token) for token in block_tokens).encode("ascii"))
    return digest.hexdigest()[:BIDIR_BLOCK_HASH_HEX_LEN]


def _bidir_block_hash_chain(
    token_ids: Sequence[int],
    block_size: int,
    max_blocks: int | None = None,
) -> list[str]:
    """Content-addressed hash chain over consecutive blocks of ``token_ids``.

    Each block's hash incorporates its parent's hash (chained, like vLLM's
    prefix-cache block hashes), so a match at block ``i`` proves the whole
    prefix ``[0, (i + 1) * block_size)`` is token-identical. The chain is
    computed identically on the D side (over prompt + decode output) and on
    the P side (over the next turn's prompt), so the longest common chain
    prefix is the exact token-alignment boundary between D's cached KV and
    the new request.
    """
    num_blocks = len(token_ids) // block_size
    if max_blocks is not None:
        num_blocks = min(num_blocks, max_blocks)
    hashes: list[str] = []
    parent = ""
    for block_index in range(num_blocks):
        block_tokens = token_ids[block_index * block_size : (block_index + 1) * block_size]
        parent = _bidir_block_hash(parent, block_tokens)
        hashes.append(parent)
    return hashes


class MooncakeSchedulerSendingThread(threading.Thread):
    """Serve worker metadata and collect D-side completion messages."""

    def __init__(
        self,
        host: str,
        port: int,
        engine_id: str,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        use_kv_pp: bool,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerSendingThread")
        encoder = msgspec.msgpack.Encoder()
        self.host = host
        self.port = port
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.engine_id = engine_id
        self.use_kv_pp = use_kv_pp
        if self.use_kv_pp and self.dcp_size != 1:
            raise ValueError(f"Mooncake KVPP cannot be combined with DCP, got dcp_size={self.dcp_size}")
        metadata_by_pp_rank = self._merge_metadata_by_pp_rank(metadata)
        self.encoded_metadata = encoder.encode(
            MooncakeTransferMetadataGroups(
                engine_id=engine_id,
                scheduler_host=host,
                scheduler_port=port,
                pp_size=pp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                tp_size=tp_size,
                use_kv_pp=self.use_kv_pp,
                metadata_by_pp_rank=metadata_by_pp_rank,
            )
        )
        logger.info(
            "Initialized Mooncake scheduler handshake metadata: engine_id=%s, "
            "worker_count=%d, pp_ranks=%s, tp_size=%d, pp_size=%d, "
            "pcp_size=%d, dcp_size=%d, use_kv_pp=%s",
            engine_id,
            len(metadata),
            sorted(metadata_by_pp_rank),
            tp_size,
            pp_size,
            pcp_size,
            dcp_size,
            self.use_kv_pp,
        )
        self.ready_event = ready_event
        self.stop_event = threading.Event()
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        # request_id -> block count; total feeds the D pinned-block budget.
        self.delayed_free_blocks: dict[str, int] = {}
        self.pinned_block_total = 0
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.state_lock = threading.Lock()

    def _merge_metadata_by_pp_rank(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> dict[int, MooncakePPTransferMetadata]:
        """Merge worker metadata into one PP-wide layer table.

        LayerSplit may assign different layers to any PCP/TP worker. Per-layer
        fields are therefore aligned by layer name instead of requiring every
        worker to expose the same layer list. Worker-private address tables are
        padded to the PP-union layer order, while ``layer_indices`` records the
        entries physically owned by each PCP/TP worker.
        """
        # pp_rank -> pcp_rank -> tp_rank -> worker handshake metadata.
        workers_by_pp_rank: dict[int, dict[int, dict[int, MooncakeTransferMetadata]]] = {}
        for metadata_key, rank_metadata in metadata.items():
            if isinstance(metadata_key, int):
                pp_rank, pcp_rank, tp_rank = 0, 0, metadata_key
            elif len(metadata_key) == 2:
                pp_rank, tp_rank = metadata_key
                pcp_rank = 0
            elif len(metadata_key) == 3:
                pp_rank, pcp_rank, tp_rank = metadata_key
            else:
                raise ValueError(
                    "Mooncake handshake metadata key must be tp_rank, "
                    f"(pp_rank, tp_rank), or (pp_rank, pcp_rank, tp_rank), got {metadata_key!r}"
                )

            if not isinstance(rank_metadata, MooncakeTransferMetadata):
                raise ValueError(
                    "Mooncake scheduler expects MooncakeTransferMetadata, "
                    f"got {type(rank_metadata).__name__} for key {metadata_key!r}"
                )
            if rank_metadata.engine_id != self.engine_id:
                raise ValueError(
                    "Mooncake worker metadata engine ID mismatch: expected "
                    f"{self.engine_id!r}, got {rank_metadata.engine_id!r}"
                )

            workers_by_tp_rank = workers_by_pp_rank.setdefault(pp_rank, {}).setdefault(pcp_rank, {})
            if tp_rank in workers_by_tp_rank:
                raise ValueError(
                    f"Duplicate Mooncake metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, TP rank {tp_rank}"
                )
            workers_by_tp_rank[tp_rank] = rank_metadata

        expected_pp_ranks = set(range(self.pp_size))
        if set(workers_by_pp_rank) != expected_pp_ranks:
            raise ValueError(
                "Mooncake worker metadata has incomplete PP ranks: expected "
                f"{sorted(expected_pp_ranks)}, got {sorted(workers_by_pp_rank)}"
            )

        expected_pcp_ranks = set(range(self.pcp_size))
        expected_tp_ranks = set(range(self.tp_size))
        merged: dict[int, MooncakePPTransferMetadata] = {}
        for pp_rank, workers_by_pcp_rank in sorted(workers_by_pp_rank.items()):
            if set(workers_by_pcp_rank) != expected_pcp_ranks:
                raise ValueError(
                    "Mooncake worker metadata has incomplete PCP ranks for "
                    f"PP rank {pp_rank}: expected {sorted(expected_pcp_ranks)}, "
                    f"got {sorted(workers_by_pcp_rank)}"
                )
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                if set(workers_by_tp_rank) != expected_tp_ranks:
                    raise ValueError(
                        "Mooncake worker metadata has incomplete TP ranks for "
                        f"PP rank {pp_rank}, PCP rank {pcp_rank}: "
                        f"expected {sorted(expected_tp_ranks)}, got {sorted(workers_by_tp_rank)}"
                    )

            reference_pcp_rank = min(workers_by_pcp_rank)
            reference_tp_rank = min(workers_by_pcp_rank[reference_pcp_rank])
            reference = workers_by_pcp_rank[reference_pcp_rank][reference_tp_rank]
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                for tp_rank, worker_metadata in workers_by_tp_rank.items():
                    mismatched_fields = [
                        field_name
                        for field_name in ("block_size", "num_blocks")
                        if getattr(worker_metadata, field_name) != getattr(reference, field_name)
                    ]
                    if mismatched_fields:
                        raise ValueError(
                            "Mooncake worker metadata differs across PCP/TP ranks for "
                            f"PP rank {pp_rank}: PCP/TP {reference_pcp_rank}/{reference_tp_rank} and "
                            f"{pcp_rank}/{tp_rank} mismatch in {mismatched_fields}"
                        )

            # layer_name -> (first owning worker metadata, its local layer index).
            # The source supplies PP-shared fields after all duplicate owners
            # have been checked against the corresponding structural signature.
            layer_source_by_name: dict[str, tuple[MooncakeTransferMetadata, int]] = {}
            # layer_name -> (group index, block size, strides, lengths, shapes,
            #                block-size scales).
            layer_signature_by_name: dict[str, tuple[object, ...]] = {}
            reference_layer_names: set[str] | None = None
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    worker_layer_names = set(worker_metadata.layer_names)
                    if len(worker_layer_names) != len(worker_metadata.layer_names):
                        raise ValueError(
                            f"Mooncake worker metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, "
                            f"TP rank {tp_rank} contains duplicate layers"
                        )
                    if reference_layer_names is None:
                        reference_layer_names = worker_layer_names
                    elif not self.use_kv_pp and worker_layer_names != reference_layer_names:
                        raise ValueError(
                            "Mooncake workers expose different KV-cache layers while KVPP is disabled: "
                            f"PP rank {pp_rank}, PCP/TP rank {pcp_rank}/{tp_rank}"
                        )
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_signature = (
                            worker_metadata.group_indices[local_layer_index],
                            worker_metadata.layer_block_sizes[local_layer_index],
                            worker_metadata.block_strides[local_layer_index],
                            worker_metadata.block_lens[local_layer_index],
                            worker_metadata.block_shapes[local_layer_index],
                            worker_metadata.block_size_scales[local_layer_index],
                        )
                        previous_signature = layer_signature_by_name.get(layer_name)
                        if previous_signature is not None and previous_signature != layer_signature:
                            layer_field_names = (
                                "group_indices",
                                "layer_block_sizes",
                                "block_strides",
                                "block_lens",
                                "block_shapes",
                                "block_size_scales",
                            )
                            mismatched_layer_fields = [
                                field_name
                                for field_name, previous_value, value in zip(
                                    layer_field_names, previous_signature, layer_signature
                                )
                                if previous_value != value
                            ]
                            raise ValueError(
                                "Mooncake worker metadata differs across PCP/TP ranks for "
                                f"PP rank {pp_rank}, layer {layer_name!r}: mismatch in {mismatched_layer_fields}"
                            )
                        if previous_signature is None:
                            layer_source_by_name[layer_name] = (worker_metadata, local_layer_index)
                            layer_signature_by_name[layer_name] = layer_signature
            layer_names = sorted(layer_source_by_name)
            layer_block_sizes: list[int] = []
            group_indices: list[int] = []
            block_strides: list[list[int]] = []
            block_lens: list[list[int]] = []
            block_shapes: list[list[tuple[int, ...]]] = []
            block_size_scales: list[list[int]] = []
            for layer_name in layer_names:
                worker_metadata, local_layer_index = layer_source_by_name[layer_name]
                layer_block_sizes.append(worker_metadata.layer_block_sizes[local_layer_index])
                group_indices.append(worker_metadata.group_indices[local_layer_index])
                block_strides.append(worker_metadata.block_strides[local_layer_index])
                block_lens.append(worker_metadata.block_lens[local_layer_index])
                block_shapes.append(worker_metadata.block_shapes[local_layer_index])
                block_size_scales.append(worker_metadata.block_size_scales[local_layer_index])

            layer_index_by_name = {layer_name: layer_index for layer_index, layer_name in enumerate(layer_names)}
            metadata_by_pcp_rank: dict[int, MooncakePCPTransferMetadata] = {}
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                metadata_by_tp_rank: dict[int, MooncakeTPTransferMetadata] = {}
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    layer_indices = sorted(
                        layer_index_by_name[layer_name] for layer_name in worker_metadata.layer_names
                    )
                    # PP-union layer index -> this worker's per-cache-tensor
                    # addresses; unowned entries remain empty.
                    aligned_base_addrs: list[list[int]] = [[] for _ in layer_names]
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_index = layer_index_by_name[layer_name]
                        aligned_base_addrs[layer_index] = worker_metadata.kv_caches_base_addr[local_layer_index]
                    metadata_by_tp_rank[tp_rank] = MooncakeTPTransferMetadata(
                        te_rpc_port=worker_metadata.te_rpc_port,
                        layer_indices=layer_indices,
                        kv_caches_base_addr=aligned_base_addrs,
                        local_ip=worker_metadata.local_ip,
                        handshake_port=worker_metadata.handshake_port,
                    )
                metadata_by_pcp_rank[pcp_rank] = MooncakePCPTransferMetadata(metadata_by_tp_rank=metadata_by_tp_rank)

            merged[pp_rank] = MooncakePPTransferMetadata(
                block_size=reference.block_size,
                num_blocks=reference.num_blocks,
                layer_names=layer_names,
                layer_block_sizes=layer_block_sizes,
                group_indices=group_indices,
                block_strides=block_strides,
                block_lens=block_lens,
                block_shapes=block_shapes,
                block_size_scales=block_size_scales,
                metadata_by_pcp_rank=metadata_by_pcp_rank,
            )
        return merged

    def add_delayed_request(self, request_id: str, delay_start_time: float) -> None:
        with self.state_lock:
            self.delayed_free_requests[request_id] = delay_start_time

    def add_delayed_request_with_blocks(self, request_id: str, expiry_time: float, num_blocks: int) -> None:
        """Track a delayed-free request together with its block count.

        The running total (self.pinned_block_total) feeds the scheduler's
        pinned-block budget so D can refuse to pin more than
        ``decoder_kv_block_budget`` blocks at once.
        """
        with self.state_lock:
            self.delayed_free_requests[request_id] = expiry_time
            self.delayed_free_blocks[request_id] = num_blocks
            self.pinned_block_total += num_blocks

    @property
    def total_pinned_blocks(self) -> int:
        with self.state_lock:
            return self.pinned_block_total

    def get_and_clear_finished_requests(self) -> set[str]:
        finished: set[str] = set()
        with self.state_lock:
            self._retrieve_expired_requests_locked()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.ident is not None:
            self.join()
        with self.state_lock:
            self.delayed_free_requests.clear()
        while not self.finished_requests.empty():
            self.finished_requests.get_nowait()

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host, self.port)
        try:
            logger.info("Mooncake scheduler sending thread listening on %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore[attr-defined]
                sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
                sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
                self.ready_event.set()
                self._run_busy_loop(sock)
        except Exception:
            self.ready_event.set()
            logger.exception("Mooncake scheduler sending thread failed on %s", path)

    def _run_busy_loop(self, sock: Any) -> None:
        decoder = msgspec.msgpack.Decoder(type=tuple)
        # Proactive expiry check interval: expired delayed-free blocks are
        # released periodically even when no new requests arrive, so the
        # pinned-block budget is replenished without depending on traffic.
        last_expiry_sweep = time.time()
        EXPIRY_SWEEP_INTERVAL_S = 30.0
        while not self.stop_event.is_set():
            try:
                frames = sock.recv_multipart()
            except zmq.Again:  # type: ignore[attr-defined]
                # ZMQ RCVTIMEO (1s) expired with no incoming message — use
                # this idle tick to proactively release expired blocks.
                now = time.time()
                if now - last_expiry_sweep >= EXPIRY_SWEEP_INTERVAL_S:
                    with self.state_lock:
                        self._retrieve_expired_requests_locked()
                    last_expiry_sweep = now
                continue

            identity = frames[0]
            payload = [frame for frame in frames[1:] if frame]
            if len(payload) != 1:
                logger.warning("Invalid Mooncake scheduler control frames: %s", frames)
                continue

            try:
                msg = decoder.decode(payload[0])
                if msg and msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", self.encoded_metadata))
                elif msg[0] == DONE_RECVING_MSG and len(msg) == 2:
                    self._handle_finished_request(str(msg[1]))
                    sock.send_multipart((identity, b"", ACK_MSG))
                else:
                    logger.warning("Unexpected Mooncake scheduler control message: %s", msg)
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_finished_request(self, request_id: str) -> None:
        with self.state_lock:
            self._mark_finished_locked(request_id)

    def _mark_finished_locked(self, request_id: str) -> None:
        # request_finished registers the request before exposing its transfer
        # parameters to D. Unknown IDs are therefore stale or duplicate DONEs;
        # the control loop still ACKs them so D can stop retrying.
        if request_id not in self.delayed_free_requests:
            logger.warning(
                "Received Mooncake DONE for request %s with no pending delayed free; ignoring completion",
                request_id,
            )
            return
        del self.delayed_free_requests[request_id]
        released = self.delayed_free_blocks.pop(request_id, None)
        if released is not None:
            self.pinned_block_total -= released
        self.finished_requests.put(request_id)

    def _retrieve_expired_requests_locked(self) -> None:
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            expiry_time = self.delayed_free_requests[request_id]

            # Check if expired (expiry_time is absolute timestamp)
            if current_time < expiry_time:
                break

            self._mark_finished_locked(request_id)
            logger.error(
                "Force freed expired Mooncake request %s at %.2f (expired at %.2f)",
                request_id,
                current_time,
                expiry_time,
            )

    def sweep_expired_blocks(self) -> int:
        """Proactively release blocks whose TTL has expired.

        Called from the scheduler's budget check (before deciding whether to
        reject a new request) and from the busy-loop idle sweep. Returns the
        number of expired requests freed.
        """
        with self.state_lock:
            before = len(self.delayed_free_requests)
            self._retrieve_expired_requests_locked()
            freed = before - len(self.delayed_free_requests)
        if freed > 0:
            logger.info(
                "Mooncake expiry sweep: freed %d expired delayed requests "
                "(%d blocks released, %d still pinned)",
                freed,
                self.pinned_block_total,
                len(self.delayed_free_requests),
            )
        return freed

    def evict_oldest_for_budget(self, needed_blocks: int, budget: int) -> int:
        """Evict the oldest delayed blocks (LRU) until enough budget is free.

        The delayed_free_requests OrderedDict is expiry-ordered (oldest
        first). Evicting from the front releases blocks closest to their
        TTL expiry — the least valuable for a next-turn pull. Returns the
        number of evicted requests.
        """
        evicted = 0
        with self.state_lock:
            while (
                self.pinned_block_total + needed_blocks > budget
                and self.delayed_free_requests
            ):
                request_id = next(iter(self.delayed_free_requests))
                self._mark_finished_locked(request_id)
                evicted += 1
                logger.warning(
                    "Mooncake LRU eviction: force-freed request %s to reclaim "
                    "budget (pinned=%d, needed=%d, budget=%d)",
                    request_id,
                    self.pinned_block_total,
                    needed_blocks,
                    budget,
                )
        return evicted


class MooncakeSchedulerRecvingThread(threading.Thread):
    """Send scheduler-level completion messages from D to P."""

    def __init__(self, ready_event: threading.Event) -> None:
        super().__init__(daemon=True, name="MooncakeSchedulerRecvingThread")
        self.ready_event = ready_event
        self.stop_event = threading.Event()
        self.request_queue: queue.Queue[tuple[str, int, str] | None] = queue.Queue()
        self.encoder = msgspec.msgpack.Encoder()
        self.remote_sockets: dict[str, deque[Any]] = defaultdict(deque)
        self.remote_sockets_lock = threading.Lock()
        self.zmq_context: Any | None = None

    def add_request(self, remote_host: str, remote_port: int, request_id: str) -> None:
        self.request_queue.put((remote_host, remote_port, request_id))

    def shutdown(self) -> None:
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.request_queue.put(None)
        if self.ident is not None:
            self.join()
        # Also handle shutdown before start and a sentinel enqueued while the
        # thread was already completing its final cleanup.
        self._clear_pending_requests()

    def _clear_pending_requests(self) -> None:
        while True:
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
            self.request_queue.task_done()

    def run(self) -> None:
        self.ready_event.set()
        try:
            while not self.stop_event.is_set():
                request = self.request_queue.get()
                try:
                    if request is None or self.stop_event.is_set():
                        break
                    self._send_done_recving(*request)
                except Exception:
                    # The socket helpers already retry. Let P's delayed-free
                    # timeout reclaim blocks if this notification cannot arrive.
                    logger.exception(
                        "Failed to send Mooncake scheduler completion for request %s; "
                        "producer cleanup will rely on its delayed-free timeout",
                        request[2] if request is not None else None,
                    )
                finally:
                    self.request_queue.task_done()
        finally:
            self.stop_event.set()
            # Close sockets in their owning thread, after the last send ends.
            with self.remote_sockets_lock:
                for sockets in self.remote_sockets.values():
                    for sock in sockets:
                        sock.close(linger=0)
                self.remote_sockets.clear()
                if self.zmq_context is not None:
                    self.zmq_context.destroy(linger=0)
                    self.zmq_context = None
            self._clear_pending_requests()

    def _send_done_recving(self, remote_host: str, remote_port: int, request_id: str) -> None:
        path = make_zmq_path("tcp", remote_host, remote_port)
        sock = self._get_remote_socket(path)
        try:
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_RECVING_MSG, request_id)),
                path,
            )
            response = ensure_zmq_recv(sock, path)
            if response != ACK_MSG:
                raise RuntimeError(f"Unexpected Mooncake scheduler completion response: {response!r}")
        except Exception:
            sock.close(linger=0)
            raise
        self._return_remote_socket(path, sock)

    def _get_remote_socket(self, path: str) -> Any:
        """Borrow a persistent REQ socket for one producer scheduler."""
        with self.remote_sockets_lock:
            sockets = self.remote_sockets[path]
            if sockets:
                return sockets.popleft()

            if self.zmq_context is None:
                self.zmq_context = zmq.Context()  # type: ignore[attr-defined]
            sock = make_zmq_socket(
                ctx=self.zmq_context,
                path=path,
                socket_type=zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
            return sock

    def _return_remote_socket(self, path: str, sock: Any) -> None:
        """Return a healthy REQ socket to its endpoint pool."""
        with self.remote_sockets_lock:
            self.remote_sockets[path].append(sock)


class MooncakePullConnectorScheduler(MooncakeBaseConnectorScheduler):
    """Scheduler-side Mooncake pull connector implementation."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Requests waiting for the worker to start a READ transfer.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, BlockIds, int]] = {}
        # Producer requests whose blocks must remain allocated until read.
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()
        # D request -> (P scheduler host, port, P request id).
        self._reqs_recv_info: dict[str, tuple[str, int, str]] = {}
        self._sending_thread: MooncakeSchedulerSendingThread | None = None
        self._recving_thread: MooncakeSchedulerRecvingThread | None = None
        # D-node bidirectional pinned-block budget: cap total blocks held for
        # D→P reuse so multi-turn bursts cannot exhaust the KV pool. When the
        # budget is exhausted, new requests skip delay-free entirely (blocks
        # are released normally; the next turn does a local prefill).
        total_blocks = kv_cache_config.num_blocks
        self.decoder_kv_block_budget = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_block_budget", max(1, total_blocks // 5)
            )
        )
        logger.info(
            "Bidirectional KV: decoder pinned-block budget=%d (pool=%d blocks)",
            self.decoder_kv_block_budget,
            total_blocks,
        )

        # The recving thread handles scheduler-to-scheduler DONE notifications
        # after pull transfers complete. It is needed by kv_consumer nodes
        # (standard D pulls from P) and by kv_producer nodes under bidirectional
        # transfer (P pulls from D and must ACK D on completion).
        needs_recving_thread = self.kv_transfer_config.is_kv_consumer or (
            self.is_bidirectional_kv_xfer_enabled and self.kv_transfer_config.is_kv_producer
        )
        if needs_recving_thread:
            recving_ready_event = threading.Event()
            self._recving_thread = MooncakeSchedulerRecvingThread(recving_ready_event)
            self._recving_thread.start()
            recving_ready_event.wait()

    def shutdown(self) -> None:
        if self._sending_thread is not None:
            self._sending_thread.shutdown()
        if self._recving_thread is not None:
            self._recving_thread.shutdown()
        self._reqs_need_recv.clear()
        self._reqs_need_send.clear()
        self._reqs_in_batch.clear()
        self._reqs_recv_info.clear()

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        # The sending thread is needed by kv_producer nodes (standard P→D) and
        # also by kv_consumer nodes when bidirectional KV transfer is enabled
        # (D must expose its blocks so P can pull them in the next turn).
        needs_sending_thread = self.kv_transfer_config.is_kv_producer or (
            self.is_bidirectional_kv_xfer_enabled and self.kv_transfer_config.is_kv_consumer
        )
        if not needs_sending_thread or not metadata or self._sending_thread is not None:
            return

        ready_event = threading.Event()
        self._sending_thread = MooncakeSchedulerSendingThread(
            self.side_channel_host,
            self.side_channel_port,
            self.engine_id,
            metadata,
            self.tp_size,
            self.pp_size,
            self.pcp_size,
            self.dcp_size,
            self.ascend_config.kvpp_config.size > 1,
            ready_event,
        )
        self._sending_thread.start()
        if not ready_event.wait(timeout=10):
            raise RuntimeError("Timed out starting Mooncake scheduler sending thread")
        if not self._sending_thread.is_alive():
            raise RuntimeError("Mooncake scheduler sending thread failed to start")

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        # Bidirectional KV transfer: P pulls blocks from D
        if (
            self.is_bidirectional_kv_xfer_enabled
            and params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # Use synchronous load (load_kv_async=False) with reserve-1 to avoid
            # the async load deadlock (WAITING_FOR_REMOTE_KVS state never resolves
            # because the scheduler doesn't produce a step to trigger start_load_kv).
            # The has_sync_kv_loads fix in patch_balance_schedule.py ensures
            # start_load_kv is called before the forward pass for sync loads.
            #
            # Alignment boundary (hash index preferred, conservative fallback):
            # 1. Hash index: if D reported a content-addressed block hash chain
            #    (remote_block_hashes), walk this request's prompt chain against
            #    it block by block. The longest matching chain prefix is the
            #    exact token-alignment boundary, which also recovers the
            #    assistant-text KV from the previous turn's decode output (the
            #    conservative bound deliberately excludes it).
            # 2. Conservative fallback: bound the pull by D's reported prompt
            #    length (remote_num_prompt_tokens). Only D's prompt-side KV is
            #    guaranteed token-aligned with this prompt; decode output
            #    (thinking tokens, template tails) does not reliably reappear.
            remote_block_hashes = params.get("remote_block_hashes") or None
            if remote_block_hashes:
                remote_block_size = params.get("remote_block_size") or self.block_size
                local_token_ids = list(request.prompt_token_ids or [])
                local_hashes = _bidir_block_hash_chain(
                    local_token_ids,
                    remote_block_size,
                    max_blocks=len(remote_block_hashes),
                )
                matched_blocks = 0
                for local_hash, remote_hash in zip(local_hashes, remote_block_hashes):
                    if local_hash != remote_hash:
                        break
                    matched_blocks += 1
                aligned_tokens = matched_blocks * remote_block_size
                # Full-sequence alignment check: all full blocks matched AND
                # the tail hash (tokens beyond the last full block) matches.
                # Only then does D's final state snapshot land exactly on the
                # resume boundary — required for stateful (hybrid) models.
                fully_aligned = False
                remote_full_len = params.get("remote_num_tokens") or 0
                remote_tail_hash = params.get("remote_tail_hash")
                if (
                    matched_blocks == len(remote_block_hashes)
                    and remote_tail_hash is not None
                    and remote_full_len > 0
                ):
                    tail_start = matched_blocks * remote_block_size
                    local_tail = local_token_ids[tail_start:remote_full_len]
                    if len(local_tail) == remote_full_len - tail_start:
                        tail_parent = remote_block_hashes[-1] if remote_block_hashes else ""
                        fully_aligned = (
                            _bidir_block_hash(tail_parent, local_tail) == remote_tail_hash
                        )
                        if fully_aligned:
                            aligned_tokens = remote_full_len
                if self.has_stateful_groups:
                    # Stateful models: D's mamba state is a single snapshot at
                    # the END of its sequence. Pulling a partial prefix would
                    # hand P a wrong initial state (fluent but incorrect
                    # output). Only a verified FULL alignment may be pulled,
                    # and only when the prompt continues past it (no
                    # reserve-1: the state is valid exactly at the boundary).
                    if not fully_aligned:
                        logger.info(
                            "Bidirectional KV: stateful model requires full-sequence "
                            "alignment, matched %d/%d blocks (tail=%s); skipping pull for %s",
                            matched_blocks,
                            len(remote_block_hashes),
                            "present" if remote_tail_hash is not None else "absent",
                            request.request_id,
                        )
                        return 0, False
                    count = aligned_tokens - num_computed_tokens
                    if count <= 0 or aligned_tokens >= request.num_prompt_tokens:
                        # Nothing new to compute: reserve-1 would shift the
                        # state boundary by one — unsafe on stateful models.
                        return 0, False
                    logger.info(
                        "Bidirectional KV: full-sequence alignment verified "
                        "(%d tokens), pulling for stateful request %s",
                        aligned_tokens,
                        request.request_id,
                    )
                    return count, False
                count = min(aligned_tokens, request.num_prompt_tokens) - num_computed_tokens
                logger.info(
                    "Bidirectional KV hash-align: matched %d blocks (%d tokens) of "
                    "%d remote hashes; prompt_tokens=%d, remote_prompt_tokens=%s, "
                    "num_computed_tokens=%d, external_count=%d",
                    matched_blocks,
                    aligned_tokens,
                    len(remote_block_hashes),
                    request.num_prompt_tokens,
                    params.get("remote_num_prompt_tokens"),
                    num_computed_tokens,
                    max(count, 0),
                )
            else:
                # Stateful models cannot use the conservative fallback: the
                # prompt-bound prefix alone never carries D's final state.
                if self.has_stateful_groups:
                    return 0, False
                # Conservative fallback: no hash chain reported (older D-side
                # metadata cached by the proxy, or D running a pre-hash build).
                remote_prompt_tokens = params.get("remote_num_prompt_tokens") or 0
                count = min(remote_prompt_tokens, request.num_prompt_tokens) - num_computed_tokens
            # Reserve the last token for local prefill to avoid num_new_tokens=0.
            if count >= request.num_prompt_tokens - num_computed_tokens:
                count = max(count - 1, 0)
            if count > 0:
                return count, False

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._log_mamba_request_allocation(request.request_id, blocks)
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)

        # Standard P->D prefill path
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                required_remote_fields = (
                    "remote_engine_id",
                    "remote_host",
                    "remote_port",
                    "remote_request_id",
                )
                if all(field in params for field in required_remote_fields):
                    remote = (
                        params["remote_host"],
                        params["remote_port"],
                        params["remote_request_id"],
                    )
                    if num_external_tokens > 0:
                        self._reqs_need_recv[request.request_id] = (
                            request,
                            blocks.get_unhashed_block_ids_all_groups(),
                            blocks.get_block_ids(),
                            num_external_tokens,
                        )
                        self._reqs_recv_info[request.request_id] = remote
                    else:
                        if self._recving_thread is None:
                            raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                        self._recving_thread.add_request(*remote)
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s.", params)
            else:
                assert num_external_tokens == 0

            # Only trigger one transfer for a request.
            params["do_remote_prefill"] = False
            return

        # Bidirectional KV transfer: P pulls blocks from D
        if (
            self.is_bidirectional_kv_xfer_enabled
            and params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and not params.get("_remote_blocks_processed")
        ):
            required_remote_fields = (
                "remote_engine_id",
                "remote_host",
                "remote_port",
                "remote_request_id",
            )
            if all(field in params for field in required_remote_fields):
                remote = (
                    params["remote_host"],
                    params["remote_port"],
                    params["remote_request_id"],
                )
                if num_external_tokens > 0:
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        blocks.get_unhashed_block_ids_all_groups(),
                        blocks.get_block_ids(),
                        num_external_tokens,
                    )
                    self._reqs_recv_info[request.request_id] = remote
                else:
                    if self._recving_thread is None:
                        raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                    self._recving_thread.add_request(*remote)
            else:
                logger.warning("Got invalid KVTransferParams for bidirectional transfer. params=%s.", params)

            # Mark as processed to avoid duplicate transfers
            params["_remote_blocks_processed"] = True

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        for (
            req_id,
            (req, block_ids, full_block_ids, num_external_tokens),
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                local_full_block_ids=full_block_ids,
                local_num_prompt_tokens=req.num_prompt_tokens,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        self._reqs_need_recv.clear()
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Expose completed producer blocks for a remote READ transfer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished: request_status=%s, kv_transfer_params=%s",
            request.status,
            params,
        )

        # Handle case where do_remote_prefill is still set (request aborted before scheduling)
        if params is not None and params.get("do_remote_prefill"):
            required_remote_fields = (
                "remote_host",
                "remote_port",
                "remote_request_id",
            )
            if all(field in params for field in required_remote_fields):
                remote = (
                    params["remote_host"],
                    params["remote_port"],
                    params["remote_request_id"],
                )
                if self._recving_thread is None:
                    raise RuntimeError("Producer Mooncake scheduler cannot acknowledge a receive request")
                self._recving_thread.add_request(*remote)
            else:
                logger.warning("Got invalid KVTransferParams. params=%s.", params)

            params["do_remote_prefill"] = False
            return False, None

        # Determine if this is a P node or D node based on the instance role.
        # In bidirectional mode, the kv_producer instance (P) does prefill and
        # the kv_consumer instance (D) does decode. We use the connector's
        # kv_role rather than the request params, because the params flag
        # do_remote_decode is set by the proxy on both P's prefill response
        # and D's incoming request, making it unreliable for direction detection.
        is_p_node = self.kv_transfer_config.is_kv_producer
        is_d_node = self.kv_transfer_config.is_kv_consumer

        # For D node without bidirectional KV transfer, don't delay free
        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        # Check request status - only delay free for successful completions
        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            return False, None

        # Calculate blocks to potentially delay. For the D node under
        # bidirectional transfer, extend the advertised block set to the FULL
        # token sequence (prompt + decode output) so the next turn's P node
        # can also pull the decode-generated KV (assistant text), not just the
        # prompt-side KV.
        prompt_token_ids = request.prompt_token_ids or []
        prompt_len = len(prompt_token_ids)
        output_token_ids = list(request.output_token_ids or [])
        if is_d_node and self.is_bidirectional_kv_xfer_enabled:
            full_seq = list(prompt_token_ids) + output_token_ids
            advertised_len = len(full_seq)
        else:
            full_seq = list(prompt_token_ids)
            advertised_len = prompt_len
        computed_block_ids = self._get_transfer_block_ids(block_ids, advertised_len)
        num_computed_blocks = sum(len(group_block_ids) for group_block_ids in computed_block_ids)
        delay_free_blocks = num_computed_blocks > 0

        if not delay_free_blocks:
            return False, None

        # D-node memory budget: cap total pinned bidirectional blocks so a
        # burst of multi-turn sessions cannot exhaust the KV pool. When the
        # budget is exceeded, first sweep expired blocks (proactive release
        # of blocks whose TTL has passed but were not yet freed because no
        # engine step triggered the opportunistic check). If still short,
        # evict the OLDEST delayed blocks (LRU) — sessions that have not sent
        # a next-turn pull are less valuable than fresh sessions. Only when
        # the pool itself is too small do we skip delay-free entirely.
        if is_d_node and self.is_bidirectional_kv_xfer_enabled:
            current_pinned = (
                self._sending_thread.total_pinned_blocks if self._sending_thread is not None else 0
            )
            if current_pinned + num_computed_blocks > self.decoder_kv_block_budget:
                # Step 1: sweep expired blocks (free blocks past their TTL).
                if self._sending_thread is not None:
                    self._sending_thread.sweep_expired_blocks()
                    current_pinned = self._sending_thread.total_pinned_blocks
                # Step 2: evict oldest (LRU) until enough budget is available.
                if self._sending_thread is not None:
                    evicted = self._sending_thread.evict_oldest_for_budget(
                        num_computed_blocks, self.decoder_kv_block_budget
                    )
                    if evicted > 0:
                        logger.info(
                            "Bidirectional KV: evicted %d oldest delayed blocks "
                            "(LRU) to make budget for request %s",
                            evicted,
                            request.request_id,
                        )
                    current_pinned = self._sending_thread.total_pinned_blocks
                if current_pinned + num_computed_blocks > self.decoder_kv_block_budget:
                    logger.warning(
                        "Bidirectional KV: block budget exhausted (%d+%d > %d after "
                        "expiry sweep + LRU eviction), skipping delay-free for request %s",
                        current_pinned,
                        num_computed_blocks,
                        self.decoder_kv_block_budget,
                        request.request_id,
                    )
                    return False, None

        # Content-addressed block hash chain over the full sequence. Only the
        # D node reports it; the next turn's P node walks its own prompt chain
        # against this chain to find the exact token-alignment boundary.
        remote_block_hashes: list[str] = []
        remote_tail_hash: str | None = None
        if is_d_node and self.is_bidirectional_kv_xfer_enabled:
            remote_block_hashes = _bidir_block_hash_chain(full_seq, BIDIR_HASH_BLOCK_SIZE)
            # Tail hash: covers the tokens beyond the last full hash block
            # (advertised_len % BIDIR_HASH_BLOCK_SIZE != 0). Stateful (hybrid)
            # models may only pull when the ENTIRE D-side sequence aligns with
            # the next prompt prefix, because D's mamba state is a single
            # snapshot at the end of the sequence; the tail hash makes the
            # full-alignment boundary verifiable.
            tail_start = len(remote_block_hashes) * BIDIR_HASH_BLOCK_SIZE
            if advertised_len > tail_start:
                tail_parent = remote_block_hashes[-1] if remote_block_hashes else ""
                remote_tail_hash = _bidir_block_hash(
                    tail_parent, full_seq[tail_start:advertised_len]
                )
            logger.debug(
                "Bidirectional KV: reporting %d block hashes over %d tokens "
                "(prompt=%d + output=%d, tail=%s) for request %s",
                len(remote_block_hashes),
                advertised_len,
                prompt_len,
                len(output_token_ids),
                "yes" if remote_tail_hash is not None else "no",
                request.request_id,
            )

        # Determine TTL based on node type
        if is_d_node:
            # D node: use decoder_kv_blocks_ttl for bidirectional transfer
            ttl = self.decoder_kv_blocks_ttl
            logger.info(
                "Bidirectional KV: Delaying free of %d blocks for request %s with TTL %d seconds",
                num_computed_blocks,
                request.request_id,
                ttl,
            )
        else:
            # P node: use standard abort timeout
            ttl = envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            logger.info(
                "Delaying free of %d blocks for request %s with TTL %d seconds",
                num_computed_blocks,
                request.request_id,
                ttl,
            )

        # Store expiry timestamp
        expiry_time = time.time() + ttl
        self._reqs_need_send[request.request_id] = expiry_time
        if self._sending_thread is None:
            raise RuntimeError("Mooncake scheduler metadata has not been initialized")
        if is_d_node and self.is_bidirectional_kv_xfer_enabled:
            self._sending_thread.add_delayed_request_with_blocks(
                request.request_id, expiry_time, num_computed_blocks
            )
        else:
            self._sending_thread.add_delayed_request(request.request_id, expiry_time)

        # Build response metadata. For the D node, remote_num_tokens is the
        # EXACT advertised sequence length (prompt + output) so the next-turn
        # P node can verify full-sequence alignment against the tail hash.
        remote_num_tokens = advertised_len if is_d_node else 0
        blocks_expiry_time = expiry_time if is_d_node else None

        return delay_free_blocks, {
            "do_remote_prefill": is_p_node,
            "do_remote_decode": is_d_node,
            "remote_block_ids": computed_block_ids,
            "remote_num_tokens": remote_num_tokens,
            "remote_num_prompt_tokens": prompt_len,
            "remote_block_hashes": remote_block_hashes,
            "remote_tail_hash": remote_tail_hash,
            "remote_block_size": BIDIR_HASH_BLOCK_SIZE,
            "remote_engine_id": self.engine_id,
            "remote_request_id": request.request_id,
            "remote_host": self.side_channel_host,
            "remote_port": self.side_channel_port,
            "remote_blocks_expiry_time": blocks_expiry_time,
        }

    def update_connector_output(
        self,
        connector_output: KVConnectorOutput,
    ) -> None:
        # D side (standard P→D): this output has already aggregated completion
        # from all workers. Send one scheduler-to-scheduler ACK for the request.
        # P side (bidirectional D→P): same ACK semantics — notify D that its
        # blocks have been read so it can release them. On P the upstream
        # scheduler must NOT see finished_recving (the request already
        # completed via the synchronous load path; _update_from_kv_xfer_finished
        # would assert because the request has been freed). Clear the sets on
        # the kv_producer side after consuming them here.
        is_p_node = self.kv_transfer_config.is_kv_producer
        for req_id in list(connector_output.finished_recving or ()):
            remote = self._reqs_recv_info.pop(req_id, None)
            if remote is not None:
                if self._recving_thread is None:
                    raise RuntimeError("Producer Mooncake scheduler received a receive-completion event")
                self._recving_thread.add_request(*remote)
                if req_id in (connector_output.failed_recving or set()):
                    logger.error(
                        "Mooncake D receive failed after worker aggregation: request_id=%s, "
                        "remote_request_id=%s, producer=%s:%s, failure_policy=%s; producer release notification queued",
                        req_id,
                        remote[2],
                        remote[0],
                        remote[1],
                        self.kv_transfer_config.kv_load_failure_policy,
                    )
        if is_p_node and self.is_bidirectional_kv_xfer_enabled:
            # Consume and drop: P's synchronous bidir loads complete within the
            # scheduling step; upstream must not process them as async loads.
            connector_output.finished_recving = None
            connector_output.failed_recving = None

        # P side: feed scheduler-received ACKs into vLLM's standard delayed
        # free path. Scheduler._update_from_kv_xfer_finished reads this same
        # KVConnectorOutput immediately after this hook returns.
        finished_sending = (
            self._sending_thread.get_and_clear_finished_requests() if self._sending_thread is not None else set()
        )
        if finished_sending:
            for req_id in finished_sending:
                self._reqs_need_send.pop(req_id, None)
            if connector_output.finished_sending is None:
                connector_output.finished_sending = finished_sending
            else:
                connector_output.finished_sending.update(finished_sending)


__all__ = [
    "MooncakePullConnectorScheduler",
    "MooncakeSchedulerRecvingThread",
    "MooncakeSchedulerSendingThread",
]
