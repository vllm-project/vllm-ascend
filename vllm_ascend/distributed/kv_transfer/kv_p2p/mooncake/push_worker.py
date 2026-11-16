# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layerwise Push: proxy-provided destinations, P WRITE pool and D DONE listener."""

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from typing import Any

import msgspec
import torch
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from zmq.constants import SocketOption, SocketType
from zmq.error import Again, ZMQError
from zmq.sugar.context import Context

from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadataGroups,
    PushSendReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

# Rank tuples are (PP, PCP, TP), matching scheduler handshake keys.
WorkerRank = tuple[int, int, int]


@dataclass
class _ReceiveState:
    request_id: str
    start_time: float
    expected_ranks: set[WorkerRank] = field(default_factory=set)
    finished_ranks: set[WorkerRank] = field(default_factory=set)
    failed: bool = False


class MooncakeDecodeListenerThread(threading.Thread):
    """Listen for P worker DONE messages on D; acknowledge and queue results."""

    def __init__(self, worker: "MooncakePushConnectorWorker") -> None:
        super().__init__(daemon=True, name="MooncakeDecodeListenerThread")
        self.stop_event = worker._stop_event
        self.ready_event = threading.Event()
        self.error: Exception | None = None
        self.engine_id = worker.engine_id
        self.rank = [worker.pp_rank, worker.pcp_rank, worker.tp_rank]
        self.address = make_zmq_path("tcp", worker.side_channel_host, worker.handshake_port)
        self.result_queue = worker._result_queue

    def start(self) -> None:
        super().start()
        if not self.ready_event.wait(timeout=10):
            self.shutdown()
            raise TimeoutError(f"Timed out starting {self.name}")
        if self.error is not None:
            self.shutdown()
            raise RuntimeError(f"{self.name} failed to start") from self.error

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.ident is not None:
            self.join()

    def run(self) -> None:
        try:
            with zmq_ctx(SocketType.ROUTER, self.address) as sock:
                sock.setsockopt(SocketOption.RCVTIMEO, 100)
                sock.setsockopt(SocketOption.SNDTIMEO, 1000)
                self.address = sock.getsockopt(SocketOption.LAST_ENDPOINT).decode()
                self.ready_event.set()
                while not self.stop_event.is_set():
                    try:
                        frames = sock.recv_multipart()
                    except Again:
                        continue
                    if self.stop_event.is_set():
                        return
                    self._handle_done_message(sock, frames)
        except Exception as exc:
            self.error = exc
            logger.exception("%s failed", self.name)
        finally:
            self.ready_event.set()

    def _handle_done_message(self, sock: Any, frames: list[bytes]) -> None:
        identity, *payload = frames
        # Support REQ (empty delimiter) and DEALER callbacks.
        delimiter = [b""] if payload and payload[0] == b"" else []
        if delimiter:
            payload = payload[1:]
        completion = None
        try:
            if len(payload) != 1:
                raise ValueError("Expected one completion payload")
            message = msgspec.msgpack.decode(payload[0])
            if not isinstance(message, dict) or message.get("kind") != "done":
                raise ValueError("Expected DONE")
            if message.get("consumer_engine_id") != self.engine_id or message.get("consumer_rank") != self.rank:
                raise ValueError("DONE belongs to another consumer")
            rank = message.get("producer_rank")
            if not isinstance(rank, list) or len(rank) != 3 or any(type(v) is not int or v < 0 for v in rank):
                raise ValueError("Invalid producer rank")
            sizes = message.get("producer_sizes")
            if (
                not isinstance(sizes, list)
                or len(sizes) != 3
                or any(type(size) is not int or size <= 0 for size in sizes)
                or any(value >= size for value, size in zip(rank, sizes))
            ):
                raise ValueError("Invalid producer topology")
            for key in ("request_id", "producer_engine_id"):
                if not isinstance(message.get(key), str) or not message[key]:
                    raise ValueError(f"Missing completion identity: {key}")
            if message.get("status") not in ("ok", "error"):
                raise ValueError("DONE requires a transfer status")
            completion = message
            reply = {"kind": "done_ack", "request_id": message["request_id"]}
        except Exception as exc:
            # Isolate malformed network messages, including decoder errors
            # such as RecursionError, from the shared listener's lifetime.
            logger.warning("Failed to parse Mooncake DONE message: %s", exc)
            reply = {"kind": "error", "error": str(exc)}
        try:
            sock.send_multipart([identity, *delimiter, msgspec.msgpack.encode(reply)])
        except ZMQError:
            logger.exception("Failed to acknowledge Push completion")
            if completion is not None:
                # Preserve request failure while waiting for the other sources.
                completion["status"] = "error"
        if completion is not None:
            self.result_queue.put(("done", completion["request_id"], completion))


@dataclass(frozen=True)
class LayerTransferRequest:
    request_id: str
    local_block_ids: tuple[int, ...]
    remote_block_ids: tuple[int, ...]
    start_token: int
    end_token: int


@dataclass(frozen=True)
class LayerTransferBatch:
    """One layer/destination WRITE batch; snapshots outlive the model step."""

    layer_name: str
    remote_rank: WorkerRank
    remote_metadata: MooncakeTransferMetadataGroups
    requests: tuple[LayerTransferRequest, ...]
    event: Any


@dataclass
class RemoteReqInfo:
    transfer_params: dict[str, Any]
    metadata_future: Future | None  # Aborts before dispatch need only DONE routing.
    required_layers: set[str] | None = None
    sent_tokens: dict[str, int] = field(default_factory=dict)
    pending_layers: set[str] = field(default_factory=set)
    completion: Future | None = None


@dataclass
class SendReqInfo:
    local_request_id: str
    metadata: PushSendReqMeta | None = None
    start_time: float = field(default_factory=time.monotonic)
    # Available end token and device fence for each persistent cache layer.
    available_layers: dict[str, tuple[int, Any]] = field(default_factory=dict)
    destinations: dict[WorkerRank, RemoteReqInfo] = field(default_factory=dict)
    expected_ranks: set[WorkerRank] | None = None
    consumer_engine_id: str | None = None
    inference_finished: bool = False
    error: str | None = None


class MooncakePushConnectorWorker(MooncakeBaseConnectorWorker):
    """Coordinate P layer batches and D completion accounting."""

    def __init__(self, vllm_config, engine_id, kv_cache_config) -> None:
        # TODO: add DCP rank/sequence mapping and completion accounting before
        # accepting DCP > 1; current worker identities contain only PP/PCP/TP.
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError("Mooncake Push worker currently requires DCP=1")
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._stop_event = threading.Event()
        self._result_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._reqs_recv_state: dict[str, _ReceiveState] = {}
        self._finished_requests: SizedDict = SizedDict(max_size=16000)
        self._decode_listener_thread: MooncakeDecodeListenerThread | None = None
        self._pending_completions: SizedDict = SizedDict(max_size=16000)
        self._reqs_send_state: dict[str, SendReqInfo] = {}
        self._prefilling_request_ids: tuple[str, ...] = ()
        self._pending_sends: list[tuple[LayerTransferBatch, Future]] = []
        self._sending_futures: dict[str, list[Future]] = {}
        self._remote_metadata: SizedDict = SizedDict(max_size=128)
        self._sender_executor: ThreadPoolExecutor | None = None
        self._zmq_context: Context | None = None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor | list[torch.Tensor]"]) -> None:
        super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.is_kv_producer:
            # KVPP is outside the Push support plan.
            if self.ascend_config.kvpp_config.size > 1:
                raise ValueError("Mooncake Push does not support KVPP yet")
            self._device = torch.npu.current_device()
            self._layer_indices = {name: i for i, name in enumerate(self.transfer_metadata.layer_names)}
            model_type = self.vllm_config.model_config.hf_text_config.model_type
            num_attn_module = 2 if model_type in ("longcat_flash", "longcat_flash_ngram") else 1
            self._layer_index_by_name: dict[str, tuple[str, int]] = {}
            for name in self._layer_indices:
                layer_index = extract_layer_index(name, num_attn_module)
                parts = name.split(".")
                namespace = ".".join(parts[: next(i for i, part in enumerate(parts) if part.isdecimal())])
                # Target and draft layers can reuse layer numbers, but not KV readiness.
                self._layer_index_by_name[name] = (namespace, layer_index)
            self._layer_names_by_index: dict[tuple[str, int], list[str]] = {}
            for name, layer_index in self._layer_index_by_name.items():
                self._layer_names_by_index.setdefault(layer_index, []).append(name)
            # Layer numbers do not establish execution order across models.
            # Multiple namespaces finalize at wait_for_save after draft execution.
            namespaces = {namespace for namespace, _ in self._layer_names_by_index}
            self._last_layer_index = max(self._layer_names_by_index) if len(namespaces) == 1 else None
            # Shared Mamba address calculation uses the same registered descriptors as Pull.
            for name in ("layer_names", "kv_caches_base_addr", "block_shapes", "block_strides", "block_lens"):
                setattr(self, name, getattr(self.transfer_metadata, name))
            self._zmq_context = Context()
            self._sender_executor = ThreadPoolExecutor(
                max_workers=32,
                initializer=torch.npu.set_device,
                initargs=(self._device,),
                thread_name_prefix="MooncakePushSending",
            )
            return
        self._decode_listener_thread = MooncakeDecodeListenerThread(self)
        try:
            self._decode_listener_thread.start()
        except Exception:
            self.shutdown()
            raise

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        if self.kv_transfer_config.is_kv_producer:
            # The previous step's wait_for_save has joined its WRITE tasks.
            for local_request_id, request in metadata.push_send_requests.items():
                state = self._reqs_send_state.setdefault(request.request_id, SendReqInfo(local_request_id))
                state.metadata = request
                if not state.destinations:
                    self._prepare_destinations(state, request)
                # A prefix hit is already resident before this forward.
                if not state.available_layers and request.num_computed_tokens:
                    event = torch.npu.Event()
                    event.record()
                    state.available_layers = {
                        name: (request.num_computed_tokens, event) for name in self._layer_indices
                    }
            for state in self._reqs_send_state.values():
                if state.local_request_id in metadata.requests_to_send:
                    state.inference_finished = True
            for local_request_id, params in metadata.push_aborted_requests.items():
                request_id = params["remote_request_id"]
                state = self._reqs_send_state.setdefault(request_id, SendReqInfo(local_request_id))
                state.inference_finished = True
                state.error = "P request aborted"
                if not state.destinations:
                    # No WRITE was dispatched: notify D without querying cache
                    # metadata. Existing states retain their WRITE dependencies.
                    for pp, pcp, tp, address in params["remote_worker_addresses"]:
                        rank = (pp, pcp, tp)
                        state.destinations[rank] = RemoteReqInfo(
                            dict(
                                request_id=request_id,
                                producer_engine_id=self.engine_id,
                                producer_rank=[self.pp_rank, self.pcp_rank, self.tp_rank],
                                consumer_engine_id=params["remote_engine_id"],
                                consumer_rank=list(rank),
                                completion_address=address,
                                num_computed_tokens=0,
                                num_external_tokens=0,
                            ),
                            metadata_future=None,
                        )
            self._prefilling_request_ids = tuple(req.request_id for req in metadata.push_send_requests.values())
            self._process_sending_requests()

        else:
            for request_id, recv_request in metadata.push_requests.items():
                if (
                    recv_request.request_id in self._finished_requests
                    or recv_request.request_id in self._reqs_recv_state
                ):
                    continue
                self._reqs_recv_state[recv_request.request_id] = _ReceiveState(request_id, time.monotonic())
                for completion in self._pending_completions.pop(recv_request.request_id, {}).values():
                    self._result_queue.put(("done", recv_request.request_id, completion))
            for request_id, not_dispatched in metadata.push_failed_requests.items():
                recv_state = self._reqs_recv_state.get(request_id)
                if recv_state is None:
                    # Already finished, or no external KV was needed.
                    continue
                recv_state.failed = True
                if not_dispatched:
                    self._result_queue.put(("failed", request_id, None))
                # Otherwise wait for all P ranks or the receive timeout.
                # Mark failure before consuming queued DONEs.

    def _get_remote_metadata(self, params: dict[str, Any]) -> MooncakeTransferMetadataGroups:
        path = params["scheduler_address"]
        sock = make_zmq_socket(self._zmq_context, path, SocketType.REQ, bind=False)
        try:
            timeout = max(1, int(envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT * 1000))
            sock.setsockopt(SocketOption.SNDTIMEO, timeout)
            sock.setsockopt(SocketOption.RCVTIMEO, timeout)
            ensure_zmq_send(sock, msgspec.msgpack.encode((b"get_meta_msg", params["consumer_engine_id"])), path)
            metadata = msgspec.msgpack.decode(ensure_zmq_recv(sock, path), type=MooncakeTransferMetadataGroups)
        finally:
            sock.close(linger=0)
        if metadata.engine_id != params["consumer_engine_id"] or metadata.dcp_size != 1:
            raise ValueError("Push requires matching D engine metadata with DCP=1")
        if metadata.use_kv_pp:
            raise ValueError("Push does not support KVPP yet")
        return metadata

    def _send_done(self, params: dict[str, Any], error: str | None) -> bool:
        """A failed callback never retries the request or retains P source blocks."""
        path = params["completion_address"]
        sock = None
        try:
            sock = make_zmq_socket(self._zmq_context, path, SocketType.REQ, bind=False)
            timeout = max(1, int(envs.VLLM_MOONCAKE_CONNECTOR_TIMEOUT * 1000))
            sock.setsockopt(SocketOption.SNDTIMEO, timeout)
            sock.setsockopt(SocketOption.RCVTIMEO, timeout)
            message = {
                key: params[key]
                for key in (
                    "request_id",
                    "producer_engine_id",
                    "producer_rank",
                    "consumer_engine_id",
                    "consumer_rank",
                )
            }
            message.update(
                kind="done",
                status="error" if error else "ok",
                producer_sizes=[self.pp_size, self.pcp_size, self.tp_size],
            )
            if error:
                message["error"] = error
            ensure_zmq_send(sock, msgspec.msgpack.encode(message), path)
            reply = msgspec.msgpack.decode(ensure_zmq_recv(sock, path))
            if reply != {"kind": "done_ack", "request_id": params["request_id"]}:
                raise ValueError("Invalid DONE ACK")
            return True
        except Exception:
            # D may have died. Its receive timeout handles a missing DONE.
            logger.exception("Failed to notify Push completion for %s at %s", params["request_id"], path)
            return False
        finally:
            if sock is not None:
                sock.close(linger=0)

    def _prepare_destinations(self, state: SendReqInfo, request: PushSendReqMeta) -> None:
        path = make_zmq_path("tcp", request.remote_host, request.remote_port)
        params = dict(
            request_id=request.request_id,
            producer_engine_id=self.engine_id,
            producer_rank=[self.pp_rank, self.pcp_rank, self.tp_rank],
            consumer_engine_id=request.remote_engine_id,
            scheduler_address=path,
            block_ids=request.remote_block_ids,
            num_computed_tokens=request.remote_num_computed_tokens,
            num_external_tokens=request.remote_num_external_tokens,
        )
        key = (request.remote_engine_id, path)
        future = self._remote_metadata.get(key)
        if future is not None and future.done() and (future.cancelled() or future.exception() is not None):
            # Retry discovery for new requests after a transient D failure.
            future = None
        if future is None:
            assert self._sender_executor is not None
            future = self._sender_executor.submit(self._get_remote_metadata, params)
            self._remote_metadata[key] = future
        state.consumer_engine_id = request.remote_engine_id
        for pp, pcp, tp, address in request.remote_worker_addresses:
            rank = (pp, pcp, tp)
            target = dict(params, consumer_rank=list(rank), completion_address=address)
            state.destinations[rank] = RemoteReqInfo(target, future)
        if not state.destinations:
            raise ValueError("Missing D worker completion endpoints")

    def _get_transfer_partition(
        self, layer_name: str, remote: MooncakeTransferMetadataGroups, rank: WorkerRank
    ) -> tuple[int, int, int] | None:
        """Return source/destination head offsets and count, or None for a non-owner."""
        local_index = self._layer_indices[layer_name]
        pp = remote.metadata_by_pp_rank[rank[0]]
        if layer_name not in pp.layer_names:
            return None
        remote_index = pp.layer_names.index(layer_name)
        worker = pp.metadata_by_pcp_rank[rank[1]].metadata_by_tp_rank[rank[2]]
        if remote_index not in worker.layer_indices:
            return None
        spec = self.kv_cache_specs[self.layer_name_to_spec_index[layer_name]]
        local_rank = self.tp_rank
        remote_rank = rank[2]
        local_size, remote_size = self.tp_size, remote.tp_size
        if isinstance(spec, MambaSpec):
            if getattr(spec, "tp_replicated", False):
                owner = (rank[2] * remote.pcp_size + rank[1]) % (self.tp_size * self.pcp_size)
                return (0, 0, 1) if self.tp_rank * self.pcp_size + self.pcp_rank == owner else None
            local_rank = self.tp_rank * self.pcp_size + self.pcp_rank
            remote_rank = rank[2] * remote.pcp_size + rank[1]
            local_size *= self.pcp_size
            remote_size *= remote.pcp_size
            larger, smaller = max(local_size, remote_size), min(local_size, remote_size)
            if larger % smaller:
                raise ValueError("Mamba P/D PCP*TP sizes must have an integer ratio")
            ratio = larger // smaller
            matched = (
                local_rank // ratio == remote_rank if local_size >= remote_size else remote_rank // ratio == local_rank
            )
            return (0, 0, 1) if matched else None
        whole = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) or is_circular_kv_cache_spec(spec)
        if whole:
            # Replicated pages: exactly one P worker writes each D replica.
            owner = (rank[2] * remote.pcp_size + rank[1]) % (self.tp_size * self.pcp_size)
            return (0, 0, 1) if self.tp_rank * self.pcp_size + self.pcp_rank == owner else None
        if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            raise ValueError(f"Unsupported Push cache spec: {type(spec).__name__}")
        if self.pcp_rank != rank[1] % self.pcp_size:
            return None
        local_heads = self.transfer_metadata.block_shapes[local_index][0][0]
        remote_heads = pp.block_shapes[remote_index][0][0]
        totals = {heads * size for heads, size in ((local_heads, local_size), (remote_heads, remote_size)) if heads > 1}
        if len(totals) > 1:
            raise ValueError("Inconsistent P/D attention head counts")
        total_heads = next(iter(totals), min(local_size, remote_size))
        if max(total_heads, local_size) % min(total_heads, local_size) or max(total_heads, remote_size) % min(
            total_heads, remote_size
        ):
            raise ValueError("Unsupported attention head partition")
        # Where TP exceeds KV heads, choose only one P replica for each head.
        replicas = max(1, local_size // total_heads)
        if local_rank % replicas != remote_rank % replicas:
            return None
        local_start = local_rank // replicas * local_heads
        remote_start = remote_rank // max(1, remote_size // total_heads) * remote_heads
        start, end = max(local_start, remote_start), min(local_start + local_heads, remote_start + remote_heads)
        return (start - local_start, start - remote_start, end - start) if start < end else None

    def _build_layer_transfer_request(
        self, request_id: str, state: SendReqInfo, destination: RemoteReqInfo, layer_name: str
    ) -> tuple[LayerTransferRequest, Any] | None:
        meta = state.metadata
        assert meta is not None
        params = destination.transfer_params
        available, event = state.available_layers[layer_name]
        end = min(available, params["num_computed_tokens"] + params["num_external_tokens"])
        start = destination.sent_tokens.get(layer_name, params["num_computed_tokens"])
        if start >= end:
            return None
        assert destination.metadata_future is not None
        remote = destination.metadata_future.result()
        pp = remote.metadata_by_pp_rank[params["consumer_rank"][0]]
        local_index = self._layer_indices[layer_name]
        remote_index = pp.layer_names.index(layer_name)
        spec = self.kv_cache_specs[self.layer_name_to_spec_index[layer_name]]
        final = available >= meta.num_prompt_tokens
        final_only = (
            isinstance(spec, MambaSpec)
            or is_circular_kv_cache_spec(spec)
            or getattr(spec, "sliding_window", None) is not None
        )
        if final_only and not final:
            return None
        local_blocks = meta.local_block_ids[self.transfer_metadata.group_indices[local_index]]
        remote_blocks = params["block_ids"][pp.group_indices[remote_index]]
        local_ids: list[int]
        remote_ids: list[int]
        if isinstance(spec, MambaSpec):
            local_ids = [local_blocks[-1]]
            remote_ids = [remote_blocks[-1]]
            if not local_ids[0] or not remote_ids[0]:
                raise ValueError("Required Mamba state block is not resident")
        elif is_circular_kv_cache_spec(spec):
            if len(local_blocks) != 1 or len(remote_blocks) != 1:
                raise ValueError("Circular Push requires one state block per request")
            local_scale = self.transfer_metadata.block_size_scales[local_index][0]
            remote_scale = pp.block_size_scales[remote_index][0]
            if local_scale != remote_scale:
                raise ValueError("Circular Push requires equal P/D physical block counts")
            local_ids = [local_blocks[0] * local_scale + i for i in range(local_scale)]
            remote_ids = [remote_blocks[0] * remote_scale + i for i in range(remote_scale)]
        else:
            local_scale = self.transfer_metadata.block_size_scales[local_index][0]
            remote_scale = pp.block_size_scales[remote_index][0]
            block_size = self.transfer_metadata.layer_block_sizes[local_index] // local_scale
            if block_size != pp.layer_block_sizes[remote_index] // remote_scale:
                raise ValueError("Push requires equal P/D kernel block sizes")
            if not final:
                end = end // block_size * block_size
            if end <= start:
                return None
            if final_only:
                # Window tables retain logical positions, including null slots.
                start = max(start, end - getattr(spec, "num_retained_tokens", spec.sliding_window - 1) - 1)
            first, last = start // block_size, (end + block_size - 1) // block_size
            local_ids, remote_ids = [], []
            for index in range(first, last):
                local_block = local_blocks[index // local_scale]
                remote_block = remote_blocks[index // remote_scale]
                if final_only and remote_block == 0:
                    continue
                if local_block == 0 or remote_block == 0:
                    raise ValueError("Required Push block is not resident")
                local_ids.append(local_block * local_scale + index % local_scale)
                remote_ids.append(remote_block * remote_scale + index % remote_scale)
        return LayerTransferRequest(request_id, tuple(local_ids), tuple(remote_ids), start, end), event

    def _execute_layer_transfer(self, batch: LayerTransferBatch) -> str | None:
        # Backend contract: synchronous WRITE returns only after host/device
        # accesses finish, including on failure. The connector waits for this
        # return before notifying D or allowing source blocks to be reused.
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        try:
            local_index = self._layer_indices[batch.layer_name]
            remote = batch.remote_metadata
            pp = remote.metadata_by_pp_rank[batch.remote_rank[0]]
            remote_index = pp.layer_names.index(batch.layer_name)
            worker = pp.metadata_by_pcp_rank[batch.remote_rank[1]].metadata_by_tp_rank[batch.remote_rank[2]]
            spec = self.kv_cache_specs[self.layer_name_to_spec_index[batch.layer_name]]
            if isinstance(spec, MambaSpec):
                self._append_mamba_transfer_addresses(
                    spec,
                    batch.remote_rank[1],
                    batch.remote_rank[2],
                    remote.tp_size,
                    {
                        (local_index, remote_index): [
                            (req.request_id, list(req.local_block_ids), list(req.remote_block_ids))
                            for req in batch.requests
                        ]
                    },
                    pp,
                    src_list,
                    dst_list,
                    length_list,
                )
            else:
                partition = self._get_transfer_partition(batch.layer_name, remote, batch.remote_rank)
                assert partition is not None
                local_offset, remote_offset, heads = partition
                whole = isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)) or is_circular_kv_cache_spec(spec)
                local = self.transfer_metadata
                for cache_index, (local_addr, remote_addr) in enumerate(
                    zip(local.kv_caches_base_addr[local_index], worker.kv_caches_base_addr[remote_index], strict=True)
                ):
                    local_len = local.block_lens[local_index][cache_index]
                    remote_len = pp.block_lens[remote_index][cache_index]
                    if not whole:
                        local_len //= local.block_shapes[local_index][cache_index][0]
                        remote_len //= pp.block_shapes[remote_index][cache_index][0]
                    if local_len != remote_len:
                        raise ValueError("P/D cache slice lengths differ")
                    for req in batch.requests:
                        for local_block, remote_block in zip(req.local_block_ids, req.remote_block_ids, strict=True):
                            src_list.append(
                                local_addr
                                + local_block * local.block_strides[local_index][cache_index]
                                + local_offset * local_len
                            )
                            dst_list.append(
                                remote_addr
                                + remote_block * pp.block_strides[remote_index][cache_index]
                                + remote_offset * remote_len
                            )
                            length_list.append(local_len * heads)
            batch.event.synchronize()
            if src_list:
                session = f"{worker.local_ip}:{worker.te_rpc_port}"
                ret = self.engine.batch_transfer_sync_write(session, src_list, dst_list, length_list)
                if ret < 0:
                    return f"Mooncake WRITE failed: session={session}, ret={ret}"
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None

    def _process_sending_requests(self, wait: bool = False, *, layer_hook: bool = False, force: bool = False) -> None:
        """Collect WRITE results and submit ready layers; never send DONE here."""
        pending = []
        for batch, future in self._pending_sends:
            if not wait and not future.done():
                pending.append((batch, future))
                continue
            try:
                error = future.result()
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            if error:
                logger.error(
                    "Push transfer failed: layer=%s, rank=%s, error=%s", batch.layer_name, batch.remote_rank, error
                )
            for req in batch.requests:
                state = self._reqs_send_state[req.request_id]
                destination = state.destinations[batch.remote_rank]
                destination.pending_layers.remove(batch.layer_name)
                if error:
                    state.error = error
                else:
                    destination.sent_tokens[batch.layer_name] = req.end_token
        self._pending_sends = pending
        self._sending_futures = {
            engine_id: unfinished
            for engine_id, futures in self._sending_futures.items()
            if (unfinished := [future for future in futures if not future.done()])
        }
        batches: dict[
            tuple[str, WorkerRank, str, int],
            tuple[MooncakeTransferMetadataGroups, Any, list[LayerTransferRequest]],
        ] = {}
        # A free engine flushes every ready layer accumulated while it was busy.
        # The last layer and the step boundary bypass this submission gate.
        request_ids = self._prefilling_request_ids if layer_hook else self._reqs_send_state
        now = time.monotonic() if wait else 0
        for request_id in request_ids:
            state = self._reqs_send_state[request_id]
            # Metadata/WRITE waits may have crossed the request's deadline.
            if wait and state.error is None and now - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                state.error = "Push transfer timed out"
            if state.error or state.metadata is None:
                continue
            assert state.consumer_engine_id is not None
            if not (wait or force) and state.consumer_engine_id in self._sending_futures:
                continue
            for rank, destination in state.destinations.items():
                if state.error or destination.completion is not None:
                    continue
                assert destination.metadata_future is not None
                if not destination.metadata_future.done():
                    continue
                try:
                    remote = destination.metadata_future.result()
                    if state.expected_ranks is None:
                        expected = {
                            (pp, pcp, tp)
                            for pp in range(remote.pp_size)
                            for pcp in range(remote.pcp_size)
                            for tp in range(remote.tp_size)
                        }
                        if set(state.destinations) != expected:
                            raise ValueError("D worker endpoints do not match its topology")
                        state.expected_ranks = expected
                    if destination.required_layers is None:
                        destination.required_layers = {
                            name
                            for name in self._layer_indices
                            if self._get_transfer_partition(name, remote, rank) is not None
                        }
                    for layer_name in state.available_layers:
                        if layer_name not in destination.required_layers or layer_name in destination.pending_layers:
                            continue
                        task = self._build_layer_transfer_request(request_id, state, destination, layer_name)
                        if task is not None:
                            request, event = task
                            key = (state.consumer_engine_id, rank, layer_name, id(event))
                            entry = batches.setdefault(key, (remote, event, []))
                            entry[2].append(request)
                            destination.pending_layers.add(layer_name)
                except Exception as exc:
                    state.error = str(exc) or type(exc).__name__
                    logger.exception("Failed to prepare Push transfer %s", request_id)
        for (engine_id, rank, layer_name, _), (remote, event, requests) in batches.items():
            # A later layer/destination may have failed while building this pass.
            active = []
            for req in requests:
                state = self._reqs_send_state[req.request_id]
                if state.error is None:
                    active.append(req)
                else:
                    state.destinations[rank].pending_layers.remove(layer_name)
            if active:
                batch = LayerTransferBatch(layer_name, rank, remote, tuple(active), event)
                assert self._sender_executor is not None
                future = self._sender_executor.submit(self._execute_layer_transfer, batch)
                self._pending_sends.append((batch, future))
                self._sending_futures.setdefault(engine_id, []).append(future)

    def _send_done_after_transfers(self, params: dict[str, Any], error: str | None, transfers: list[Future]) -> bool:
        # This task is submitted after all its dependencies. Its Future includes
        # the notification itself, unlike a WRITE Future's add_done_callback.
        for future in transfers:
            try:
                transfer_error = future.result()
            except Exception as exc:
                transfer_error = str(exc) or type(exc).__name__
            error = error or transfer_error
        return self._send_done(params, error)

    def _send_completions(self) -> None:
        """Queue DONE after a request's complete set of submitted WRITEs."""
        transfers_by_request: dict[str, list[Future]] = {}
        pending_by_request: dict[str, dict[WorkerRank, dict[str, int]]] = {}
        for batch, future in self._pending_sends:
            for request in batch.requests:
                transfers_by_request.setdefault(request.request_id, []).append(future)
                pending_tokens = pending_by_request.setdefault(request.request_id, {})
                pending_tokens.setdefault(batch.remote_rank, {})[batch.layer_name] = request.end_token
        for request_id, state in self._reqs_send_state.items():
            transfers = transfers_by_request.get(request_id, [])
            pending_tokens = pending_by_request.get(request_id, {})
            for rank, destination in state.destinations.items():
                if destination.completion is not None:
                    continue
                pending = pending_tokens.get(rank, {})
                end = (
                    destination.transfer_params["num_computed_tokens"]
                    + destination.transfer_params["num_external_tokens"]
                )
                complete = destination.required_layers is not None and all(
                    max(destination.sent_tokens.get(layer, -1), pending.get(layer, -1)) >= end
                    for layer in destination.required_layers
                )
                if state.error or complete:
                    assert self._sender_executor is not None
                    destination.completion = self._sender_executor.submit(
                        self._send_done_after_transfers, destination.transfer_params, state.error, transfers
                    )

    def save_kv_layer(self, layer_name: str) -> None:
        if not self.kv_transfer_config.is_kv_producer or not self._prefilling_request_ids:
            return
        layer_index = self._layer_index_by_name.get(layer_name)
        if layer_index is None:
            return
        # Share one fence across the physical layer's caches and requests.
        # Duplicate hooks with no new tokens do not record another device event.
        layer_names = self._layer_names_by_index[layer_index]
        event = None
        for request_id in self._prefilling_request_ids:
            state = self._reqs_send_state[request_id]
            if state.error:
                continue
            assert state.metadata is not None
            end = state.metadata.num_computed_tokens + state.metadata.num_scheduled_tokens
            for cache_name in layer_names:
                if state.available_layers.get(cache_name, (-1, None))[0] < end:
                    if event is None:
                        event = torch.npu.Event()
                        event.record()
                    state.available_layers[cache_name] = end, event
        last_layer = layer_index == self._last_layer_index
        if event is not None or last_layer:
            self._process_sending_requests(layer_hook=True, force=last_layer)
        if last_layer:
            self._send_completions()

    def wait_for_save(self) -> None:
        if not self.kv_transfer_config.is_kv_producer or not self._reqs_send_state:
            return
        # Includes cancelled requests and connector-only steps. Check deadlines
        # at the step boundary instead of on every layer hook.
        now = time.monotonic()
        for state in self._reqs_send_state.values():
            if state.error is None and now - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                state.error = "Push transfer timed out"
        # End-of-forward fallback for Mamba/auxiliary caches without named hooks.
        event = None
        for request_id in self._prefilling_request_ids:
            state = self._reqs_send_state[request_id]
            if state.error:
                continue
            assert state.metadata is not None
            end = state.metadata.num_computed_tokens + state.metadata.num_scheduled_tokens
            for layer_name in self._layer_indices:
                if state.available_layers.get(layer_name, (-1, None))[0] < end:
                    if event is None:
                        event = torch.npu.Event()
                        event.record()
                    state.available_layers[layer_name] = end, event
        # A late metadata query must not leave this step's KV unsent when core
        # handles normal completion and immediately releases the source blocks.
        wait_futures(
            {
                dest.metadata_future
                for state in self._reqs_send_state.values()
                if state.error is None
                for dest in state.destinations.values()
                if dest.completion is None and dest.metadata_future is not None
            }
        )
        self._process_sending_requests(wait=True)
        # Drain tasks just submitted above before the next step can overwrite
        # window/state pages or the scheduler can recycle any source blocks.
        while self._pending_sends:
            self._process_sending_requests(wait=True)
        self._send_completions()
        wait_futures(
            dest.completion
            for state in self._reqs_send_state.values()
            for dest in state.destinations.values()
            if dest.completion is not None
        )
        self._prefilling_request_ids = ()

    def maybe_raise_thread_error(self) -> None:
        thread = self._decode_listener_thread
        if thread is not None and thread.error is not None:
            raise RuntimeError("Push completion listener failed") from thread.error

    def get_transfer_results(self) -> KVConnectorTransferResults:
        self.maybe_raise_thread_error()
        results = KVConnectorTransferResults()
        if self.kv_transfer_config.is_kv_producer:
            for request_id, send_state in list(self._reqs_send_state.items()):
                if send_state.inference_finished and all(
                    not dest.pending_layers and dest.completion is not None and dest.completion.done()
                    for dest in send_state.destinations.values()
                ):
                    results.finished_sending.add(send_state.local_request_id)
                    del self._reqs_send_state[request_id]
        else:
            while True:
                try:
                    kind, request_id, payload = self._result_queue.get_nowait()
                except queue.Empty:
                    break
                if request_id in self._finished_requests:
                    continue
                state = self._reqs_recv_state.get(request_id)
                if state is None:
                    # Proxy may dispatch P before D's allocation metadata reaches
                    # this worker. Preserve early DONE, deduplicated by source.
                    pending = self._pending_completions.setdefault(request_id, {})
                    source_key = (
                        payload["producer_engine_id"],
                        tuple(payload["producer_sizes"]),
                        tuple(payload["producer_rank"]),
                        payload["status"],
                    )
                    pending[source_key] = payload
                    continue
                if kind == "failed":
                    results.finished_recving.add(state.request_id)
                    results.failed_recving.add(state.request_id)
                    self._finished_requests[request_id] = True
                    del self._reqs_recv_state[request_id]
                    continue
                if not state.expected_ranks:
                    sizes = payload["producer_sizes"]
                    state.expected_ranks = {
                        (pp, pcp, tp) for pp in range(sizes[0]) for pcp in range(sizes[1]) for tp in range(sizes[2])
                    }
                source = tuple(payload["producer_rank"])
                state.failed |= payload["status"] == "error"
                if source in state.finished_ranks:
                    continue
                state.finished_ranks.add(source)
                if state.finished_ranks == state.expected_ranks:
                    results.finished_recving.add(state.request_id)
                    if state.failed:
                        results.failed_recving.add(state.request_id)
                    self._finished_requests[request_id] = True
                    del self._reqs_recv_state[request_id]
            for request_id, state in list(self._reqs_recv_state.items()):
                if time.monotonic() - state.start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                    logger.error("Missing terminal Push completion for request %s", state.request_id)
                    results.finished_recving.add(state.request_id)
                    results.failed_recving.add(state.request_id)
                    self._finished_requests[request_id] = True
                    del self._reqs_recv_state[request_id]
        return results

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Request results and control errors are handled by get_transfer_results.
        return set()

    def shutdown(self) -> None:
        self._stop_event.set()
        for thread in (self._decode_listener_thread,):
            if thread is not None:
                thread.shutdown()
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True)
            self._sender_executor = None
        if self._zmq_context is not None:
            self._zmq_context.destroy(linger=0)
            self._zmq_context = None
