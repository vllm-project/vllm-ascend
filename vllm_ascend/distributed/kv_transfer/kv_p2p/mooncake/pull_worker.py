# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side implementation entry point for Mooncake pull transfers."""

import queue
import threading
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class MooncakePullRecvingThread(threading.Thread):
    """D-side execution thread for Mooncake pull requests.

    The queue/result lifecycle is intentionally implemented separately from
    the transfer algorithm. Metadata lookup, block mapping, and Mooncake READ
    submission belong in _handle_requests.
    """

    def __init__(
        self,
        engine: Any,
        vllm_config: VllmConfig,
        local_metadata: MooncakeTransferMetadata,
        tp_rank: int,
        tp_size: int,
        dp_rank: int,
        dp_size: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_rank: int,
        dcp_size: int,
        device: Any,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="MooncakePullRecvingThread")
        self.engine = engine
        self.vllm_config = vllm_config
        self.local_metadata = local_metadata
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.dcp_rank = dcp_rank
        self.dcp_size = dcp_size
        assert self.pcp_size == 1, (
            "Mooncake pull worker temporarily requires pcp_size=1, "
            f"got {self.pcp_size}"
        )
        self.device = device
        self.ready_event = ready_event
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
        self.remote_metadata: SizedDict = SizedDict()
        self.request_queue: queue.Queue[tuple[str, int, dict[str, ReqMeta]]] = queue.Queue()
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.invalid_block_ids: set[int] = set()
        self.invalid_block_ids_lock = threading.Lock()
        assert self.local_metadata is not None

    def add_requests(
        self,
        remote_engine_id: str,
        remote_port: int,
        requests: dict[str, ReqMeta],
    ) -> None:
        """Queue requests for one remote engine and scheduler port."""
        if requests:
            self.request_queue.put((remote_engine_id, remote_port, requests))

    def get_and_clear_finished_requests(self) -> set[str]:
        """Drain requests whose transfer attempt has completed."""
        finished: set[str] = set()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Drain local block IDs affected by failed pull attempts."""
        with self.invalid_block_ids_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def run(self) -> None:
        """Consume queued requests and invoke the transfer implementation."""
        try:
            torch.npu.set_device(self.device)
        except Exception:
            self.ready_event.set()
            logger.exception("Failed to initialize the Mooncake pull thread device")
            return

        self.ready_event.set()
        while True:
            remote_engine_id, remote_port, requests = self.request_queue.get()
            try:
                self._handle_requests(remote_engine_id, remote_port, requests)
            except Exception:
                for request_metadata in requests.values():
                    self._mark_request_failed(request_metadata)
                logger.exception(
                    "Mooncake pull failed for remote engine %s, port %d",
                    remote_engine_id,
                    remote_port,
                )
            finally:
                for request_id in requests:
                    self.finished_requests.put(request_id)
                self.request_queue.task_done()

    def _mark_request_failed(self, request_metadata: ReqMeta) -> None:
        with self.invalid_block_ids_lock:
            for group_block_ids in request_metadata.local_block_ids:
                self.invalid_block_ids.update(group_block_ids)

    def _handle_requests(
        self,
        remote_engine_id: str,
        remote_port: int,
        requests: dict[str, ReqMeta],
    ) -> None:
        """Fetch peer metadata and submit its aggregated requests."""
        remote_hosts = {request.remote_host for request in requests.values()}
        if len(remote_hosts) != 1:
            raise ValueError(
                "Mooncake requests grouped for one remote engine and port must "
                f"share one host, got {sorted(remote_hosts)}"
            )
        remote_host = next(iter(remote_hosts))
        self._get_remote_metadata(remote_engine_id, remote_host, remote_port)

        # Transfer planning and Mooncake READ submission are implemented next.
        # Until then, do not report an incomplete transfer as successful.
        raise NotImplementedError("Mooncake pull transfer submission is not implemented")

    def _get_remote_metadata(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_port: int,
    ) -> MooncakeTransferMetadataGroups:
        cache_key = (
            remote_engine_id,
            remote_host,
            remote_port,
            self.tp_rank,
        )
        cached_metadata = self.remote_metadata.get(cache_key)
        if cached_metadata is not None:
            return cached_metadata

        path = make_zmq_path("tcp", remote_host, remote_port)
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.SNDTIMEO, 1000)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ensure_zmq_send(
                sock,
                self.encoder.encode((b"get_meta_msg", self.tp_rank)),
                path,
            )
            metadata_bytes = ensure_zmq_recv(sock, path)

        if not metadata_bytes:
            raise RuntimeError(
                "Mooncake producer scheduler returned no transfer metadata "
                f"for TP rank {self.tp_rank} from {path}"
            )

        remote_metadata = self.decoder.decode(metadata_bytes)
        if remote_metadata.tp_rank != self.tp_rank:
            raise ValueError(
                "Mooncake producer scheduler returned metadata for TP rank "
                f"{remote_metadata.tp_rank}, expected {self.tp_rank}"
            )
        if not remote_metadata.metadata_by_pp_rank:
            raise ValueError(
                "Mooncake producer scheduler returned no PP metadata for "
                f"TP rank {self.tp_rank}"
            )
        unexpected_engine_ids = {
            metadata.engine_id
            for metadata in remote_metadata.metadata_by_pp_rank.values()
            if metadata.engine_id != remote_engine_id
        }
        if unexpected_engine_ids:
            raise ValueError(
                "Mooncake producer metadata engine ID mismatch: expected "
                f"{remote_engine_id!r}, got {sorted(unexpected_engine_ids)!r}"
            )

        self.remote_metadata[cache_key] = remote_metadata
        return remote_metadata


class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker-side framework for Mooncake pull transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._recving_thread: MooncakePullRecvingThread | None = None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register caches and start the D-side pull execution thread."""
        super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self._recving_thread = MooncakePullRecvingThread(
                engine=self.engine,
                vllm_config=self.vllm_config,
                local_metadata=self.xfer_handshake_metadata,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                pcp_rank=self.pcp_rank,
                pcp_size=self.pcp_size,
                dcp_rank=self.dcp_rank,
                dcp_size=self.dcp_size,
                device=torch.npu.current_device(),
                ready_event=ready_event,
            )
            self._recving_thread.start()
            if not ready_event.wait(timeout=10):
                raise RuntimeError("Timed out starting Mooncake pull receiving thread")
            if not self._recving_thread.is_alive():
                raise RuntimeError("Mooncake pull receiving thread failed to start")

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start the pull operations described by scheduler metadata."""
        if not metadata.requests:
            return
        if self.kv_transfer_config.is_kv_consumer:
            request_groups: dict[str, dict[int, dict[str, ReqMeta]]] = {}
            for request_id, request_metadata in metadata.requests.items():
                req_engine_id, req_remote_port = request_metadata.remote_engine_id, request_metadata.remote_port
                requests_by_engine = request_groups.setdefault(req_engine_id, {})
                requests_by_port = requests_by_engine.setdefault(req_remote_port, {})
                requests_by_port[request_id] = request_metadata

            for remote_engine_id, requests_by_port in request_groups.items():
                for remote_port, requests_batch in requests_by_port.items():
                    self._recving_thread.add_requests(
                        remote_engine_id,
                        remote_port,
                        requests_batch,
                    )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        finished_recving = (
            self._recving_thread.get_and_clear_finished_requests()
            if self._recving_thread is not None
            else set()
        )
        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose pull operations failed."""
        if self._recving_thread is None:
            return set()
        return self._recving_thread.get_and_clear_invalid_block_ids()


__all__ = [
    "MooncakePullConnectorWorker",
    "MooncakePullRecvingThread",
]
