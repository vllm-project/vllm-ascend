# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side implementation entry point for Mooncake pull transfers."""

import queue
import threading
from typing import Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePPTransferMetadata,
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
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        layer_name_to_group_index: dict[str, int],
        layer_name_to_spec_index: dict[str, int],
        local_metadata: MooncakeTransferMetadata,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
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
        self.model_config = vllm_config.model_config
        self.kv_cache_config = kv_cache_config
        self.kv_cache_specs = kv_cache_specs
        self.layer_name_to_group_index = layer_name_to_group_index
        self.layer_name_to_spec_index = layer_name_to_spec_index
        self.local_metadata = local_metadata

        self.block_size = local_metadata.block_size
        self.spec_block_sizes = local_metadata.spec_block_sizes
        self.spec_head_sizes = local_metadata.spec_head_sizes
        self.layer_names = local_metadata.layer_names
        self.group_indices = local_metadata.group_indices
        self.spec_indices = local_metadata.spec_indices
        self.kv_caches_base_addr = local_metadata.kv_caches_base_addr
        self.block_strides = local_metadata.block_strides
        self.block_lens = local_metadata.block_lens
        self.block_shapes = local_metadata.block_shapes
        self.block_size_scales = local_metadata.block_size_scales

        self.use_mla = self.model_config.is_deepseek_mla
        hf_text_config = self.model_config.hf_text_config
        self.num_key_value_heads = getattr(
            hf_text_config, "num_key_value_heads", 0
        )
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
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
        self.remote_metadata: SizedDict[
            str, dict[int, MooncakePPTransferMetadata]
        ] = SizedDict()
        self.request_queue: queue.Queue[tuple[str, dict[str, ReqMeta]]] = queue.Queue()
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.invalid_block_ids: set[int] = set()
        self.invalid_block_ids_lock = threading.Lock()
        assert self.local_metadata is not None

    def add_requests(
        self,
        remote_engine_id: str,
        requests: dict[str, ReqMeta],
    ) -> None:
        """Queue requests for one remote engine."""
        if requests:
            self.request_queue.put((remote_engine_id, requests))

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
        torch.npu.set_device(self.device)
        self.ready_event.set()
        while True:
            remote_engine_id, requests = self.request_queue.get()
            request_endpoints = {
                (request.remote_host, request.remote_port)
                for request in requests.values()
            }
            try:
                remote_host, remote_port = self._get_remote_endpoint(
                    request_endpoints
                )
                self._handle_requests(
                    remote_engine_id,
                    remote_host,
                    remote_port,
                    requests,
                )
            except Exception as exc:
                for request_metadata in requests.values():
                    self._mark_request_failed(request_metadata)
                logger.exception(
                    "Mooncake pull failed for remote engine %s at %s: %s",
                    remote_engine_id,
                    sorted(request_endpoints),
                    exc,
                )
            finally:
                for request_id in requests:
                    self.finished_requests.put(request_id)
                self.request_queue.task_done()

    def _mark_request_failed(self, request_metadata: ReqMeta) -> None:
        with self.invalid_block_ids_lock:
            for group_block_ids in request_metadata.local_block_ids:
                self.invalid_block_ids.update(group_block_ids)

    @staticmethod
    def _get_remote_endpoint(
        endpoints: set[tuple[str, int]],
    ) -> tuple[str, int]:
        if len(endpoints) != 1:
            raise ValueError(
                "Requests for one remote engine must share one scheduler "
                f"endpoint, got {sorted(endpoints)}"
            )
        return next(iter(endpoints))

    def _handle_requests(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_port: int,
        requests: dict[str, ReqMeta],
    ) -> None:
        """Fetch peer metadata and submit its aggregated requests."""
        self._get_remote_metadata(remote_engine_id, remote_host, remote_port)

        raise NotImplementedError("Mooncake pull transfer submission is not implemented")

    @staticmethod
    def _validate_remote_metadata(
        metadata: MooncakeTransferMetadataGroups,
        expected_engine_id: str,
    ) -> None:
        if metadata.engine_id != expected_engine_id:
            raise ValueError(
                "Mooncake producer metadata engine ID mismatch: expected "
                f"{expected_engine_id!r}, got {metadata.engine_id!r}"
            )
        if not metadata.metadata_by_pp_rank:
            raise ValueError(
                "Mooncake producer scheduler returned no PP metadata"
            )

        empty_pp_ranks = [
            pp_rank
            for pp_rank, pp_metadata in metadata.metadata_by_pp_rank.items()
            if not pp_metadata.metadata_by_tp_rank
        ]
        if empty_pp_ranks:
            raise ValueError(
                "Mooncake producer scheduler returned no TP metadata for "
                f"PP ranks {sorted(empty_pp_ranks)}"
            )

        invalid_tp_ranks = {
            tp_rank
            for pp_metadata in metadata.metadata_by_pp_rank.values()
            for tp_rank in pp_metadata.metadata_by_tp_rank
            if tp_rank < 0 or tp_rank >= pp_metadata.tp_size
        }
        if invalid_tp_ranks:
            raise ValueError(
                "Mooncake producer scheduler returned invalid TP ranks "
                f"{sorted(invalid_tp_ranks)}"
            )
    def _get_remote_metadata(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_port: int,
    ) -> dict[int, MooncakePPTransferMetadata]:
        cached_metadata = self.remote_metadata.get(remote_engine_id)
        if cached_metadata is not None:
            return cached_metadata

        path = make_zmq_path("tcp", remote_host, remote_port)
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.SNDTIMEO, 1000)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ensure_zmq_send(
                sock,
                self.encoder.encode((b"get_meta_msg",)),
                path,
            )
            metadata_bytes = ensure_zmq_recv(sock, path)

        if not metadata_bytes:
            raise RuntimeError(
                "Mooncake producer scheduler returned no transfer metadata "
                f"for engine {remote_engine_id!r} from {path}"
            )

        transfer_metadata = self.decoder.decode(metadata_bytes)
        self._validate_remote_metadata(transfer_metadata, remote_engine_id)
        remote_metadata = transfer_metadata.metadata_by_pp_rank
        self.remote_metadata[remote_engine_id] = remote_metadata
        return remote_metadata


class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker-side framework for Mooncake pull transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
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
                kv_cache_config=self.kv_cache_config,
                kv_cache_specs=self.kv_cache_specs,
                layer_name_to_group_index=self.layer_name_to_group_index,
                layer_name_to_spec_index=self.layer_name_to_spec_index,
                local_metadata=self.xfer_handshake_metadata,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
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
            assert self._recving_thread is not None
            request_groups: dict[str, dict[str, ReqMeta]] = {}
            for request_id, request_metadata in metadata.requests.items():
                requests = request_groups.setdefault(
                    request_metadata.remote_engine_id, {}
                )
                requests[request_id] = request_metadata

            for remote_engine_id, requests in request_groups.items():
                self._recving_thread.add_requests(remote_engine_id, requests)

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
