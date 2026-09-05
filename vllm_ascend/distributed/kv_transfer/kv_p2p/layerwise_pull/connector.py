# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Backend-independent layerwise pull KV-transfer connector."""

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import LayerwisePullHandshakeMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.scheduler import (
    LayerwisePullConsumerScheduler,
    LayerwisePullProducerScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.worker import (
    LayerwisePullConsumerWorker,
    LayerwisePullProducerWorker,
    _validate_tcp_port,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request


class LayerwisePullConnector(KVConnectorBase_V1, SupportsHMA):
    """Let Decode pull ready layer buffers exposed by Prefill.

    * SCHEDULER + producer : P-side request/block metadata.
    * SCHEDULER + consumer : D-side destination block tracking.
    * WORKER + producer    : P-side layer-wise READ_READY notifications.
    * WORKER + consumer    : D-side reads through the configured backend.

    Sparse KV offload is one optional destination-layout adapter. It is not a
    connector prerequisite and does not affect the wire protocol.
    """

    supports_layerwise_buffer_reuse = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_consumer = vllm_config.kv_transfer_config.is_kv_consumer
        self.requires_full_blocks_on_update_after_alloc = role == KVConnectorRole.SCHEDULER and self.is_producer
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.use_layerwise = extra_config.get("use_layerwise", True)

        if role == KVConnectorRole.SCHEDULER:
            if self.is_producer:
                self.connector_scheduler = LayerwisePullProducerScheduler(
                    vllm_config,
                    kv_cache_config,
                )
            else:
                self.connector_scheduler = LayerwisePullConsumerScheduler(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )
            self.connector_worker = None
        else:
            self.connector_scheduler = None
            if self.is_producer:
                self.connector_worker = LayerwisePullProducerWorker(vllm_config, kv_cache_config)
            else:
                self.connector_worker = LayerwisePullConsumerWorker(
                    vllm_config,
                    self.use_layerwise,
                    kv_cache_config,
                )

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_handshake_metadata(self) -> LayerwisePullHandshakeMetadata:
        worker = self.connector_worker
        assert worker is not None
        if self.is_producer:
            return LayerwisePullHandshakeMetadata(tuple(sorted(worker.layer_layouts)))
        reader = worker._read_thread
        if reader is None or not reader.ready_event.is_set() or reader.startup_error is not None:
            raise RuntimeError("Layerwise pull D listener must be ready before publishing its endpoint")
        return LayerwisePullHandshakeMetadata(
            tuple(sorted(worker.layer_layouts)), reader._host, reader.side_channel_port + reader.tp_rank
        )

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        scheduler = self.connector_scheduler
        assert scheduler is not None
        pc = scheduler.vllm_config.parallel_config
        expected = {(pp, tp) for pp in range(pc.pipeline_parallel_size) for tp in range(pc.tensor_parallel_size)}
        if set(metadata) != expected:
            raise ValueError(
                "Layerwise pull requires complete (PP, TP) worker handshake metadata; PCP is not supported"
            )
        endpoints = []
        pp_layers = []
        owned_layers: set[int] = set()
        for pp in range(pc.pipeline_parallel_size):
            stage = []
            layers = None
            for tp in range(pc.tensor_parallel_size):
                info = metadata[pp, tp]
                if not isinstance(info, LayerwisePullHandshakeMetadata):
                    raise TypeError("Unexpected layerwise pull worker handshake metadata")
                if not info.layer_ids or len(set(info.layer_ids)) != len(info.layer_ids):
                    raise ValueError("Layerwise pull workers must publish nonempty, unique layer IDs")
                if layers is None:
                    layers = tuple(sorted(info.layer_ids))
                elif tuple(sorted(info.layer_ids)) != layers:
                    raise ValueError("Layerwise pull requires identical layer ownership within each TP group")
                if self.is_consumer:
                    if not info.host:
                        raise ValueError("Layerwise pull D worker published an empty host")
                    _validate_tcp_port(info.port, description="Layerwise pull D worker endpoint")
                stage.append({"host": info.host, "port": info.port, "layer_ids": list(layers)})
            if owned_layers.intersection(layers):
                raise ValueError("Layerwise pull PP stages must own disjoint layers")
            owned_layers.update(layers)
            pp_layers.append(layers)
            endpoints.append(stage)
        if self.is_producer:
            scheduler.producer_pp_layers = tuple(pp_layers)
        else:
            scheduler.remote_endpoints = endpoints
            # Hash once at startup, not once per request/chunk on every P worker.
            scheduler.remote_topology_id = hashlib.sha256(msgspec.msgpack.encode(endpoints)).hexdigest()

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        if self.is_consumer:
            return self.connector_worker.get_finished(finished_req_ids)
        return self.connector_worker.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(self._get_connector_metadata())

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Layerwise Pull does not load KV at the attention-layer boundary."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        # Attention calls this every forward, including profiling / graph
        # capture where no per-step connector metadata is bound. Nothing to save
        # then; skip rather than trip _get_connector_metadata's assert.
        if not self.has_connector_metadata():
            return
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._get_connector_metadata())

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        # Producer-only early dispatch of the PD pull notification at scatter.
        if not self.is_producer or self.connector_worker is None:
            return
        if not self.has_connector_metadata():
            return
        hook = getattr(self.connector_worker, "on_kv_cache_written", None)
        if callable(hook):
            hook(layer_name, self._get_connector_metadata())

    def wait_for_save(self):
        # P-side completion is tracked by READ_DONE/storage_send_done_events.
        if self.is_consumer and self.connector_worker is not None:
            self.connector_worker.wait_for_save()

    def shutdown(self) -> None:
        for component in (self.connector_worker, self.connector_scheduler):
            shutdown = getattr(component, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def close(self) -> None:
        self.shutdown()

    def wait_for_slot_release(self, layer_idx: int) -> None:
        """Wait until Decode releases a physical slot before AscendStore reuses it."""
        worker = self.connector_worker
        if worker is None or not hasattr(worker, "wait_for_slot_release"):
            return
        worker.wait_for_slot_release(layer_idx)
