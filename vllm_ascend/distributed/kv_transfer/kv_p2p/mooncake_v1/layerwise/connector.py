# SPDX-License-Identifier: Apache-2.0
"""Layerwise mooncake connector facade (moved out of the entry module)"""

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base.connector import (
    MooncakeConnectorBase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.protocol import MooncakeLayerwiseConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.scheduler import (
    MooncakeLayerwiseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.worker import MooncakeLayerwiseConnectorWorker

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu_model_runner import ForwardContext  # type: ignore[import-not-found]


class MooncakeLayerwiseConnector(MooncakeConnectorBase):
    """Layerwise facade; the six shared delegations live in the base class."""

    requires_full_blocks_on_update_after_alloc = True
    supports_layerwise_buffer_reuse = True
    # Subclasses that override __init__ without calling super().__init__() keep
    # inherited shutdown() safe through this class default.
    _shutdown_complete: bool = False

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_producer = self._is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = MooncakeLayerwiseConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(vllm_config, kv_cache_config, str(self.engine_id))

    def shutdown(self) -> None:
        """Idempotent connector shutdown; delegates to scheduler and/or worker side."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids that have load errors."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        assert isinstance(connector_metadata, MooncakeLayerwiseConnectorMetadata)
        assert self.connector_worker is not None
        # Newer V1 runners may defer start_load_kv until after target forward.
        # Layerwise writes need the step's counters and send mappings beforehand.
        self.connector_worker.prepare_layerwise_step(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        assert self.connector_worker is not None
        self.connector_worker.wait_for_layer_reuse(layer_idx)

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        if not self._is_kv_producer or not self.has_connector_metadata():
            return
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        if not self._connector_metadata.requests:
            return
        assert self.connector_worker is not None
        plan = getattr(self.connector_worker, "_layerwise_step_plan", None)
        if plan is None or not plan.requests:
            return
        self.connector_worker.on_kv_cache_written(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: list[torch.Tensor], attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

    def wait_for_save(self):
        """MooncakeLayerwiseConnector does not save explicitly."""
        pass
