# SPDX-License-Identifier: Apache-2.0
"""Pull-side mooncake connector facade (moved out of the entry module)"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.base.connector import (
    MooncakeConnectorBase,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.metadata import MooncakeConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.scheduler import MooncakeConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.worker import MooncakeConnectorWorker

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu_model_runner import ForwardContext  # type: ignore[import-not-found]


class MooncakeConnector(MooncakeConnectorBase):
    """Pull-side facade; the six shared delegations live in the base class."""

    _shutdown_complete: bool = False

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return True

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        # vLLM main's KVConnectorBase_V1.__init__ tracks `_kv_cache_config`
        # (and `_role`), which `get_loaded_kv_cache_group_ids` reads.
        super().__init__(vllm_config, role, kv_cache_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
        # Scheduler side holds no owned resources; the worker side owns threads,
        # the executor and the remote socket pool.
        self._shutdown_complete = False

    def shutdown(self) -> None:
        """Idempotent connector shutdown; delegates to the worker side."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        if self.connector_worker is not None:
            self.connector_worker.shutdown()

    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids whose KV load failed."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        return self.connector_worker.get_transfer_results()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata]
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)
