# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Mooncake connector facades.

The connector classes in this module bridge vLLM's KV connector interface to
the scheduler-side and worker-side Mooncake implementations. Transfer and
scheduling details belong in those implementations rather than in this facade.

``MooncakeConnector`` remains an alias for the pull-based connector. The push
connector is declared as part of the new architecture but is not implemented.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    MooncakePullConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
    MooncakePromMetrics,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class MooncakeBaseConnector(KVConnectorBase_V1, SupportsHMA):
    """Common facade for the Mooncake scheduler and worker implementations."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None

        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_cache_config = kv_cache_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.connector_scheduler: MooncakeBaseConnectorScheduler | None = None
        self.connector_worker: MooncakeBaseConnectorWorker | None = None

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """Use HND to avoid reformatting KV cache before D2D transfer."""
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect the current vLLM model config; falling back to the default KV cache layout."
            )
            return None

        if vllm_config.model_config.use_mla:
            # MLA has no separate head dimension, so HND versus NHD does not
            # affect its transfer layout.
            return None

        logger.info_once("MooncakeConnector is setting the KV cache layout to HND for direct D2D transfer.")
        return "HND"

    ############################################################
    # Scheduler-side methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

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

    def set_xfer_handshake_metadata(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Set worker handshake metadata on the scheduler."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    def set_xfer_handshake_metadata_pp_aware(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Set PP-aware worker handshake metadata on the scheduler."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata_from_workers(metadata)

    ############################################################
    # Worker-side methods
    ############################################################

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Return requests whose receive and send operations have finished."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs whose Mooncake KV load failed."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(cls, data: dict[str, Any] | None = None) -> KVConnectorStats:
        return MooncakeKVConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return MooncakePromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not load KV cache layer by layer."""

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """MooncakeConnector does not save KV cache explicitly."""

    def wait_for_save(self) -> None:
        """MooncakeConnector does not save KV cache explicitly."""

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Return this worker's out-of-band transfer handshake metadata."""
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata


class MooncakePullConnector(MooncakeBaseConnector):
    """Pull-based Mooncake KV transfer connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakePullConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MooncakePullConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")


class MooncakePushConnector(MooncakeBaseConnector):
    """Placeholder for the not-yet-implemented push-based connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        raise NotImplementedError("MooncakePushConnector is not implemented yet.")


# Backward compatibility: MooncakeConnector is the pull-based connector.
MooncakeConnector = MooncakePullConnector


__all__ = [
    "MooncakeBaseConnector",
    "MooncakeConnector",
    "MooncakePullConnector",
    "MooncakePushConnector",
]
