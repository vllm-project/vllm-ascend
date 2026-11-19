# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry for ordinary and SFA MooncakeConnectorV2 deployments."""

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeBaseConnector,
    MooncakeConnectorV2,
    MooncakePullConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class MooncakeSFACompatibleConnector(MooncakePullConnector):
    """Keep the V2 config name while isolating optional SFA policy."""

    def __new__(
        cls,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> MooncakeBaseConnector:
        if not cls._uses_sparse_offload(vllm_config):
            return MooncakeConnectorV2(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        if vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False):
            raise ValueError("Mooncake SFA KV offload requires pull mode (use_layerwise=false)")
        return super().__new__(cls)

    def _create_scheduler(
        self, vllm_config: VllmConfig, kv_cache_config: "KVCacheConfig"
    ) -> MooncakePullConnectorScheduler:
        if self._uses_sparse_offload(vllm_config):
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.scheduler import MooncakeSFAConnectorScheduler

            return MooncakeSFAConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)
        return super()._create_scheduler(vllm_config, kv_cache_config)

    def _create_worker(self, vllm_config: VllmConfig, kv_cache_config: "KVCacheConfig") -> MooncakePullConnectorWorker:
        if self._uses_sparse_offload(vllm_config):
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import MooncakeSFAConnectorWorker

            return MooncakeSFAConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
        return super()._create_worker(vllm_config, kv_cache_config)

    @staticmethod
    def _uses_sparse_offload(vllm_config: VllmConfig) -> bool:
        config = init_ascend_config(vllm_config).sparse_kv_offload_config
        return config is not None and config.enabled

    def get_copy_sfa_slot_bindings(self) -> dict[str, int]:
        state = getattr(self.connector_worker, "_sfa_pd_state", None)
        return dict(state.bindings) if state is not None else {}

    def prepare_copy_sfa_decode(self, input_batch, slots, block_tables, dense_fills, *, restore_tails: bool):
        prepare = getattr(self.connector_worker, "prepare_copy_sfa_decode", None)
        if prepare is None:
            return restore_tails, dense_fills, None
        return prepare(input_batch, slots, block_tables, dense_fills, restore_tails=restore_tails)

    def prepare_sparse_kv_offload(self, metadata) -> None:
        prepare = getattr(self.connector_worker, "prepare_sparse_kv_offload", None)
        if prepare is not None:
            prepare(metadata)
