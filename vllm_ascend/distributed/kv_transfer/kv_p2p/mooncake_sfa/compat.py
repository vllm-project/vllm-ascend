# SPDX-License-Identifier: Apache-2.0
"""Select the legacy V2 configuration before constructing any components."""

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakePullConnector as BaseMooncakePullConnector,
)


class MooncakePullConnector(BaseMooncakePullConnector):
    def _uses_sparse_offload(self, vllm_config):
        config = init_ascend_config(vllm_config).sparse_kv_offload_config
        return config is not None and config.enabled

    def _create_scheduler(self, vllm_config, kv_cache_config):
        if self._uses_sparse_offload(vllm_config):
            # Keep the ordinary path independent of offload/manager imports.
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import connector

            return connector.MooncakeSFAConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)
        return super()._create_scheduler(vllm_config, kv_cache_config)

    def _create_worker(self, vllm_config, kv_cache_config):
        if self._uses_sparse_offload(vllm_config):
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import connector

            return connector.MooncakeSFAConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)
        return super()._create_worker(vllm_config, kv_cache_config)

    def get_copy_sfa_pd_state(self):
        return getattr(self.connector_worker, "_sfa_pd_state", None)

    def get_copy_sfa_slot_bindings(self) -> dict[str, int]:
        state = self.get_copy_sfa_pd_state()
        return dict(state.bindings) if state is not None else {}

    def prepare_sparse_kv_offload(self, metadata) -> None:
        prepare = getattr(self.connector_worker, "prepare_sparse_kv_offload", None)
        if prepare is not None:
            prepare(metadata)


# Preserve the historical class-name key used when reconstructing metrics.
MooncakeConnector = MooncakePullConnector
