# SPDX-License-Identifier: Apache-2.0
"""Explicit Mooncake Pull connector with SFA-owned Host KV."""

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.scheduler import MooncakeSFAConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.worker import MooncakeSFAConnectorWorker


class MooncakeSFAConnector(MooncakePullConnector):
    def _create_scheduler(self, vllm_config, kv_cache_config):
        return MooncakeSFAConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)

    def _create_worker(self, vllm_config, kv_cache_config):
        return MooncakeSFAConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    def get_copy_sfa_slot_bindings(self) -> dict[str, int]:
        """Use the same prebound-row contract as the SFA runner path."""
        state = self.get_copy_sfa_pd_state()
        return dict(state.bindings) if state is not None else {}

    def get_copy_sfa_pd_state(self):
        return getattr(self.connector_worker, "_sfa_pd_state", None)

    def prepare_sparse_kv_offload(self, metadata: MooncakeSFAConnectorMetadata) -> None:
        if self.connector_worker is not None:
            assert isinstance(self.connector_worker, MooncakeSFAConnectorWorker)
            self.connector_worker.prepare_sparse_kv_offload(metadata)
