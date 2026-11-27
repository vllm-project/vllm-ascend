# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Select the Ascend worker while reusing the Core HiSparse scheduler."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector as CoreHiSparseConnector

from vllm_ascend.hisparse.worker import AscendHiSparseConnectorWorker


class HiSparseConnector(CoreHiSparseConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        if role != KVConnectorRole.WORKER:
            super().__init__(vllm_config, role, kv_cache_config)
            return
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        if kv_cache_config.hisparse_host_num_blocks is None:
            raise ValueError("HiSparseConnector requires a HiSparse host pool")
        self.connector_scheduler = None
        self.connector_worker = AscendHiSparseConnectorWorker(vllm_config, kv_cache_config)
