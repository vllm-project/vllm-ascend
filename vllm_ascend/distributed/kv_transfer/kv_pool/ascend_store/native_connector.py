# SPDX-License-Identifier: Apache-2.0
"""NPU bindings for the upstream Mooncake Store connector."""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector

from .native_worker import AscendNativeStoreWorker


class AscendNativeStoreConnector(MooncakeStoreConnector):
    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config):
        return False

    def _create_worker(self, vllm_config, kv_cache_config):
        return AscendNativeStoreWorker(vllm_config, kv_cache_config)
