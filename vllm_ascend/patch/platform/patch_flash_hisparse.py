# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bridge Flash HiSparse cache planning and terminal spill on the verified Core pin."""

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
from vllm.v1.core import kv_cache_planning

from vllm_ascend.hisparse import supports_glm5_next_hisparse, supports_glm5_next_int8_hisparse
from vllm_ascend.hisparse.flash_terminal_spill import install_flash_terminal_spill_compat

_ORIGINAL_POOL_BYTES_PER_BLOCK = kv_cache_planning._pool_bytes_per_block_from_builder
_ORIGINAL_BIND_KV_CACHE_MANAGER = HiSparseConnector.bind_kv_cache_manager


def _pool_bytes_per_block_from_builder(builder, vllm_config, kv_cache_groups):
    if kv_cache_groups and (supports_glm5_next_hisparse(vllm_config) or supports_glm5_next_int8_hisparse(vllm_config)):
        pool_bytes = builder._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
        if pool_bytes is not None:
            return pool_bytes
    return _ORIGINAL_POOL_BYTES_PER_BLOCK(builder, vllm_config, kv_cache_groups)


def _bind_kv_cache_manager(connector, kv_cache_manager):
    _ORIGINAL_BIND_KV_CACHE_MANAGER(connector, kv_cache_manager)
    config = connector._vllm_config
    if supports_glm5_next_hisparse(config) or supports_glm5_next_int8_hisparse(config):
        scheduler = connector.connector_scheduler
        assert scheduler is not None and scheduler.coordinator is not None
        install_flash_terminal_spill_compat(scheduler.coordinator)


kv_cache_planning._pool_bytes_per_block_from_builder = _pool_bytes_per_block_from_builder
HiSparseConnector.bind_kv_cache_manager = _bind_kv_cache_manager
