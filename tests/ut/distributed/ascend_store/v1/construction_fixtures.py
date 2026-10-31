"""Exercise Connector initialization without binding an RPC socket."""

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector


def create_worker(vllm_config, cache_config):
    config = connector.resolve_kv_pool_config(vllm_config, cache_config)
    connector.validate_kv_pool_config(vllm_config, cache_config, config)
    return connector.AscendStoreV1Connector._create_worker(vllm_config, cache_config, config)


def create_scheduler(vllm_config, cache_config, address):
    config = connector.resolve_kv_pool_config(vllm_config, cache_config)
    connector.validate_kv_pool_config(vllm_config, cache_config, config)
    return connector.AscendStoreV1Connector._create_scheduler(config, address)
