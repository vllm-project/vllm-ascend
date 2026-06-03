no_enable_prefix_caching: true
tensor-parallel-size: 1
data-parallel-size: 1
block-size: 128
max-num-batched-tokens: 4096
port: 8100
max-model-len: 10000
kv-transfer-config: '{
    "kv_connector": "LMCacheAscendConnectorV1Dynamic",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
}'
