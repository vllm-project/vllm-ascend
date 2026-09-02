#!/bin/bash
# DeepSeek-V4-Flash layerwise + PP 特性验证启动脚本 (npu2 / lsy_pd)
# PP=2 x TP=4 (8 卡), kv_both 单实例, memcache host_shm 后端
# 分支: feat/dsv4-layerwise-pp (global layer addressing + multi-cache-spec)

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

export PYTHONPATH=/home/vllm:/home/vllm-ascend:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local.conf

export HCCL_IF_IP=172.28.1.220
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=200
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

KV_CONFIG='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"layerwise_prefetch_layers":3,"mooncake_rpc_port":"0"}}'

LOG_FILE=/home/lsy/npu2_lwpp_full.log
echo "=== layerwise PP2 x TP4 ==="
echo "=== KV_CONFIG: $KV_CONFIG ==="

/usr/local/python3.11.15/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --port 8006 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --served-model-name deepseek_v4 \
    --enable-expert-parallel \
    --max-num-seqs 4 --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --kv-transfer-config "$KV_CONFIG" > ${LOG_FILE} 2>&1

echo "=== vllm serve exited, log: ${LOG_FILE} ==="
