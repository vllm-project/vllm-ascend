rm -rf ./.torchair_cache/
rm -rf ./dynamo_*
rm -rf /root/ascend/log/debug/plog/*

export HCCL_IF_IP=2.0.0.0
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=1
export ASCEND_LAUNCH_BLOCKING=0

vllm serve /data/weights/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 20002 \
    --served-model-name Qwen \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-address 2.0.0.0 \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --max-num-seqs 16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":{"enabled":false, "enable_multistream_moe":false, "use_cached_graph":false}}'
