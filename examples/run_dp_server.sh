export HCCL_IF_IP=141.61.41.74
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=0

export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_DP_SIZE=2
export VLLM_DP_RANK=0
export VLLM_DP_MASTER_IP="141.61.41.74"
export VLLM_DP_MASTER_PORT=40001
export VLLM_DP_PROXY_IP="141.61.41.74"
export VLLM_DP_PROXY_PORT=30003
export VLLM_HTTP_PORT=20001

vllm serve /data/weights/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name Qwen \
    --max-model-len 2000 \
    --max-num-batched-tokens 2000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \