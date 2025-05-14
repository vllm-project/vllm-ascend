export HCCL_IF_IP=2.0.0.0
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

export VLLM_USE_V1=0

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --host 0.0.0.0 \
    --port 20002 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name deepseek \
    --max-model-len 2000 \
    --max-num-batched-tokens 2000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector": "AscendSimpleConnector",
    "kv_buffer_device": "npu",
    "kv_role": "kv_consumer",
    "kv_parallel_size": 8,
    "kv_port":"21001",
    "kv_connector_extra_config":
    {"prompt_device_ips": ["1.2.3.1", "1.2.3.2", "1.2.3.3", "1.2.3.4", "1.2.3.5", "1.2.3.6", "1.2.3.7", "1.2.3.8"],
    "decode_device_ips": ["1.2.3.9", "1.2.3.10", "1.2.3.11", "1.2.3.12", "1.2.3.13", "1.2.3.14", "1.2.3.15", "1.2.3.16"],
    "llmdatadist_comm_port": 26000,
    "proxy_ip":"3.0.0.0",
    "proxy_port":"30001",
    "http_port": 10002}
    }'
