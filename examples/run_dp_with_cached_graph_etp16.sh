export HCCL_IF_IP=2.0.0.0
export GLOO_SOCKET_IFNAME="enp189s0f0"
export TP_SOCKET_IFNAME="enp189s0f0"
export HCCL_SOCKET_IFNAME="enp189s0f0"

export VLLM_USE_V1=1
export ASCEND_LAUNCH_BLOCKING=0
# export VLLM_VERSION=0.9.0

nohup python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
    --host 0.0.0.0 \
    --port 20002 \
    --quantization ascend \
    -dp=2 \
    -tp=8 \
    --no-enable-prefix-caching \
    --max-num-seqs 24 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.96 \
    --trust-remote-code \
    --distributed-executor-backend=mp \
    --additional-config '{"torchair_graph_config":{"enabled":true,"use_cached_graph":true,"graph_batch_sizes":[24]},"ascend_scheduler_config":{"enabled":true}}' \
    & > run.log &
disown
