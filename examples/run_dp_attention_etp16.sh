export VLLM_ENABLE_MC2=0
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_VERSION=0.9.0

nohup python -m vllm.entrypoints.openai.api_server --model=/mnt/deepseek/DeepSeek-R1-W8A8-VLLM \
    --quantization ascend \
    --trust-remote-code \
    --distributed-executor-backend=mp \
    --port 8006 \
    -tp=8 \
    -dp=2 \
    --max-num-seqs 24 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --additional-config '{"torchair_graph_config":{"enabled":true,"use_cached_graph":true,"graph_batch_sizes":[24]},"ascend_scheduler_config":{"enabled":true},"expert_tensor_parallel_size":16}' \
    --gpu-memory-utilization 0.96 &> run.log &
disown