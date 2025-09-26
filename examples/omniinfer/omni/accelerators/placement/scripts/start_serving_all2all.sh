export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMNI_PLANNER_CONFIG="/home/omni/omni_planner/config.yaml"

export ENABLE_MOE_EP=1
export DP_SIZE=4
unset VLLM_ENABLE_PROFILING
unset VLLM_TORCH_PROFILER_DIR
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -c unlimited
python -m vllm.entrypoints.openai.api_server   --model /opt/models/models/dsv3/DeepSeek-V3-w8a8-0208-50/   --tensor-parallel-size 16  --gpu-memory-utilization 0.9   --dtype bfloat16    --block-size 128 --trust-remote-code  --served-model-name deepseek --distributed-executor-backend=ray --max-model-len=1024 --host="127.0.0.1" --port=8999

# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export GLOO_SOCKET_IFNAME=enp67s0f5
# export TP_SOCKET_IFNAME=enp67s0f5
# # export GLOO_SOCKET_IFNAME=eth4
# # export TP_SOCKET_IFNAME=eth4
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# export RAY_DEDUP_LOGS=0
# export HCCL_OP_EXPANSION_MODE="AIV"

# export ENABLE_MOE_EP=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# VLLM_TORCH_PROFILER_DIR=/home/yyx/profiling python -m vllm.entrypoints.openai.api_server --model /home/yyx/ram_data/DeepSeek-V3-w8a8-0208-50 --tensor-parallel-size 16  --gpu-memory-utilization 1.0  --dtype bfloat16 --block-size 128 --trust-remote-code --served-model-name deepseek --distributed-executor-backend=ray --max-model-len=1024  --host="127.0.0.1"  --port=8999

#--max_num_seqs 4096  --max_num_batched_tokens 204800
