export HCCL_IF_IP=your_ip_here
export GLOO_SOCKET_IFNAME=your_socket_ifname_here
export TP_SOCKET_IFNAME=your_socket_ifname_here
export HCCL_SOCKET_IFNAME=your_socket_ifname_here
export VLLM_LOGGING_LEVEL="info"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DETERMINISTIC=True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve model_path \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv3 \
    --max-model-len 8192 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \