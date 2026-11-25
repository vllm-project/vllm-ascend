#!/usr/bin/env bash

# Arguments: devices, API port, DP size, DP rank, DP address, DP RPC port, TP, PCP, PP.
# The launcher supplies an optional tenth argument with this rank's explicit device IDs.
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

device_args=()
if [[ -n "${10:-}" ]]; then
    # Keep the inherited visibility: integer IDs index that list when it is set.
    device_args=(--device-ids "${10}")
else
    export ASCEND_RT_VISIBLE_DEVICES="$1"
fi

vllm serve model_path \
    "${device_args[@]}" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size $8 \
    --pipeline-parallel-size $9 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv3 \
    --max-model-len 8192 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens": 1, "method":"mtp"}'
