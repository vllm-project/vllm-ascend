#!/usr/bin/env bash
#
# MiniMax-M2.5 w8a8 QuaRot - PD separation PREFILL node template.
#
# Called by launch_online_dp.py (../launch_online_dp.py), which passes:
#   $1 visible_devices, $2 vllm port, $3 dp_size, $4 dp_rank,
#   $5 dp_address, $6 dp_rpc_port, $7 tp_size
#
# Validated 1P1D configuration (2x Atlas 800 A3, 64GB x 16 each):
#   prefill: DP2/TP8/EP16, max-model-len 200000, Mooncake kv_producer.
# See docs/source/tutorials/models/MiniMax-M2.md section 5.2.

set -euo pipefail

unset http_proxy https_proxy ftp_proxy

# ---- edit these for your environment ----
MODEL_PATH="${MODEL_PATH:-/path/to/weight/MiniMax-M2.5-w8a8-QuaRot}"
EAGLE_MODEL_PATH="${EAGLE_MODEL_PATH:-/path/to/weight/MiniMax-M2.5-eagle-model-0318}"
NIC_NAME="${NIC_NAME:-<your_nic_name>}"
LOCAL_IP="${LOCAL_IP:-<your_ip>}"

export HCCL_IF_IP=$LOCAL_IP
export GLOO_SOCKET_IFNAME=$NIC_NAME
export TP_SOCKET_IFNAME=$NIC_NAME
export HCCL_SOCKET_IFNAME=$NIC_NAME

export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export PYTHONHASHSEED=0

export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --served-model-name minimax \
    --max-model-len 200000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --quantization ascend \
    --enforce-eager \
    --speculative_config "{\"method\": \"eagle3\", \"model\": \"$EAGLE_MODEL_PATH\", \"num_speculative_tokens\": 1}" \
    --additional-config '{"enable_cpu_binding":true}' \
    --kv-transfer-config \
        '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "35880",
        "engine_id": "0",
        "kv_connector_extra_config": {
             "use_ascend_direct": true,
             "prefill": {"dp_size": 2, "tp_size": 8},
             "decode":  {"dp_size": 2, "tp_size": 8}
        }}'
