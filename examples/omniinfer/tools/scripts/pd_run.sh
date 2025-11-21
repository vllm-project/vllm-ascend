#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Default parameters
# llmdatadist-specific parameters
GLOBAL_RANK_TABLE_FILE_PATH="1p1d_save_dir/global_ranktable_merge.json"
RANK_TABLE_FILE_PATH="save_dir_64/local_ranktable_7.242.108.64_0123.json"
LOCAL_DECODE_SERVER_IP_LIST="7.242.108.196"
GLOBAL_DECODE_SERVER_IP_LIST="7.242.108.196"
ROLE="prefill"
PREFILL_POD_NUM=1
DECODE_POD_NUM=1
VLLM_LLMDATADIST_ZMQ_PORT="5568"
# Ascend-specific parameters
HCCL_INTRA_ROCE_ENABLE=1
HCCL_INTRA_PCIE_ENABLE=0
ascend_rt_set=0
# mockModel configuration parameters
#RANDOM_MODE=0
#KV_CACHE_MOD=0
#FORWARD_TIME=0
# Multi-API Server specific parameters
NUM_SERVERS=1
NUM_DP=1
SERVER_OFFSET=0
MASTER_IP="7.242.108.64"
MASTER_PORT=8503
BASE_API_PORT=9001
# vLLM framework parameters
GLOO_SOCKET_IFNAME="enp23s0f3"
TP_SOCKET_IFNAME="enp23s0f3"
VLLM_LOGGING_LEVEL="INFO"
VLLM_USE_V1=1
VLLM_WORKER_MULTIPROC_METHOD="fork"
MODEL_PATH="/home/dsv3/models/DeepSeek-V3-w8a8-0423"
TP=4
SERVED_MODEL_NAME="deepseek"
MAX_MODEL_LEN=4096
LOG_DIR="apiserverlog"
# PD separation parameters
KV_CONNECTOR="AscendHcclConnectorV1"
KV_BUFFER_DEVICE="npu"
KV_ROLE="kv_producer"
KV_RANK=0
KV_ENGINE_ID=0
KV_PARALLEL_SIZE=2

MODEL_EXTRA_CFG_PATH="/workspace/omni_infer/test/test_config/test_config_prefill.json"
GPU_UTIL=0.9
EXTRA_ARGS=""
ADDITIONAL_CONFIG=""
VLLM_ENABLE_MC2=0
HCCL_BUFFSIZE=0
HCCL_OP_EXPANSION_MODE=""

# Help information
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                        Display this help message"
    echo "  --global-rank-table-path         llmdatadist-specific: Global rank table file path. For merged P/D instances, usually global_ranktable_merge.json (default: $GLOBAL_RANK_TABLE_FILE_PATH)"
    echo "  --rank-table-path                llmdatadist-specific: Local rank table file path for P or D instances. Usually local_ranktable_{IP}_rank.json; for cross-machine D instances, use local_ranktable_merge*.json (default: $RANK_TABLE_FILE_PATH)"
    echo "  --local-decode-server-ip-list    llmdatadist-specific: IP list of current D instance. Separate multiple IPs with commas, maintaining same order as ranktable (default: $LOCAL_DECODE_SERVER_IP_LIST)"
    echo "  --global-decode-server-ip-list   llmdatadist-specific: IP list of all D instances. Combination of all d instances' LOCAL_DECODE_SERVER_IP_LIST, separated by ';'. For 1d scenarios, same as LOCAL_DECODE_SERVER_IP_LIST (default: $GLOBAL_DECODE_SERVER_IP_LIST)"
    echo "  --role                           llmdatadist-specific: Instance role type. Use 'prefill' for P, 'decode' for D (default: $ROLE)"
    echo "  --prefill-pod-num                llmdatadist-specific: Number of P instances (default: $PREFILL_POD_NUM)"
    echo "  --decode-pod-num                 llmdatadist-specific: Number of D instances (default: $DECODE_POD_NUM)"
    echo "  --vllm-llmdatadist-zmq-port      llmdatadist-specific: ZMQ port for llmdatadist connector (must be string) (default: $VLLM_LLMDATADIST_ZMQ_PORT)"
    echo "  --hcc-intra-roce-enable          Ascend-specific: Set to 1 for A3, enable intra-HCCL ROCE (default: $HCCL_INTRA_ROCE_ENABLE)"
    echo "  --hcc-intra-pcie-enable          Ascend-specific: Set to 0 for A3, enable intra-HCCL PCIE (default: $HCCL_INTRA_PCIE_ENABLE)"
    echo "  --ascend-rt-visible-devices      Ascend-specific: Visible physical devices for the instance. (default: $ASCEND_RT_VISIBLE_DEVICES)"
    echo "  --random-mode                    mockModel configuration: Enable mock model when set to 1 (default: $RANDOM_MODE)"
    echo "  --kv-cache-mod                   mockModel configuration: Enable mock model when set to 1 (default: $KV_CACHE_MOD)"
    echo "  --forward-time                   mockModel configuration: Forward time in ms (default: $FORWARD_TIME)"
    echo "  --num-servers                    Multi-API Server: Number of API servers (default: $NUM_SERVERS)"
    echo "  --num-dp                         Multi-API Server: Data parallel size (≥ number of servers) (default: $NUM_DP)"
    echo "  --server-offset                  Multi-API Server: Server offset for multi-node setup. For dual-node A3, set to 16 on d_2 instance (default: $SERVER_OFFSET)"
    echo "  --master-ip                      Multi-API Server: Master node IP for multi-node setup. For dual-node A3, set to head node IP (corresponds to vllm data-parallel-address) (default: $MASTER_IP)"
    echo "  --master-port                    Multi-API Server: Master node Gloo socket communication port (corresponds to vllm data-parallel-rpc-port) (default: $MASTER_PORT)"
    echo "  --base-api-port                  Multi-API Server: Base API port for multi API servers (default: $BASE_API_PORT)"
    echo "  --gloo-socket-ifname             vLLM framework: DP communication parameter. Your network interface. Query with: ip -4 route list 0/0 | awk '{print $5}' | head -n 1 (default: $GLOO_SOCKET_IFNAME)"
    echo "  --tp-socket-ifname               vLLM framework: DP communication parameter. Your network interface. Query with: ip -4 route list 0/0 | awk '{print $5}' | head -n 1 (default: $TP_SOCKET_IFNAME)"
    echo "  --vllm-logging-level             vLLM framework: VLLM logging level. Default INFO, set to DEBUG for debugging (default: $VLLM_LOGGING_LEVEL)"
    echo "  --vllm-use-v1                    vLLM framework: Use VLLM V1 version (1 to enable) (default: $VLLM_USE_V1)"
    echo "  --vllm-worker-multiproc-method   vLLM framework: VLLM worker process method (fork or spawn) (default: $VLLM_WORKER_MULTIPROC_METHOD)"
    echo "  --model-path                     vLLM framework: Model path (default: $MODEL_PATH)"
    echo "  --max-model-len                  vLLM framework: Maximum model length (default: $MAX_MODEL_LEN)"
    echo "  --tp                             vLLM framework: Tensor parallel (default: $TP)"
    echo "  --served-model-name              vLLM framework: Served model name (default: $SERVED_MODEL_NAME)"
    echo "  --log-dir                        vLLM framework: Log directory (default: $LOG_DIR)"
    echo "  --kv-connector                   vLLM framework: PD separation parameter, kv connector name (default: $KV_CONNECTOR)"
    echo "  --kv-buffer-device               vLLM framework: PD separation parameter, kv transfer buffer device (default: $KV_BUFFER_DEVICE)"
    echo "  --kv-role                        vLLM framework: PD separation parameter, kv role (p: kv_producer, d: kv_consumer) (default: $KV_ROLE)"
    echo "  --kv-rank                        vLLM framework: PD separation parameter, kv rank (p_num/d_num-1) (default: $KV_RANK)"
    echo "  --kv-engine-id                   vLLM framework: PD separation parameter, kv engine ID (default: $KV_ENGINE_ID)"
    echo "  --kv-parallel-size               vLLM framework: PD separation parameter, kv parallel size (equal to num_p + num_d) (default: $KV_PARALLEL_SIZE)"
    echo "  --extra-args                     vLLM framework: Additional VLLM arguments (space-separated, e.g., '--enable-expert-parallel') (default: $EXTRA_ARGS)"
    echo "  --additional-args                vLLM framework: Additional VLLM arguments"
    echo "  --vllm-enable-mc2                vLLM framework: GRAPH parameter (default: $VLLM_ENABLE_MC2)"
    echo "  --hccl-op-expansion-mode         vLLM framework: HCCL_OP_EXPANSION_MODE"
    echo "  --hccl-buffsize                  vLLM framework: HCCL_BUFFSIZE"
    exit 0
}

# Parse long options
parse_long_option() {
    case "$1" in
        --global-rank-table-path)
            GLOBAL_RANK_TABLE_FILE_PATH="$2"
            ;;
        --rank-table-path)
            RANK_TABLE_FILE_PATH="$2"
            ;;
        --local-decode-server-ip-list)
            LOCAL_DECODE_SERVER_IP_LIST="$2"
            ;;
        --global-decode-server-ip-list)
            GLOBAL_DECODE_SERVER_IP_LIST="$2"
            ;;
        --role)
            ROLE="$2"
            ;;
        --prefill-pod-num)
            PREFILL_POD_NUM="$2"
            ;;
        --decode-pod-num)
            DECODE_POD_NUM="$2"
            ;;
        --vllm-llmdatadist-zmq-port)
            VLLM_LLMDATADIST_ZMQ_PORT="$2"
            ;;
        --hcc-intra-roce-enable)
            HCCL_INTRA_ROCE_ENABLE="$2"
            ;;
        --hcc-intra-pcie-enable)
            HCCL_INTRA_PCIE_ENABLE="$2"
            ;;
        --ascend-rt-visible-devices)
            ASCEND_RT_VISIBLE_DEVICES="$2"
            ascend_rt_set=1
            ;;
        --random-mode)
            RANDOM_MODE="$2"
            ;;
        --kv-cache-mod)
            KV_CACHE_MOD="$2"
            ;;
        --forward-time)
            FORWARD_TIME="$2"
            ;;
        --num-servers)
            NUM_SERVERS="$2"
            ;;
        --num-dp)
            NUM_DP="$2"
            ;;
        --server-offset)
            SERVER_OFFSET="$2"
            ;;
        --master-ip)
            MASTER_IP="$2"
            ;;
        --master-port)
            MASTER_PORT="$2"
            ;;
        --base-api-port)
            BASE_API_PORT="$2"
            ;;
        --gloo-socket-ifname)
            GLOO_SOCKET_IFNAME="$2"
            ;;
        --tp-socket-ifname)
            TP_SOCKET_IFNAME="$2"
            ;;
        --vllm-logging-level)
            VLLM_LOGGING_LEVEL="$2"
            ;;
        --vllm-use-v1)
            VLLM_USE_V1="$2"
            ;;
        --vllm-worker-multiproc-method)
            VLLM_WORKER_MULTIPROC_METHOD="$2"
            ;;
        --model-path)
            MODEL_PATH="$2"
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            ;;
        --tp)
            TP="$2"
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            ;;
        --log-dir)
            LOG_DIR="$2"
            ;;
        --kv-connector)
            KV_CONNECTOR="$2"
            ;;
        --kv-buffer-device)
            KV_BUFFER_DEVICE="$2"
            ;;
        --kv-role)
            KV_ROLE="$2"
            ;;
        --kv-rank)
            KV_RANK="$2"
            ;;
        --kv-engine-id)
            KV_ENGINE_ID="$2"
            ;;
        --kv-parallel-size)
            KV_PARALLEL_SIZE="$2"
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            ;;
        --model-extra-cfg-path)
            MODEL_EXTRA_CFG_PATH="$2"
            ;;
        --gpu-util)
            GPU_UTIL="$2"
            ;;
        --vllm-enable-mc2)
            VLLM_ENABLE_MC2="$2"
            ;;
        --additional-config)
            ADDITIONAL_CONFIG="$2"
            ;;
        --hccl-buffsize)
            HCCL_BUFFSIZE="$2"
            ;;
        --hccl-op-expansion-mode)
            HCCL_OP_EXPANSION_MODE="$2"
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            ;;
    esac
    return 0
}

# Parse options
# Modified main loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --*)
            parse_long_option "$1" "$2"  # Parse without shift
            shift 2  # Shift in main loop
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            ;;
    esac
done

# Build KV transfer config JSON
KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "$KV_CONNECTOR",
    "kv_buffer_device": "$KV_BUFFER_DEVICE",
    "kv_role": "$KV_ROLE",
    "kv_rank": $KV_RANK,
    "engine_id": $KV_ENGINE_ID,
    "kv_parallel_size": $KV_PARALLEL_SIZE
}
EOF
)

# Export environment variables
export GLOBAL_RANK_TABLE_FILE_PATH
export RANK_TABLE_FILE_PATH
export LOCAL_DECODE_SERVER_IP_LIST
export GLOBAL_DECODE_SERVER_IP_LIST
export ROLE
export PREFILL_POD_NUM
export DECODE_POD_NUM
export VLLM_LLMDATADIST_ZMQ_PORT

#export RANDOM_MODE
#export KV_CACHE_MOD
#export FORWARD_TIME

export HCCL_INTRA_ROCE_ENABLE
export HCCL_INTRA_PCIE_ENABLE
if [ $ascend_rt_set -eq 1 ]; then
    export ASCEND_RT_VISIBLE_DEVICES
    echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
fi
export GLOO_SOCKET_IFNAME
export TP_SOCKET_IFNAME
export VLLM_LOGGING_LEVEL
export VLLM_USE_V1
export VLLM_WORKER_MULTIPROC_METHOD
export SERVER_OFFSET
export MODEL_EXTRA_CFG_PATH
export PYTHONPATH=/usr/local/Ascend/CANN-7.7/toolkit/python/site-packages:$PYTHONPATH
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export USING_LCCL_COM=0
export OMNI_USE_DSV3=1
export VLLM_ENABLE_MC2
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:40-41,npu2:80-81,npu3:120-121,npu4:160-161,npu5:200-201,npu6:240-241,npu7:280-281

if [ -n "$HCCL_OP_EXPANSION_MODE" ]; then
    export HCCL_OP_EXPANSION_MODE
    echo "HCCL_OP_EXPANSION_MODE: $HCCL_OP_EXPANSION_MODE"
fi
if [ $HCCL_BUFFSIZE -gt 0 ] ; then
    export HCCL_BUFFSIZE
    echo "HCCL_BUFFSIZE: $HCCL_BUFFSIZE"
fi

export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=120
# 随路拷贝
export TNG_HOST_COPY=1
# 使能双页表 pd 分离
export AUTO_USE_UC_MEMORY=1
export TASK_QUEUE_ENABLE=2

export CPU_AFFINITY_CONF=2

export OMNI_REUSE_PREFILLED_TOKENS=1
export OMNI_SKIP_DECODE_TOKENIZE=1
export TOKENIZER_PROC_POOL=0

# enable to overwrite request IDs
export ENABLE_OVERWRITE_REQ_IDS=0

# enable apc event
export ENABLE_APC_EVENT=0

# Print current configuration
echo "==== Current Configuration ===="
echo "GLOBAL_RANK_TABLE_FILE_PATH: $GLOBAL_RANK_TABLE_FILE_PATH"
echo "RANK_TABLE_FILE_PATH: $RANK_TABLE_FILE_PATH"
echo "LOCAL_DECODE_SERVER_IP_LIST: $LOCAL_DECODE_SERVER_IP_LIST"
echo "GLOBAL_DECODE_SERVER_IP_LIST: $GLOBAL_DECODE_SERVER_IP_LIST"
echo "ROLE: $ROLE"
echo "PREFILL_POD_NUM: $PREFILL_POD_NUM"
echo "DECODE_POD_NUM: $DECODE_POD_NUM"
echo "VLLM_LLMDATADIST_ZMQ_PORT: $VLLM_LLMDATADIST_ZMQ_PORT"
echo "HCCL_INTRA_ROCE_ENABLE: $HCCL_INTRA_ROCE_ENABLE"
echo "HCCL_INTRA_PCIE_ENABLE: $HCCL_INTRA_PCIE_ENABLE"
echo "RANDOM_MODE: $RANDOM_MODE"
echo "KV_CACHE_MOD: $KV_CACHE_MOD"
echo "FORWARD_TIME: $FORWARD_TIME"
echo "NUM_SERVERS: $NUM_SERVERS"
echo "NUM_DP: $NUM_DP"
echo "SERVER_OFFSET: $SERVER_OFFSET"
echo "MASTER_IP: $MASTER_IP"
echo "MASTER_PORT: $MASTER_PORT"
echo "BASE_API_PORT: $BASE_API_PORT"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "TP_SOCKET_IFNAME: $TP_SOCKET_IFNAME"
echo "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "MODEL_PATH: $MODEL_PATH"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "TP: $TP"
echo "SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
echo "LOG_DIR: $LOG_DIR"
echo "KV_TRANSFER_CONFIG: $KV_TRANSFER_CONFIG"
echo "EXTRA_ARGS: $EXTRA_ARGS"
echo "MODEL_EXTRA_CFG_PATH: $MODEL_EXTRA_CFG_PATH"
echo "GPU_UTIL: $GPU_UTIL"
echo "ADDITIONAL_CONFIG: $ADDITIONAL_CONFIG"
echo "VLLM_ENABLE_MC2: $VLLM_ENABLE_MC2"
echo "TNG_HOST_COPY: $TNG_HOST_COPY"
echo "CPU_AFFINITY_CONF: $CPU_AFFINITY_CONF"
echo "AUTO_USE_UC_MEMORY: $AUTO_USE_UC_MEMORY"
echo "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES: $RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"
echo "RAY_CGRAPH_get_timeout: $RAY_CGRAPH_get_timeout"
echo "TASK_QUEUE_ENABLE: $TASK_QUEUE_ENABLE"
echo "=================="

EXTRA_ARGS="$EXTRA_ARGS --middleware omni.adaptors.vllm.entrypoints.middleware.param_check.ValidateSamplingParams"

# Execute Python script

common_operations() {
  python start_api_servers.py \
    --num-servers "$NUM_SERVERS" \
    --num-dp "$NUM_DP" \
    --server-offset "$SERVER_OFFSET" \
    --model-path "$MODEL_PATH" \
    --master-ip "$MASTER_IP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --master-port "$MASTER_PORT" \
    --base-api-port "$BASE_API_PORT" \
    --tp "$TP" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --log-dir "$LOG_DIR" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --gpu-util "$GPU_UTIL" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --enable-mtp \
    --extra-args "$EXTRA_ARGS"
}

if [ $(echo -n "$NODE_IP_LIST" | tr -cd ',' | wc -c) -ge 1 ]; then
  if [ "$IP" = "$HOST_IP" ]; then
    export RAY_USAGE_STATS_ENABLED=0
    ray start --head --num-gpus=$NUM_SERVERS
    sleep 10s
    common_operations
  else
    sleep 5s
    command="ray start --address='$HOST_IP:6379' --num-gpus=$NUM_SERVERS &> /dev/null"
    echo $command
    cost_time=0
    end_time=300
    while true; do
      if [ $cost_time -ge $end_time ]; then
        echo "error, connection timeout"
        exit 1
      fi

      eval $command
      if [ $? -eq 0 ]; then
        echo "succeed to connect to ray head node"
        break
      else
        echo "failed to connect to ray head node, wait 5s....."
        sleep 5
        cost_time=$((cost + 5))
      fi
    done
  fi
else
  common_operations
fi
