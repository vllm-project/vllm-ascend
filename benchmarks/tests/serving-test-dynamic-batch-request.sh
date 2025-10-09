#!/bin/bash
set -ex


wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  local endpoin=$2
  timeout 1200 bash -c "
    until curl -s localhost:${port}${endpoin} > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip
nic_name=$4
local_ip=$5

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024


MOEL_DIR=$1 
DATASET_NAME=sharegpt  # random sharegpt hf
DATASET=$2 
MODEL_LEN=9000
CHUNK_SIZE=1024
NUM_PROMPT=200
SLO_LITMIT=$3


if (( ${SLO_LITMIT} == 0 )); then
  OUT_DIR=benchmarks/results/${DATASET%.*}/chunk_prefill 
else
  OUT_DIR=benchmarks/results/${DATASET%.*}/dynamic_batch_${SLO_LITMIT} 
fi

ENDPOINT_TYPE=openai-chat # openai openai-chat vllm
ENDPOINT=/v1/chat/completions # /v1/chat/completions /v1/completions __


request_rate=18
GPU_MEM=0.9
SERVER_PORT=12090
TP=8   
TEST_NAME=$(basename "$MOEL_DIR")
OUT_JSON_DIR="${OUT_DIR}/${TEST_NAME}"
mkdir -p $OUT_JSON_DIR



# iterate over different QPS
for qps in 4 18 'inf'; do
    # remove the surrounding single quote from qps
  if [[ "$qps" == *"inf"* ]] ; then
  echo "qps was $qps"
  qps="inf"
  echo "now qps is $qps"
  fi

  new_TEST_NAME=$TEST_NAME"_qps_"$qps

  echo "Running test case $TEST_NAME with qps $qps"

  vllm bench serve \
  --save-result \
  --save_detailed \
  --result-dir $OUT_JSON_DIR \
  --result-filename ${new_TEST_NAME}.json \
  --request-rate $qps \
  --model ${MOEL_DIR} \
  --endpoint_type ${ENDPOINT_TYPE} \
  --endpoint ${ENDPOINT} \
  --dataset_name ${DATASET_NAME} \
  --dataset_path ${DATASET} \
  --served-model-name $TEST_NAME \
  --host localhost \
  --trust_remote_code \
  --port $SERVER_PORT \
  --num_prompts $NUM_PROMPT > ${OUT_JSON_DIR}/client_${qps}.log 2>&1
done

