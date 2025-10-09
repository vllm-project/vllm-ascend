#!/bin/bash
set -ex
pgrep python | xargs -r kill -9
sleep 1
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



MOEL_DIR=$1  
SLO_LITMIT=$2
DATASET=$3   
NUM_PROMPT=$4
TP=$5


if (( ${SLO_LITMIT} == 0 )); then
  OUT_DIR=benchmarks/results/${DATASET%.*}/chunk_prefill 
else
  OUT_DIR=benchmarks/results/${DATASET%.*}/dynamic_batch_${SLO_LITMIT} 
fi

ENDPOINT_TYPE=openai # openai openai-chat vllm
ENDPOINT=/v1/completions # /v1/chat/completions /v1/completions __

DATASET_NAME=sharegpt  # random sharegpt hf
GPU_MEM=0.9
SERVER_PORT=12091
MODEL_LEN=9000
CHUNK_SIZE=1024
 
TEST_NAME=$(basename "$MOEL_DIR")
OUT_JSON_DIR="${OUT_DIR}/${TEST_NAME}"
mkdir -p $OUT_JSON_DIR

# extra args:
    # --enable-expert-parallel  
    # --no-enable-prefix-caching \
    # --quantization ascend \ 
    # --enforce-eager \
    # "torchair_graph_config":{"enabled":true}
vllm serve ${MOEL_DIR} \
    --max-num-seqs 256 \
    --block-size 128 \
    --tensor_parallel_size $TP \
    --load_format dummy \
    --max_num_batched_tokens $CHUNK_SIZE \
    --max_model_len ${MODEL_LEN} \
    --host localhost \
    --port $SERVER_PORT \
    --gpu-memory-utilization $GPU_MEM \
    --trust-remote-code \
    --data-parallel-size 2 \
    --additional_config '{"ascend_scheduler_config":{"enabled": true, "enable_chunked_prefill": true, "SLO_limits_for_dynamic_batch":'${SLO_LITMIT}'}}' > ${OUT_JSON_DIR}/server_tmp.log 2>&1 &

server_pid=$!

# wait until the server is alive
wait_for_server $SERVER_PORT $ENDPOINT

# iterate over different QPS
for qps in 4 18 32 'inf'; do
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
  --host localhost \
  --trust_remote_code \
  --port $SERVER_PORT \
  --num_prompts $NUM_PROMPT > ${OUT_JSON_DIR}/client_${qps}.log 2>&1
done


# terminate process
sleep 15
kill -9 $server_pid
pkill -f timeout
pkill -f python
sleep 30