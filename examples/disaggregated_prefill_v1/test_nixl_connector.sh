#!/bin/bash

# Requirement: 2x NPUs.


# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# Resource: 2x NPU
# Approaches:
#  - Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_npu_processes() {
  # kill all processes on NPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 30000 30100 30200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


launch_disagg_prefill() {
  export VLLM_USE_V1=1
  model="/models/DeepSeek-R1-Distill-Qwen-7B"
  # disagg prefill
  VLLM_NIXL_SIDE_CHANNEL_PORT=5557 ASCEND_RT_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 30100 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"NixlNpuConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_buffer_size":5e9}' &

  VLLM_NIXL_SIDE_CHANNEL_PORT=5558 ASCEND_RT_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 30200 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"NixlNpuConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_buffer_size":5e9}' &

  wait_for_server 30100
  wait_for_server 30200
  python3 load_balance_proxy_server_example.py --host 0.0.0.0 --port 30000 \
    --prefiller-hosts 127.0.0.1 --prefiller-port 30100 \
    --decoder-hosts 127.0.0.1 --decoder-ports 30200 &
  sleep 1
}


benchmark() {
  model="/models/DeepSeek-R1-Distill-Qwen-7B"
  dataset_path="/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
  num_prompts=4

  vllm bench serve \
    --backend vllm \
    --model $model \
    --dataset-name sharegpt \
    --dataset-path $dataset_path \
    --num-prompts $num_prompts \
    --port 30000 \
    --endpoint /v1/completions
  sleep 1
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip3 install pandas datasets

  cd "$(dirname "$0")"

  launch_disagg_prefill

  benchmark

  kill_npu_processes

}


main "$@"
