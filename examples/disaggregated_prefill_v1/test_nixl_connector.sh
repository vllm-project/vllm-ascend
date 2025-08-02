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
  ASCEND_RT_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 30100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_buffer_size":5e9}' &

  ASCEND_RT_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 30200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_buffer_size":5e9}' &

  wait_for_server 30100
  wait_for_server 30200
  python3 toy_proxy_server.py --host 0.0.0.0 --port 30000 --prefiller-hosts localhost --prefiller-port 30100 --decoder-hosts localhost --decoder-ports 30200 &
  sleep 1
}


benchmark() {
  model="/models/DeepSeek-R1-Distill-Qwen-7B"
  num_prompts=1024
  input_len=4096
  output_len=1536
  cd /vllm-workspace/vllm/benchmarks
  python3 benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --random-input-len $input_len \
      --random-output-len $output_len \
      --num-prompts $num_prompts \
      --ignore-eos \
      --model $model \
      --tokenizer $model \
      --host localhost \
      --port 30000 \
      --endpoint /v1/completions \
      --max-concurrency 10 \
      --request-rate 10

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
