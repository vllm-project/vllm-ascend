#!/bin/bash

# Requirement: 2x NPUs.


# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x NPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

kill_npu_processes() {
  # kill all processes on NPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
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

vllm_benchmarks_path=/vllm-workspace/vllm/benchmarks

launch_chunked_prefill() {
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  # disagg prefill
  ASCEND_RT_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.6 &
  ASCEND_RT_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.6 &
  wait_for_server 8100
  wait_for_server 8200
  python3 $vllm_benchmarks_path/disagg_benchmarks/round_robin_proxy.py &
  sleep 1
}


launch_disagg_prefill() {
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
  # disagg prefill
  ASCEND_RT_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  ASCEND_RT_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 $vllm_benchmarks_path/disagg_benchmarks/disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  dataset_name="sonnet"
  dataset_path="$vllm_benchmarks_path/sonnet_4x.txt"
  num_prompts=100
  qps=$1
  prefix_len=50
  input_len=1024
  output_len=$2
  tag=$3

  python3 $vllm_benchmarks_path/benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len "$output_len" \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --request-rate "$qps"

  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat $vllm_benchmarks_path/sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_output_len=6

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  launch_chunked_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len chunked_prefill
  done
  kill_npu_processes

  launch_disagg_prefill
  for qps in 2 4 6 8; do
  benchmark $qps $default_output_len disagg_prefill
  done
  kill_npu_processes

  python3 $vllm_benchmarks_path/disagg_benchmarks/visualize_benchmark_results.py

}


main "$@"
