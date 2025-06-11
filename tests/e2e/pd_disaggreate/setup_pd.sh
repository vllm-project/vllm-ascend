#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

function run_prefill_instance() {
  local model_name=$1
  local tp_size=$2
  local prefill_port=$3
  local register_port=$4
  local prefill_device_ips=$5
  local decode_device_ips=$6

  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"
  # Start prefill instance

  KV_CONFIG=$(jq -n \
    --arg kv_connector "AscendSimpleConnector" \
    --arg kv_buffer_device "npu" \
    --arg kv_role "kv_producer" \
    --argjson kv_parallel_size 8 \
    --arg kv_port 11001 \
    --argjson prefill_device_ips "$prefill_device_ips" \
    --argjson decode_device_ips "$decode_device_ips" \
    --argjson llmdatadist_comm_port 26000 \
    --arg proxy_ip "0.0.0.0" \
    --argjson proxy_port "$register_port" \
    --argjson http_port "$prefill_port" \
    '{
      "kv_connector": $kv_connector,
      "kv_buffer_device": $kv_buffer_device,
      "kv_role": $kv_role,
      "kv_parallel_size": $kv_parallel_size,
      "kv_port": $kv_port,
      "kv_connector_extra_config": {
        "prefill_device_ips": $prefill_device_ips,
        "decode_device_ips": $decode_device_ips,
        "llmdatadist_comm_port": $llmdatadist_comm_port,
        "proxy_ip": $proxy_ip,
        "proxy_port": $proxy_port,
        "http_port": $http_port
      }
    }')

  # start prefill instance
  ASCEND_RT_VISIBLE_DEVICES=0 vllm serve $model_name \
  --host 0.0.0.0 \
  --port $prefill_port \
  --tensor-parallel-size $tp_size \
  --served-model-name Deepseek \
  --max-model-len 2000 \
  --trust-remote-code \
  --kv-transfer-config "$KV_CONFIG"
}



function run_decode_instance() {
  # Start decode instance
  local model_name=$1
  local tp_size=$2
  local decode_port=$3
  local register_port=$4
  local prefill_device_ips=$5
  local decode_device_ips=$6

  KV_CONFIG=$(jq -n \
    --arg kv_connector "AscendSimpleConnector" \
    --arg kv_buffer_device "npu" \
    --arg kv_role "kv_consumer" \
    --argjson kv_parallel_size 8 \
    --arg kv_port 21001 \
    --argjson prefill_device_ips "$prefill_device_ips" \
    --argjson decode_device_ips "$decode_device_ips" \
    --argjson llmdatadist_comm_port 26000 \
    --arg proxy_ip "0.0.0.0" \
    --argjson proxy_port "$register_port" \
    --argjson http_port "$decode_port" \
    '{
      "kv_connector": $kv_connector,
      "kv_buffer_device": $kv_buffer_device,
      "kv_role": $kv_role,
      "kv_parallel_size": $kv_parallel_size,
      "kv_port": $kv_port,
      "kv_connector_extra_config": {
        "prefill_device_ips": $prefill_device_ips,
        "decode_device_ips": $decode_device_ips,
        "llmdatadist_comm_port": $llmdatadist_comm_port,
        "proxy_ip": $proxy_ip,
        "proxy_port": $proxy_port,
        "http_port": $http_port
      }
    }')

  # start decode instance
  ASCEND_RT_VISIBLE_DEVICES=1 vllm serve $model_name \
    --host 0.0.0.0 \
    --port $decode_port \
    --tensor-parallel-size $tp_size \
    --seed 1024 \
    --served-model-name Deepseek \
    --max-model-len 2000 \
    --max-num-batched-tokens 2000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config "$KV_CONFIG"
}

function run_proxy_server() {
  # Build the command for the proxy server with all the hosts and ports
  register_port=$1
  proxy_port=$2
  PROXY_CMD="python examples/disaggregated_prefill/p2p_disaggrefated_prefill_proxy.py --http-port $proxy_port --register-port $register_port"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &
}
