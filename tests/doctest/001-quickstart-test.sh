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

CURL_TIMEOUT=1
CURL_COOLDOWN=5
CURL_MAX_TRIES=120

function wait_url_ready() {
  local serve_name="$1"
  local url="$2"
  i=0
  while true; do
    _info "===> Waiting for ${serve_name} to be ready...${i}s"
    i=$((i + CURL_COOLDOWN))
    set +e
    curl --silent --max-time "$CURL_TIMEOUT" "${url}" >/dev/null
    result=$?
    set -e
    if [ "$result" -eq 0 ]; then
      break
    fi
    if [ "$i" -gt "$CURL_MAX_TRIES" ]; then
      _info "===> \$CURL_MAX_TRIES exceeded waiting for ${serve_name} to be ready"
      return 1
    fi
    sleep "$CURL_COOLDOWN"
  done
  _info "===> ${serve_name} is ready."
}

function wait_for_exit() {
  local VLLM_PID="$1"
  while kill -0 "$VLLM_PID"; do
    _info "===> Wait for ${VLLM_PID} to exit."
    sleep 1
  done
  _info "===> Wait for ${VLLM_PID} to exit."
}

function simple_test() {
  python3 -c "import vllm; print(vllm.__version__)"
}

function quickstart_offline_test() {
  export VLLM_USE_MODELSCOPE=true
  python3 "${SCRIPT_DIR}/../../examples/offline_inference_npu.py"
}

function quickstart_online_test() {
  export VLLM_USE_MODELSCOPE=true
  vllm serve Qwen/Qwen2.5-0.5B-Instruct &
  wait_url_ready "vllm serve" "localhost:8000/v1/models"
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
  VLLM_PID=$(pgrep -f "vllm serve")
  _info "===> Try kill -2 ${VLLM_PID} to exit."
  kill -2 "$VLLM_PID"
  wait_for_exit "$VLLM_PID"
}

_info "====> Start simple_test"
simple_test
_info "====> Start quickstart_offline_test"
quickstart_offline_test
_info "====> Start quickstart_online_test"
quickstart_online_test
