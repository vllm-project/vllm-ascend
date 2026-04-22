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
function install_system_packages() {
    if command -v apt-get >/dev/null; then
        sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
        apt-get update -y && apt install -y curl
    elif command -v yum >/dev/null; then
        yum update -y && yum install -y curl
    else
        echo "Unknown package manager. Please install gcc, g++, numactl-devel, git, curl, and jq manually."
    fi
}

function simple_test() {
  # Do real import test
  python3 -c "import vllm; print(vllm.__version__)"
}

function quickstart_offline_test() {
  # Do real script test
  python3 "${SCRIPT_DIR}/../../examples/offline_inference_npu.py"
}

function quickstart_online_test() {
  install_system_packages
  vllm serve Qwen/Qwen2.5-0.5B-Instruct &
  wait_url_ready "vllm serve" "localhost:8000/v1/models"
  # Do real curl test
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
time simple_test
_info "====> Start quickstart_offline_test"
time quickstart_offline_test
_info "====> Start quickstart_online_test"
time quickstart_online_test
