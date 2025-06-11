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
trap clean_venv EXIT

function install_binary_test() {

    create_vllm_venv

    PIP_VLLM_VERSION=$(get_version pip_vllm_version)
    PIP_VLLM_ASCEND_VERSION=$(get_version pip_vllm_ascend_version)
    _info "====> Install vllm==${PIP_VLLM_VERSION} and vllm-ascend ${PIP_VLLM_ASCEND_VERSION}"

    pip install vllm=="$(get_version pip_vllm_version)"
    pip install vllm-ascend=="$(get_version pip_vllm_ascend_version)"

    pip list | grep vllm

    # Verify the installation
    _info "====> Run offline example test"
    pip install modelscope
    python3 "${SCRIPT_DIR}/../../examples/offline_inference_npu.py"

}

_info "====> Start install_binary_test"
install_binary_test
