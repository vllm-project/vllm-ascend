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

function install_system_packages() {
    if command -v apt-get >/dev/null; then
        sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
        apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
    elif command -v yum >/dev/null; then
        yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq
    else
        echo "Unknown package manager. Please install curl manually."
    fi
}

function config_pip_mirror() {
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
}

function install_binary_test() {

    install_system_packages
    config_pip_mirror
    create_vllm_venv

    PIP_VLLM_VERSION=$(get_version pip_vllm_version)
    VLLM_VERSION=$(get_version vllm_version)
    PIP_VLLM_ASCEND_VERSION=$(get_version pip_vllm_ascend_version)
    _info "====> Install vllm==${PIP_VLLM_VERSION} and vllm-ascend ${PIP_VLLM_ASCEND_VERSION}"

    # Setup extra-index-url for x86 & torch_npu dev version
    pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

    if [[ "${VLLM_VERSION} " != "v0.11.0rc3" ]]; then
        # The vLLM version already in pypi, we install from pypi.
        pip install vllm=="${PIP_VLLM_VERSION}"
    else
        # The vLLM version not in pypi, we install from source code with a specific tag.
        git clone --depth 1 --branch "${VLLM_VERSION}" https://github.com/vllm-project/vllm
        cd vllm
        VLLM_TARGET_DEVICE=empty pip install -v -e .
        cd ..
    fi

    pip install vllm-ascend=="${PIP_VLLM_ASCEND_VERSION}"

    pip list | grep vllm

    # Verify the installation
    _info "====> Run offline example test"
    pip install modelscope
    cd ${SCRIPT_DIR}/../../examples && python3 ./offline_inference_npu.py
    cd -

}

_info "====> Start install_binary_test"
time install_binary_test
