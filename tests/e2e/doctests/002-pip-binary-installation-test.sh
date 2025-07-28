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
    PIP_VLLM_ASCEND_VERSION=$(get_version pip_vllm_ascend_version)
    _info "====> Install vllm==${PIP_VLLM_VERSION} and vllm-ascend ${PIP_VLLM_ASCEND_VERSION}"

    # Setup extra-index-url for x86 & torch_npu dev version
    pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

    pip install vllm=="$(get_version pip_vllm_version)"
    pip install vllm-ascend=="$(get_version pip_vllm_ascend_version)"

    # TODO(yikun): Remove this when 0.9.1rc2 is released
    # https://github.com/vllm-project/vllm-ascend/issues/2046
    if [ "$PIP_VLLM_ASCEND_VERSION" == "0.9.1rc1" ] || [ "$PIP_VLLM_ASCEND_VERSION" == "0.9.0rc2" ] ; then
        pip install "transformers<4.53.0"
    fi

    pip list | grep vllm

    # Verify the installation
    _info "====> Run offline example test"
    pip install modelscope
    python3 "${SCRIPT_DIR}/../../examples/offline_inference_npu.py"

}

_info "====> Start install_binary_test"
install_binary_test
