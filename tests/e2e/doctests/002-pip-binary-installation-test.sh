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
set -euo pipefail

trap clean_venv EXIT
check_npus
create_vllm_venv

_info "====> Install vllm and vllm-ascend from ${VLLM_VERSION}"
if [[ "$VLLM_VERSION" == "main" ]]; then
    # TODO: Update to the vllm version when vllm-ascend v0.9.0rc1 is released
    pip install vllm==0.8.5.post1
    pip install vllm-ascend==0.8.5rc1
fi

if [[ "$VLLM_VERSION" == "v0.7.3" ]]; then
    pip install vllm==0.7.3
    pip install vllm-ascend==0.7.3 --extra-index https://download.pytorch.org/whl/cpu/
fi

pip list | grep vllm

# Verify the installation
_info "====> Run offline example test"
python3 "${SCRIPT_DIR}/../../examples/offline_inference_npu.py"
