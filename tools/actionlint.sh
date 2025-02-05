#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
# Copyright 2023 The vLLM team.
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
#

if command -v actionlint &> /dev/null; then
    # NOTE: avoid check .github/workflows/vllm_ascend_test.yaml becase sel-hosted runner `npu-arm64` is unknown
    actionlint .github/workflows/*.yml .github/workflows/mypy.yaml
    exit 0
elif [ -x ./actionlint ]; then
    ./actionlint .github/workflows/*.yml .github/workflows/mypy.yaml
    exit 0
fi

# download a binary to the current directory - v1.7.3
bash <(curl https://raw.githubusercontent.com/rhysd/actionlint/aa0a7be8e566b096e64a5df8ff290ec24fa58fbc/scripts/download-actionlint.bash)
./actionlint  .github/workflows/*.yml .github/workflows/mypy.yaml
