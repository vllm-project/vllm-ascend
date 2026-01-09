#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [ "$CI" -eq 1 ]; then
    set -e
fi

if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

run_mypy() {
    echo "Running mypy on $1"
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy vllm_ascend
run_mypy examples
run_mypy tests
