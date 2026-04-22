#!/usr/bin/env bash

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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❓❓$1 is not installed, please run:"
        echo "# Install lint deps"
        echo "pip install -r requirements-lint.txt"
        echo "# (optional) Enable git commit pre check"
        echo "pre-commit install"
        echo ""
        echo "See step by step contribution guide:"
        echo "https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution"
        exit 1
    fi
}

check_command pre-commit

# TODO: cleanup SC exclude
export SHELLCHECK_OPTS="--exclude=SC2046,SC2006,SC2086"
if [[ "$1" != 'ci' ]]; then
    pre-commit run --all-files
else
    pre-commit run --all-files --hook-stage manual
fi
