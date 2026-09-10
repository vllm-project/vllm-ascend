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

set -euo pipefail

scversion="stable"
shellcheck_args=(-S error -s bash)

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if [ -n "${SHELLCHECK_OPTS:-}" ]; then
    # Split caller-provided options the same way shell would.
    # shellcheck disable=SC2206
    extra_shellcheck_args=(${SHELLCHECK_OPTS})
    shellcheck_args+=("${extra_shellcheck_args[@]}")
fi

if [ "$#" -eq 0 ]; then
    while IFS= read -r tracked_file; do
        shellcheck "${shellcheck_args[@]}" "$tracked_file"
    done < <(git ls-files "*.sh")
    exit 0
fi

for file in "$@"; do
    if git check-ignore -q "$file"; then
        continue
    fi

    case "$file" in
        *.csh|*.tcsh)
            # Skip C shell scripts because this checker only supports sh-like shells.
            continue
            ;;
    esac

    shellcheck "${shellcheck_args[@]}" "$file"
done
