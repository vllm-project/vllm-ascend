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

show_help() {
    cat <<'EOF'
Usage: bash format.sh [changed|all|ci]

Modes:
  changed  Run pre-commit on files changed in the working tree. This is the
           default and is intended for fast local iteration.
  all      Run pre-commit on every tracked file with local hook stages.
  ci       Run pre-commit on every tracked file, including manual CI hooks.

Examples:
  bash format.sh
  bash format.sh changed
  bash format.sh all
  bash format.sh ci
EOF
}

case "${1:-changed}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
esac

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
case "${1:-changed}" in
    changed)
        check_command git
        if ! git rev-parse --is-inside-work-tree &> /dev/null; then
            echo "Error: not inside a git work tree." >&2
            exit 1
        fi

        changed_files=()
        while IFS= read -r line; do
            changed_files+=("$line")
        done < <(
            {
                git diff --name-only --diff-filter=ACMR --cached
                git diff --name-only --diff-filter=ACMR
                git ls-files --others --exclude-standard
            } | sort -u
        )

        if [[ ${#changed_files[@]} -eq 0 ]]; then
            echo "No changed files to format."
            exit 0
        fi

        pre-commit run --files "${changed_files[@]}"
        ;;
    all)
        pre-commit run --all-files
        ;;
    ci)
        pre-commit run --all-files --hook-stage manual
        ;;
    *)
        echo "Unknown format mode: $1" >&2
        show_help >&2
        exit 2
        ;;
esac
