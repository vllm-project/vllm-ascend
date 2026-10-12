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
# shellcheck disable=SC1090,SC1091

set -Eeuo pipefail

DOCTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTEST_HELPER_PATH="${DOCTEST_DIR}/scripts/doctest_helper.py"
source "${DOCTEST_DIR}/scripts/common.sh"

VLLM_PID=""
RUNTIME_DIR=""

# Extract and source a documented shell block so environment changes remain available.
function run_shell_block() {
  local marker="$1"
  local block
  block="$(python3 "${DOCTEST_HELPER_PATH}" extract "${marker}")" || return $?
  source /dev/stdin <<<"${block}"
}

# Extract and run the offline example for the selected marker prefix.
function run_offline() {
  local marker_prefix="$1"
  python3 "${DOCTEST_HELPER_PATH}" extract "${marker_prefix}-offline" >"${RUNTIME_DIR}/example.py"
  (
    cd "${RUNTIME_DIR}"
    run_shell_block "${marker_prefix}-offline-run"
  )
}

# Start the documented service, run API checks, then stop it and wait for exit.
function run_online() {
  local marker_prefix="$1"
  pushd "${RUNTIME_DIR}" >/dev/null
  run_shell_block "${marker_prefix}-online-serve"
  VLLM_PID="$!"
  popd >/dev/null
  wait_for_url_ready "vllm serve" "localhost:8000/v1/models"
  run_shell_block "${marker_prefix}-online-model-list"
  run_shell_block "${marker_prefix}-online-completion"
  run_shell_block "${marker_prefix}-online-stop"
  wait_for_process_exit "${VLLM_PID}"
  VLLM_PID=""
}

# Stop a remaining service and remove temporary files, preserving the exit status.
function cleanup_quickstart() {
  local exit_code=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -2 "${VLLM_PID}" 2>/dev/null || true
    wait_for_process_exit "${VLLM_PID}" || true
  fi
  if [[ -n "${RUNTIME_DIR}" && -d "${RUNTIME_DIR}" ]]; then
    rm -rf "${RUNTIME_DIR}"
  fi
  return "${exit_code}"
}

# Run the shared checks and then offline and online examples for one device.
function run_quickstart() {
  local device="$1"
  local marker_prefix
  export MODELSCOPE_HUB_FILE_LOCK=false
  export HF_HUB_OFFLINE=1
  trap cleanup_quickstart EXIT
  RUNTIME_DIR="$(mktemp -d)"

  case "${device}" in
    a2) marker_prefix=quickstart-standard ;;
    310p) marker_prefix=quickstart-300i-duo ;;
  esac

  run_shell_block quickstart-modelscope
  run_shell_block quickstart-container-verify
  run_offline "${marker_prefix}"
  run_online "${marker_prefix}"
}

[[ $# -eq 1 && "${1:-}" =~ ^(a2|310p)$ ]] || die "Usage: $0 {a2|310p}"

run_quickstart "$1"
