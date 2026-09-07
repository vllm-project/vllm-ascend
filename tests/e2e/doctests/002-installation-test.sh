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

INSTALL_WORK_DIR=""
VERIFY_RUNTIME_DIR=""

# Extract and source a documented shell block so environment changes remain available.
function run_shell_block() {
  local marker="$1"
  shift
  local block
  block="$(python3 "${DOCTEST_HELPER_PATH}" extract "$@" "${marker}")" || return $?
  source /dev/stdin <<<"${block}"
}

# Identify the supported container OS used to select prerequisite commands.
function detect_os() {
  [[ -r /etc/os-release ]] || die "Cannot detect the operating system: /etc/os-release is missing."
  local os_id
  os_id="$(. /etc/os-release && echo "${ID,,}")"
  case "${os_id}" in
    ubuntu) echo ubuntu ;;
    openeuler) echo openeuler ;;
    *) die "Unsupported operating system '${os_id}'. Expected Ubuntu or openEuler." ;;
  esac
}

# Run the documented system prerequisites for the detected operating system.
function run_prerequisites() {
  local os_name="$1"
  run_shell_block "installation-common-prerequisites-${os_name}"
}

# Verify the installed packages with the standard offline Quick Start example.
function verify_installation_with_quickstart() {
  VERIFY_RUNTIME_DIR="$(mktemp -d)"
  export MODELSCOPE_HUB_FILE_LOCK=false
  export HF_HUB_OFFLINE=1
  run_shell_block quickstart-modelscope
  run_shell_block quickstart-container-verify
  python3 "${DOCTEST_HELPER_PATH}" extract quickstart-standard-offline >"${VERIFY_RUNTIME_DIR}/example.py"
  (
    cd "${VERIFY_RUNTIME_DIR}"
    run_shell_block quickstart-standard-offline-run
  )
}

# Build and install from source in a temporary working directory.
function run_source_installation() {
  INSTALL_WORK_DIR="$(mktemp -d)"
  (
    cd "${INSTALL_WORK_DIR}"
    run_shell_block installation-source-install --expand-macros
  )
}

# Remove temporary installation and verification directories, preserving the exit status.
function cleanup_installation() {
  local exit_code=$?
  if [[ -n "${INSTALL_WORK_DIR}" && -d "${INSTALL_WORK_DIR}" ]]; then
    rm -rf "${INSTALL_WORK_DIR}"
  fi
  if [[ -n "${VERIFY_RUNTIME_DIR}" && -d "${VERIFY_RUNTIME_DIR}" ]]; then
    rm -rf "${VERIFY_RUNTIME_DIR}"
  fi
  return "${exit_code}"
}

# Run prerequisites, the selected installation method, and offline verification.
function run_installation() {
  local method="$1"
  local os_name
  source "${DOCTEST_DIR}/scripts/common.sh"
  trap cleanup_installation EXIT
  os_name="$(detect_os)"
  run_prerequisites "${os_name}"
  python3 -c 'import yaml' 2>/dev/null || python3 -m pip install PyYAML
  case "${method}" in
    pip)
      run_shell_block installation-pip-install --expand-macros
      run_shell_block installation-pip-device-check
      ;;
    uv)
      run_shell_block installation-uv-bootstrap
      run_shell_block installation-uv-install --expand-macros
      run_shell_block installation-uv-device-check
      ;;
    source)
      run_source_installation
      ;;
  esac
  run_shell_block installation-post-standard --expand-macros
  verify_installation_with_quickstart
}

[[ $# -eq 1 ]] || { echo "Usage: $0 {pip|uv|source}" >&2; exit 1; }

run_installation "$1"
