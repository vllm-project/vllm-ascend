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

CACHE_HOST="cache-service.nginx-pypi-cache.svc.cluster.local"

WORK_DIR=""

# Route system package downloads through the package cache available to CI runners.
function configure_system_package_mirror() {
  local os_name="$1"
  case "${os_name}" in
    ubuntu)
      if [[ -f /etc/apt/sources.list ]]; then
        sed -Ei \
          's@(ports|archive).ubuntu.com@'"${CACHE_HOST}"':8081@g' \
          /etc/apt/sources.list
      fi
      ;;
    openeuler)
      if [[ -d /etc/yum.repos.d ]]; then
        find /etc/yum.repos.d/ -name "*.repo" -exec \
          sed -Ei \
          's@https?://[^/]+/(openeuler|centos|fedora)@http://'"${CACHE_HOST}"':8081/\1@g' \
          {} +
      fi
      ;;
  esac
}

# Configure network endpoints and retry limits used only by Installation CI.
function configure_ci_environment() {
  local os_name="$1"

  export PIP_DEFAULT_TIMEOUT=300
  export PIP_RETRIES=5
  export PIP_TRUSTED_HOST="${CACHE_HOST}"

  export UV_HTTP_TIMEOUT=120
  export UV_HTTP_RETRIES=3
  export UV_NO_CACHE=1
  export UV_SYSTEM_PYTHON=1
  export UV_INSECURE_HOST="${CACHE_HOST}"

  git config --global http.version HTTP/1.1
  configure_system_package_mirror "${os_name}"
}

# Rewrite documented package sources and clone commands for CI.
function rewrite_ci_block() {
  local block="${1//git clone /ci_git_clone }"

  sed \
    -e "s|https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple|http://${CACHE_HOST}/pypi/simple|g" \
    -e "s|https://download.pytorch.org/whl/cpu/|http://${CACHE_HOST}/whl/cpu/|g" \
    -e "s|https://mirrors.huaweicloud.com/ascend/repos/pypi|http://${CACHE_HOST}/ascend/repos/pypi|g" \
    <<<"${block}"
}

# Retry GitHub clones after removing any incomplete destination directory.
function ci_git_clone() {
  local url="${!#}"
  local repo_dir

  case "${url}" in
    http://*|https://*) ;;
    *) command git clone "$@"; return ;;
  esac

  repo_dir="$(basename "${url%.git}")"
  local attempt
  for attempt in 1 2 3; do
    rm -rf "${repo_dir}"
    if command git clone "$@"; then
      return 0
    fi
    if (( attempt == 3 )); then
      return 1
    fi
    echo "git clone ${repo_dir} failed; retrying (${attempt}/3)..." >&2
    sleep $((attempt * 5))
  done
}

# Extract and source a documented shell block so environment changes remain available.
function run_shell_block() {
  local marker="$1"
  shift

  local block
  block="$(python3 "${DOCTEST_HELPER_PATH}" extract "$@" "${marker}")" || return $?
  block="$(rewrite_ci_block "${block}")"
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

# Verify the installed packages with the standard offline Quick Start example.
function verify_installation_with_quickstart() {
  local verify_dir="${WORK_DIR}/verify"
  mkdir -p "${verify_dir}"

  export MODELSCOPE_HUB_FILE_LOCK=false
  export HF_HUB_OFFLINE=1

  run_shell_block quickstart-modelscope
  run_shell_block quickstart-container-verify
  python3 "${DOCTEST_HELPER_PATH}" extract quickstart-standard-offline >"${verify_dir}/example.py"

  pushd "${verify_dir}" >/dev/null
  run_shell_block quickstart-standard-offline-run
  popd >/dev/null
}

# Build and install from source in a temporary working directory.
function run_source_installation() {
  local source_dir="${WORK_DIR}/source"
  mkdir -p "${source_dir}"

  export MAX_JOBS=23

  pushd "${source_dir}" >/dev/null
  run_shell_block installation-source-install --expand-macros
  popd >/dev/null
}

# Remove the temporary work directory, preserving the exit status.
function cleanup_installation() {
  local exit_code=$?

  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
    rm -rf "${WORK_DIR}"
  fi

  return "${exit_code}"
}

# Run prerequisites, the selected installation method, and offline verification.
function run_installation() {
  local method="$1"
  local os_name

  WORK_DIR="$(mktemp -d)"
  trap cleanup_installation EXIT

  os_name="$(detect_os)"
  configure_ci_environment "${os_name}"
  run_shell_block "installation-common-prerequisites-${os_name}"

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
    *)
      die "Unsupported installation method: ${method}"
      ;;
  esac

  run_shell_block installation-post-standard --expand-macros
  verify_installation_with_quickstart
}

[[ $# -eq 1 && "${1:-}" =~ ^(pip|uv|source)$ ]] ||
  die "Usage: $0 {pip|uv|source}"

run_installation "$1"
