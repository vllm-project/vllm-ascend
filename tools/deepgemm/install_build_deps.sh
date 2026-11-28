#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# Shared build prerequisites for A5 images and CI; compiler selection stays local
# to tools/deepgemm/build.py rather than changing the framework's default.
set -euo pipefail

# shellcheck source=/dev/null
source /etc/os-release
sdk_compiler="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/hcc/bin/aarch64-target-linux-gnu-g++"
have_sdk_compiler() { [[ $(uname -m) == aarch64 && -x "$sdk_compiler" ]]; }
case "$ID" in
    ubuntu)
        apt-get update -y
        if [[ -z "${CXX:-}" ]] && ! have_sdk_compiler && ! command -v g++-14 >/dev/null && ! command -v g++-13 >/dev/null; then
            apt-get install -y --no-install-recommends software-properties-common gpg-agent
            add-apt-repository -y ppa:ubuntu-toolchain-r/test
            apt-get update -y
            apt-get install -y --no-install-recommends g++-13
        fi
        apt-get install -y --no-install-recommends libdw-dev
        ;;
    openEuler|openeuler)
        if have_sdk_compiler || [[ -n "${CXX:-}" ]]; then
            yum install -y elfutils-devel
        else
            yum install -y gcc-toolset-14-gcc-c++ elfutils-devel
        fi
        ;;
    *)
        echo "Unsupported DeepGEMM build distribution: $ID" >&2
        exit 1
        ;;
esac
