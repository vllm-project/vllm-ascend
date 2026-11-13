#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# Shared build prerequisites for A5 images and CI; compiler selection stays local
# to tools/deepgemm/build.py rather than changing the framework's default.
set -euo pipefail

# shellcheck source=/dev/null
source /etc/os-release
case "$ID" in
    ubuntu)
        apt-get update -y
        apt-get install -y --no-install-recommends software-properties-common
        add-apt-repository -y ppa:ubuntu-toolchain-r/test
        apt-get update -y
        apt-get install -y --no-install-recommends g++-13 libdw-dev
        ;;
    openEuler|openeuler)
        yum install -y gcc-toolset-14-gcc-c++ elfutils-devel
        ;;
    *)
        echo "Unsupported DeepGEMM build distribution: $ID" >&2
        exit 1
        ;;
esac

# Build-only parser versions mirror pyproject.toml. Keep them out of runtime
# requirements so clients such as ais_bench can use their own parser version.
python3 -m pip install tree-sitter==0.25.2 tree-sitter-cpp==0.23.4
