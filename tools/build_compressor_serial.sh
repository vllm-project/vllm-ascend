#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source the CANN environment before building}"
cmake -S "$repo_root/csrc/attention/compressor_serial" \
    -B "$repo_root/build/compressor_serial" -DCMAKE_BUILD_TYPE=Release
cmake --build "$repo_root/build/compressor_serial" --parallel "${MAX_JOBS:-8}"
