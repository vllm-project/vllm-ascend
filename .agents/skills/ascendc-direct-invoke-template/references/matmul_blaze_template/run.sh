#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

die() { echo "ERROR: $*" >&2; exit 1; }

SKIP_BUILD=0
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --clean)      CLEAN=1 ;;
    esac
done

if [ "${CLEAN}" -eq 1 ]; then
    echo "Cleaning temporary files..."
    rm -rf build
    rm -rf op_kernel/include/blaze
    rm -rf op_kernel/include/tensor_api
    rm -rf scripts/input
    rm -rf scripts/output
    echo "Done."
    exit 0
fi

shift $((OPTIND - 1)) 2>/dev/null || true
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --skip-build|--clean) ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

OP_NAME="matmul_blaze"
M="${POSITIONAL[0]:-16}"
K="${POSITIONAL[1]:-128}"
N="${POSITIONAL[2]:-16384}"
TRANS_A="${POSITIONAL[3]:-false}"
TRANS_B="${POSITIONAL[4]:-true}"

echo "=== [1/5] CANN ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

echo "=== [2/5] dependencies ==="
INCLUDE_DIR="${SCRIPT_DIR}/op_kernel/include"
if [ ! -d "${INCLUDE_DIR}/blaze" ] || [ ! -d "${INCLUDE_DIR}/tensor_api" ]; then
    echo "Pulling blaze + tensor_api from ops-tensor..."
    OPS_TENSOR_TMP=$(mktemp -d)
    git clone --depth 1 https://gitcode.com/cann/ops-tensor.git "${OPS_TENSOR_TMP}" || die "git clone ops-tensor failed"
    cp -r "${OPS_TENSOR_TMP}/include/blaze" "${INCLUDE_DIR}/blaze"
    cp -r "${OPS_TENSOR_TMP}/include/tensor_api" "${INCLUDE_DIR}/tensor_api"
    rm -rf "${OPS_TENSOR_TMP}"
    echo "Done."
else
    echo "blaze + tensor_api already present."
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [3/5] skip build ==="
else
    echo "=== [3/5] build ==="
    mkdir -p build
    cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" || die "cmake failed"
    cmake --build "${SCRIPT_DIR}/build" --target "${OP_NAME}" --parallel "$(nproc 2>/dev/null || echo 4)" || die "build failed"
fi

echo "=== [4/5] gen data (mxfp8 M=${M} K=${K} N=${N} tA=${TRANS_A} tB=${TRANS_B}) ==="
cd build
python3 ../scripts/gen_data_mxfp8.py "${M}" "${K}" "${N}" "${TRANS_A}" "${TRANS_B}" || die "gen_data_mxfp8.py failed"

echo "=== [5/5] run + verify ==="
rm -f output/npu_out.bin
"./${OP_NAME}" "${M}" "${K}" "${N}" "${TRANS_A}" "${TRANS_B}" || die "kernel failed (exit $?)"
[ -f output/npu_out.bin ] || die "output/npu_out.bin not found"

python3 ../scripts/verify_result.py "${M}" "${N}" || die "verify failed"

echo "=== done ==="
exit 0
