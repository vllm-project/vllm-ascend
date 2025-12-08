#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))
JOB_NUM="-j${CPU_NUM}"

while [[ $# -gt 0 ]]; do
    case $1 in
    -s)
        PATH_TO_SOURCE="$2"
        shift 2
        ;;
    -b)
        PATH_TO_BUILD="$2"
        shift 2
        ;;
    -p)
        ASCEND_CANN_PACKAGE_PATH="$2"
        shift 2
        ;;
    --autogen-dir)
        ASCEND_AUTOGEN_DIR="$2"
        shift 2
        ;;
    --build-open-project)
        BUILD_OPEN_PROJECT="$2"
        shift 2
        ;;
    --binary-out-dir)
        ASCEND_BINARY_OUT_DIR="$2"
        shift 2
        ;;
    --impl-out-dir)
        ASCEND_IMPL_OUT_DIR="$2"
        shift 2
        ;;
    --op-build-tool)
        OP_BUILD_TOOL="$2"
        shift 2
        ;;
    --ascend-cmake-dir)
        ASCEND_CMAKE_DIR="$2"
        shift 2
        ;;
    --tiling-key)
        TILING_KEY="$2"
        shift 2
        ;;
    --ops-compile-options)
        OPS_COMPILE_OPTIONS="$2"
        shift 2
        ;;
    --check-compatible)
        CHECK_COMPATIBLE="$2"
        shift 2
        ;;
    --ascend-compute_unit)
        ASCEND_COMPUTE_UNIT="$2"
        shift 2
        ;;
    --ascend-op-name)
        ASCEND_OP_NAME="$2"
        shift 2
        ;;
    --op_debug_config)
        OP_DEBUG_CONFIG="$2"
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

function clean() {
    if [ -n "${PATH_TO_BUILD}" ];then
        rm -rf ${PATH_TO_BUILD}
        mkdir -p ${PATH_TO_BUILD}
    fi
}

function convert_string() {
    local _input=$1
    _output=$(echo $_input | sed 's/::/;/g')
    echo "${_output}"
}

function set_env() {
    CONVERT_TILING_KEY="$(convert_string ${TILING_KEY})"

    CONVERT_OPS_COMPILE_OPTIONS="$(convert_string ${OPS_COMPILE_OPTIONS})"

    CONVERT_ASCEND_COMPUTE_UNIT="$(convert_string ${ASCEND_COMPUTE_UNIT})"
}

function build() {
    cd ${PATH_TO_BUILD}
    cmake ${PATH_TO_SOURCE} \
        -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT} \
        -DPREPARE_BUILD=ON \
        -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} \
        -DASCEND_AUTOGEN_DIR=${ASCEND_AUTOGEN_DIR} \
        -DASCEND_BINARY_OUT_DIR=${ASCEND_BINARY_OUT_DIR} \
        -DASCEND_IMPL_OUT_DIR=${ASCEND_IMPL_OUT_DIR} \
        -DOP_BUILD_TOOL=${OP_BUILD_TOOL} \
        -DASCEND_CMAKE_DIR=${ASCEND_CMAKE_DIR} \
        -DCHECK_COMPATIBLE=${CHECK_COMPATIBLE} \
        -DTILING_KEY="${CONVERT_TILING_KEY}" \
        -DOPS_COMPILE_OPTIONS="${CONVERT_OPS_COMPILE_OPTIONS}" \
        -DASCEND_COMPUTE_UNIT=${CONVERT_ASCEND_COMPUTE_UNIT} \
        -DOP_DEBUG_CONFIG=${OP_DEBUG_CONFIG} \
        -DASCEND_OP_NAME=${ASCEND_OP_NAME}

    make ${JOB_NUM} prepare_build
}

function main() {
    clean
    set_env
    build
}

main
