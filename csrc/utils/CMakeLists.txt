# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_library(ops_utils_tiling_headers INTERFACE)

target_include_directories(ops_utils_tiling_headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog>>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef>>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
        $<INSTALL_INTERFACE:include/ops_adv/utils>
)

target_compile_definitions(ops_utils_tiling_headers INTERFACE
        OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
        OPS_UTILS_LOG_PACKAGE_TYPE=$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,"[Custom]","">
)

add_library(ops_utils_proto_headers INTERFACE)

target_include_directories(ops_utils_proto_headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog>>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef>>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev>>
        $<INSTALL_INTERFACE:include/ops_adv/utils>
)

target_compile_definitions(ops_utils_proto_headers INTERFACE
        OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
        OPS_UTILS_LOG_PACKAGE_TYPE=$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,"[Custom]","">
)

if(NOT BUILD_OPEN_PROJECT)
    install_package(
            PACKAGE ops_adv
            TARGETS ops_utils_tiling_headers ops_utils_proto_headers
            DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc/
            DESTINATION include/ops_adv/utils
    )
endif()
