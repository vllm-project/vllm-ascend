# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CURRENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)

set(COMMON_KERNEL_UTILS_DIR "${CMAKE_SOURCE_DIR}/common")
get_filename_component(COMMON_KERNEL_UTILS_DIR_ABS ${COMMON_KERNEL_UTILS_DIR} ABSOLUTE)

add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleFwdH
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -I${CATLASS_INCLUDE_DIR_ABS}
                -I${COMMON_KERNEL_UTILS_DIR_ABS}
)

# Host (aclnn)
target_sources(op_host_aclnnExc PRIVATE
    chunk_gated_delta_rule_fwd_h_def.cpp
)

target_sources(opapi PRIVATE
    op_api/chunk_gated_delta_rule_fwd_h.cpp
    op_api/aclnn_chunk_gated_delta_rule_fwd_h.cpp
)

# Tiling
target_sources(optiling PRIVATE
    chunk_gated_delta_rule_fwd_h_tiling.cpp 
)
message(STATUS "Catlass绝对路径：${CATLASS_INCLUDE_DIR_ABS}")

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CATLASS_INCLUDE_DIR_ABS}
)
file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_chunk_gated_delta_rule_fwd_h.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
