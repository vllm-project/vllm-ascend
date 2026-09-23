# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
# Both the operator scan and dependency expansion can visit this directory.
include_guard(GLOBAL)

set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)
set(COMMON_KERNEL_UTILS_DIR "${CMAKE_SOURCE_DIR}/moe/common")
get_filename_component(COMMON_KERNEL_UTILS_DIR_ABS ${COMMON_KERNEL_UTILS_DIR} ABSOLUTE)

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_fwd_h_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkFwdH
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CATLASS_INCLUDE_DIR_ABS}
            -I${COMMON_KERNEL_UTILS_DIR_ABS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_fwd_h ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CATLASS_INCLUDE_DIR_ABS}
    )
endif()
