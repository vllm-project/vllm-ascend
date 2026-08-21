# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
set(CURRENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkKdaFwd
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -I${CATLASS_INCLUDE_DIR_ABS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CATLASS_INCLUDE_DIR_ABS}
    )
endif()
