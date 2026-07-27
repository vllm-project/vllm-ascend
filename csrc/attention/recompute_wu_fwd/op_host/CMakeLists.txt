# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)
set(recompute_wu_fwd_depends
    third_party/catlass/include/catlass
    third_party/catlass/include/tla
    CACHE INTERNAL "Dependencies for recompute_wu_fwd")

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recompute_wu_fwd_def.cpp
    )
endif()


add_modules_sources(OPTYPE recompute_wu_fwd ACLNNTYPE aclnn_exclude)
add_ops_tiling_keys(
    OP_NAME RecomputeWUFwd
    TILING_KEYS 1 2
)
if (NOT BUILD_OPS_RTY_KERNEL)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CATLASS_INCLUDE_DIR_ABS}
    )
endif()
add_ops_compile_options(
    OP_NAME RecomputeWUFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CATLASS_INCLUDE_DIR_ABS}
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME RecomputeWUFwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS --cce-auto-sync=off
                -mllvm -cce-aicore-dcci-before-kernel-end=false
                -Wno-deprecated-declarations
                -I${CATLASS_INCLUDE_DIR_ABS}
    )
endif()
