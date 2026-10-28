# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(sparse_flash_mla_depends attention/common CACHE INTERNAL "Dependencies for sparse_flash_mla")
    target_sources(op_host_aclnn PRIVATE
        sparse_flash_mla_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SparseFlashMla
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

add_modules_sources(OPTYPE sparse_flash_mla ACLNNTYPE aclnn)

include(${CMAKE_CURRENT_SOURCE_DIR}/checkers/checker_sources.cmake)
set(SPARSE_MLA_CHECKER_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/checkers/sparse_flash_mla_checker.cpp
)

add_tiling_modules()
add_sparse_mla_common_checker_sources(${OPHOST_NAME}_tiling_obj)
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ${SPARSE_MLA_CHECKER_SRC_FILES}
    sparse_flash_mla_tiling.cpp
)
