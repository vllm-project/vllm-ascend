# Copyright (c) 2026 TriangleMix contributors.
# This file is licensed under CANN Open Software License Agreement
# Version 2.0. See LICENSE in the repository root.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(triangle_paged_sparse_attention_depends
        attention/sparse_attention_score
        CACHE INTERNAL "Dependencies for triangle_paged_sparse_attention"
    )
    target_sources(op_host_aclnn PRIVATE
        triangle_paged_sparse_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME TrianglePagedSparseAttention
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE triangle_paged_sparse_attention
        ACLNNTYPE aclnn
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
