# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sfa_remap_sparse_indices_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SfaRemapSparseIndices
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sfa_remap_sparse_indices ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
