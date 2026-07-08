# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRule
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -Wno-error=option-ignored
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_gated_delta_rule ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
