# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        fused_gdn_gating_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedGdnGating
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_gdn_gating ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
