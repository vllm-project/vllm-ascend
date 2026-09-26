# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        fused_rearrange_mix_qkv_gdn_gating_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedRearrangeMixQkvGdnGating
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_rearrange_mix_qkv_gdn_gating ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
