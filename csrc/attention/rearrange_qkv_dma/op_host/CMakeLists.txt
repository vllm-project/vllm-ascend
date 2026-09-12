# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        rearrange_qkv_dma_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME RearrangeQkvDma
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE rearrange_qkv_dma ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
