# SPDX-License-Identifier: Apache-2.0
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE vllm_mhc_expand_def.cpp)
endif()
add_ops_compile_options(OP_NAME VllmMhcExpand OPTIONS --cce-auto-sync=off)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE vllm_mhc_expand ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
