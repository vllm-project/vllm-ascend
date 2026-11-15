# Copyright (c) 2026 SeaGaruda
# Licensed under CANN Open Software License Agreement Version 2.0.

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE top_k_per_row_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME TopKPerRow
    OPTIONS -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE top_k_per_row ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
