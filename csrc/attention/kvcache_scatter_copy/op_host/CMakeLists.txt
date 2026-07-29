# Copyright (c) 2025 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    # KvcacheScatterCopy uses the standard generated ACLNN API path.
    target_sources(op_host_aclnn PRIVATE
        kvcache_scatter_copy_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME KvcacheScatterCopy
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE kvcache_scatter_copy
        ACLNNTYPE aclnn
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
