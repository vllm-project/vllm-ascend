# Copyright (c) 2025 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME KvcacheScatterCopy
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                --op_relocatable_kernel_binary=true
)

# KvcacheScatterCopy uses the standard generated ACLNN API path. Register its
# op definition with op_host_aclnn so the custom package exports
# aclnnKvcacheScatterCopy and aclnnKvcacheScatterCopyGetWorkspaceSize.
target_sources(op_host_aclnn PRIVATE
        kvcache_scatter_copy_def.cpp
)

target_sources(optiling PRIVATE
        kvcache_scatter_copy_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            kvcache_scatter_copy_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        kvcache_scatter_copy_proto.cpp
)
