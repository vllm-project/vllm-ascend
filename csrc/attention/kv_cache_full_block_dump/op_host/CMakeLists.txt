# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME KvCacheFullBlockDump
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        kv_cache_full_block_dump_def.cpp
)

target_sources(optiling PRIVATE
        kv_cache_full_block_dump_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            kv_cache_full_block_dump_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        kv_cache_full_block_dump_proto.cpp
)
