# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME TransposeKvCacheByBlock
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        transpose_kv_cache_by_block_def.cpp
)

target_sources(optiling PRIVATE
        transpose_kv_cache_by_block_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        transpose_kv_cache_by_block_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        transpose_kv_cache_by_block_proto.cpp
)

