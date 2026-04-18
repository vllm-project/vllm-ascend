# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
#add_definitions(-DMAX_BLOCK_NUM_=3)
add_ops_compile_options(
        OP_NAME StoreKVBlock
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

# -o0
# -g
# --cce-ignore-always-inline=true
target_sources(op_host_aclnn PRIVATE
        store_kv_block_def.cpp
)

target_sources(optiling PRIVATE
        store_kv_block_tiling.cpp
        store_kv_block_common.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        store_kv_block_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
       store_kv_block_infershape.cpp
)

target_link_libraries(optiling
    PRIVATE
)