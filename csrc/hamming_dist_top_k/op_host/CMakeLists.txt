# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME HammingDistTopK
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(hamming_dist_top_k PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        hamming_dist_top_k_def.cpp
)

target_sources(optiling PRIVATE
        hamming_dist_top_k_tiling.cpp
        hamming_dist_top_k.cpp
        hamming_dist_top_k_split.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        hamming_dist_top_k_tiling.cpp
        hamming_dist_top_k.cpp
        hamming_dist_top_k_split.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        hamming_dist_top_k_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
