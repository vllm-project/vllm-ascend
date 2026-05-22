# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        kv_quant_sparse_attn_sharedkv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME KvQuantSparseAttnSharedkv
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_quant_sparse_attn_sharedkv ACLNNTYPE aclnn)
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        kv_quant_sparse_attn_sharedkv_check_consistancy.cpp
        kv_quant_sparse_attn_sharedkv_check_existance.cpp
        kv_quant_sparse_attn_sharedkv_check_feature.cpp
        kv_quant_sparse_attn_sharedkv_check_single_para.cpp
        kv_quant_sparse_attn_sharedkv_check.cpp
        kv_quant_sparse_attn_sharedkv_tiling.cpp
    )
endif()