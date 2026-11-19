# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if (NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

set(KV_QUANT_SPARSE_FLASH_ATTENTION_CANN_V2_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    kv_quant_sparse_flash_attention_cann_v2_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME KvQuantSparseFlashAttentionCannV2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_quant_sparse_flash_attention_cann_v2 ACLNNTYPE aclnn)
endif()
