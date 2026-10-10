# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        generic_block_sparse_attention_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME GenericBlockSparseAttention
            OPTIONS --cce-auto-sync=off
                    -DASCENDC_DUMP=1
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
    )
else()
    add_modules_sources(OPTYPE generic_block_sparse_attention ACLNNTYPE aclnn_exclude)
endif()
