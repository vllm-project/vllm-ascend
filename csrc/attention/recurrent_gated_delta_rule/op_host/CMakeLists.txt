# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recurrent_gated_delta_rule_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME RecurrentGatedDeltaRule
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
else()
    add_modules_sources(OPTYPE recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)
endif()

