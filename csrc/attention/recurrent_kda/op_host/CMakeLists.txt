# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recurrent_kda_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
        OP_NAME RecurrentKda
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME RecurrentKda
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
else()
    add_modules_sources(OPTYPE recurrent_kda ACLNNTYPE aclnn_exclude)
endif()
