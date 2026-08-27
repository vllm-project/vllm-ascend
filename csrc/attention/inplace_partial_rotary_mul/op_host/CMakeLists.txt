# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# add_ops_compile_options(
#         OP_NAME InplacePartialRotaryMul
#         OPTIONS --cce-auto-sync=on
#                 -Wno-deprecated-declarations
#                 -Werror
#                 -mllvm -cce-aicore-hoist-movemask=false
#                 --op_relocatable_kernel_binary=true
# )

# set(inplace_partial_rotary_mul_depends transformer/posembedding/inplace_partial_rotary_mul PARENT_SCOPE)

# target_sources(op_host_aclnn PRIVATE
#         op_host/inplace_partial_rotary_mul_def.cpp
# )

# target_sources(optiling PRIVATE
#         op_host/inplace_partial_rotary_mul_tiling.cpp
# 	op_host/inplace_partial_rotary_mul_a3_tiling.cpp
#         op_host/rope_regbase_tiling_base.cpp
#         op_host/rope_regbase_tiling_a_and_b.cpp
#         op_host/rope_regbase_tiling_ab.cpp
#         op_host/rope_regbase_tiling_aba_and_ba.cpp
#         op_host/rope_regbase_tiling_bab.cpp
# )

# if (NOT BUILD_OPEN_PROJECT)
#     target_sources(opmaster_ct PRIVATE
#         op_host/inplace_partial_rotary_mul_tiling.cpp
# 	op_host/inplace_partial_rotary_mul_a3_tiling.cpp
#         op_host/rope_regbase_tiling_base.cpp
#         op_host/rope_regbase_tiling_a_and_b.cpp
#         op_host/rope_regbase_tiling_ab.cpp
#         op_host/rope_regbase_tiling_aba_and_ba.cpp
#         op_host/rope_regbase_tiling_bab.cpp
#     )
# endif ()

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
# )

# target_sources(opsproto PRIVATE
#         op_host/inplace_partial_rotary_mul_proto.cpp
# )

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        inplace_partial_rotary_mul_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME InplacePartialRotaryMul
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)


if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE inplace_partial_rotary_mul ACLNNTYPE aclnn)
endif()