# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# add_ops_compile_options(
#         OP_NAME SwigluGroupQuant
#         OPTIONS --cce-auto-sync=off
#                 -Wno-deprecated-declarations
#                 -Werror
#                 -mllvm -cce-aicore-hoist-movemask=false
#                 --op_relocatable_kernel_binary=true
# )

# set(swiglu_group_quant_depends transformer/attention/swiglu_group_quant PARENT_SCOPE)

# target_sources(op_host_aclnn PRIVATE
#         op_host/swiglu_group_quant_def.cpp
# )

# target_sources(optiling PRIVATE
#         op_host/swiglu_group_quant_tiling.cpp
# )

# if (NOT BUILD_OPEN_PROJECT)
#     target_sources(opmaster_ct PRIVATE
#         op_host/swiglu_group_quant_tiling.cpp
#     )
# endif ()

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
# )

# target_sources(opsproto PRIVATE
#         op_host/swiglu_group_quant_proto.cpp
# )


if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        swiglu_group_quant_def.cpp
    )
    add_ops_compile_options(
        OP_NAME SwigluGroupQuant
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE swiglu_group_quant ACLNNTYPE aclnn)
endif()
