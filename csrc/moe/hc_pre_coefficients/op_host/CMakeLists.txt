# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    set(hc_pre_coefficients_depends moe/hc_pre CACHE INTERNAL
        "Kernel source dependency for HcPreCoefficients")
    target_sources(op_host_aclnn PRIVATE hc_pre_coefficients_def.cpp)
    add_ops_compile_options(
        OP_NAME HcPreCoefficients
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE hc_pre_coefficients ACLNNTYPE aclnn)
endif()
