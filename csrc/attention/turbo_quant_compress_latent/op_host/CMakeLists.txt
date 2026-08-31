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

set(TURBO_QUANT_COMPRESS_LATENT_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    turbo_quant_compress_latent_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME TurboQuantCompressLatent
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE turbo_quant_compress_latent ACLNNTYPE aclnn)
endif()
