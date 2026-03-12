# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_ops_compile_options(
        OP_NAME CausalConv1dUpdate
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)


target_sources(op_host_aclnn PRIVATE
        causal_conv1d_update_def.cpp    
)

# Tiling
target_sources(optiling PRIVATE
        causal_conv1d_update_tiling.cpp
        tiling_util.cpp
)

target_sources(opsproto PRIVATE
        causal_conv1d_infershape.cpp           
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
    

