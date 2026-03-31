# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_ops_compile_options(
    OP_NAME CausalConv1dFnV310
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)
target_sources(op_host_aclnn PRIVATE causal_conv1d_v310_def.cpp)
target_sources(optiling PRIVATE causal_conv1d_v310_tiling.cpp)
target_sources(opsproto PRIVATE causal_conv1d_v310_infershape.cpp)
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB _Causal_Conv1d_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_causal_conv1d_v310.h")

install(FILES ${_Causal_Conv1d_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
