# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME MoeGroupedMatmulCustom
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
        moe_grouped_matmul_def.cpp
)

target_sources(opapi PRIVATE
        moe_grouped_matmul_l0.cpp
        aclnn_moe_grouped_matmul.cpp
)

target_sources(optiling PRIVATE
        moe_grouped_matmul_cpu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        moe_grouped_matmul_infershape.cpp
)

file(GLOB _GMM_Aclnn_header_weight_nz "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_moe_grouped_matmul_weight_nz.h")

install(FILES ${_GMM_Aclnn_header_weight_nz} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)