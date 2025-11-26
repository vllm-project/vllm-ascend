# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME GroupedMatmulSwigluQuantWeightNzTensorList
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list_def.cpp
)

target_sources(opapi PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
        aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
        aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
        aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.cpp
    )
endif ()

target_sources(optiling PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        grouped_matmul_swiglu_quant_weight_nz_tensor_list_proto.cpp
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
