# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME PagedSelectAttention
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnInner PRIVATE
        paged_select_attention_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opapi PRIVATE
        op_api/aclnn_paged_select_attention.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        op_api/aclnn_paged_select_attention.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        op_api/aclnn_paged_select_attention.cpp
    )
endif ()

target_sources(optiling PRIVATE
        paged_select_attention_tiling.cpp
        paged_select_attention_tiling_register.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        paged_select_attention_infershape.cpp
)

file(GLOB PAGED_SELECT_ATTENTION_ACLNN_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_paged_select_attention.h")

install(FILES ${PAGED_SELECT_ATTENTION_ACLNN_HEADER}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
