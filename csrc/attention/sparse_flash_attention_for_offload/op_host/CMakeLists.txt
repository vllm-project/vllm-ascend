# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME SparseFlashAttentionForOffload
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnInner PRIVATE
        sparse_flash_attention_for_offload_def.cpp
)

target_sources(optiling PRIVATE
        sparse_flash_attention_for_offload_tiling.cpp
)

target_sources(opsproto PRIVATE
        sparse_flash_attention_for_offload_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opapi PRIVATE
        op_api/aclnn_sparse_flash_attention_for_offload.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
)

install(FILES op_api/aclnn_sparse_flash_attention_for_offload.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
