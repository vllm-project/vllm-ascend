# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME GatherPaKvCacheVllm
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        gather_pa_kv_cache_vllm_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_gather_pa_kv_cache_vllm.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        aclnn_gather_pa_kv_cache_vllm.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        aclnn_gather_pa_kv_cache_vllm.cpp
    )
endif ()

target_sources(optiling PRIVATE
        gather_pa_kv_cache_vllm_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/utils/inc/kernel
)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_gather_pa_kv_cache_vllm.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)

target_sources(opsproto PRIVATE
        gather_pa_kv_cache_vllm_infershape.cpp
)
