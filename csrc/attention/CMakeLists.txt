# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(OPTEST_NAME optest_${PKG_NAME})

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
        if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
            if (NOT ${SUB_DIR} IN_LIST ASCEND_OP_NAME)
                continue()
            endif ()
        endif ()
    endif ()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    else()
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/CMakeLists.txt")
            add_subdirectory(${SUB_DIR}/op_host)
        endif()
    endif()
endforeach()
# FIA v2 sink (vendor-isolated subtree): its carried omni-lineage headers must
# be visible to the aggregate tiling object target that compiles all ops'
# tiling sources. Appended LAST so repo-wide header resolution is unaffected
# (first match wins; these only catch headers otherwise unresolved).
if(TARGET ophost_transformer_tiling_obj)
    target_include_directories(ophost_transformer_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink_common/omni_common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink_common/attention_common
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink_common/attention_common/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink_common/attention_common/op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_infer_attention_score_v2_sink/op_api
    )
endif()
