# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SPARSE_MLA_COMMON_CHECKER_SRC_FILES
    ${CMAKE_CURRENT_LIST_DIR}/base_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/checker_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mask_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/metadata_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/seq_len_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sinks_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/softmax_lse_checker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sparse_compression_checker.cpp
)

function(add_sparse_mla_common_checker_sources target_name)
    get_property(common_checker_added TARGET ${target_name} PROPERTY SPARSE_MLA_COMMON_CHECKER_ADDED)
    if(NOT common_checker_added)
        target_sources(${target_name} PRIVATE ${SPARSE_MLA_COMMON_CHECKER_SRC_FILES})
        set_property(TARGET ${target_name} PROPERTY SPARSE_MLA_COMMON_CHECKER_ADDED TRUE)
    endif()
endfunction()
