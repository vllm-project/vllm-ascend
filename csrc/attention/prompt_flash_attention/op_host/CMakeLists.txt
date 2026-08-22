# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(prompt_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    target_sources(op_host_aclnnInner PRIVATE
        prompt_flash_attention_def.cpp
    )
endif()

set(prompt_flash_attention_mc62_arch_directory arch38 CACHE INTERNAL "prompt_flash_attention mc62 arch")

add_ops_compile_options(
    OP_NAME PromptFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

add_modules_sources_with_soc(
    # 该变量用于判断op_api目录是否从op_host移出
    OP_API_INDEPENDENT ON
    # op_api新的地址
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE prompt_flash_attention ACLNNTYPE aclnn_inner)

if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub STATIC
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    target_link_libraries(softmax_tiling_ut_stub PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    )
endif()
