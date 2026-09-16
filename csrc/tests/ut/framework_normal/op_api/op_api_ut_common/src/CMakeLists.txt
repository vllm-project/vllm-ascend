# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)

file(GLOB_RECURSE op_api_ut_common_src CONFIGURE_DEPENDS "*.cpp")
add_library(op_api_ut_common SHARED ${op_api_ut_common_src})

target_compile_options(op_api_ut_common PRIVATE
        -fno-access-control
        )

target_include_directories(op_api_ut_common PRIVATE
        ${JSON_INCLUDE_DIR}
        ${HI_PYTHON_INC}
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/include/aclnn
        ${ASCEND_DIR}/pkg_inc
        )

target_link_libraries(op_api_ut_common PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        op_api_ut_common_inc
        dl
        pthread
        json
        gtest
        )
