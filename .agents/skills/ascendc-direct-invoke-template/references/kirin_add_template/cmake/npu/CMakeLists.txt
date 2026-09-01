# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# npu
file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/*.cpp
)
set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CCE)

add_executable(${smoke_testcase}_npu
    ${SRC_FILES}
)

target_compile_options(${smoke_testcase}_npu PRIVATE
    -O2
    -std=c++17
)

target_link_directories(${smoke_testcase}_npu PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64
)

set_target_properties(${smoke_testcase}_npu PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_${ASCEND_RUN_MODE}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(${smoke_testcase}_sim ALL)
add_dependencies(${smoke_testcase}_sim ${smoke_testcase}_npu)
