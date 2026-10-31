# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

file(GLOB_RECURSE OP_TILING_FILES "*.cpp")

add_library(optiling SHARED ${OP_TILING_FILES})

add_dependencies(optiling json)

target_compile_definitions(optiling PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    LOG_CPP
)

target_include_directories(optiling PRIVATE
    ${OP_TILING_INCLUDE}
)

target_link_libraries(optiling PRIVATE
    $<BUILD_INTERFACE:dlog_headers>
)