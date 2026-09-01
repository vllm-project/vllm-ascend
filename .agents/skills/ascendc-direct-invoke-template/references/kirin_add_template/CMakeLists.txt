# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE CXX)

add_subdirectory(cmake/npu)