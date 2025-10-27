# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(JOSN_INSTALL_PATH ${CMAKE_SOURCE_DIR}/third_party/json)
set(JSON_HEADER_PATH ${JOSN_INSTALL_PATH}/include/nlohmann/json.hpp)

if(EXISTS ${JSON_HEADER_PATH})
    message("Json header has been existed.")
    set(JSON_INCLUDE_DIR ${JOSN_INSTALL_PATH}/include)
    return()
endif()

include(FetchContent)

message(STATUS "Start nlohmann_json download...")
# 设置下载超时（单位：秒）
set(EP_DOWNLOAD_TIMEOUT 300)
# 设置重试次数
set(EP_RETRY_COUNT 3)
set(EP_RETRY_DELAY 3)

FetchContent_Declare(
    nlohmann_json
    URL https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip
    URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d
    SOURCE_DIR ${JOSN_INSTALL_PATH}
)

FetchContent_MakeAvailable(nlohmann_json)
message(STATUS "Json availabled success.")

set(JSON_INCLUDE_DIR ${JOSN_INSTALL_PATH}/include)
