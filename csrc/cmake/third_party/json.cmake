# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(json_FOUND)
    return()
endif()

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

if(NOT CANN_3RD_PKG_PATH)
  set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/third_party/pkg)
endif()

set(JSON_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/pkg)
set(JSON_INSTALL_PATH ${CANN_3RD_LIB_PATH}/json)

find_path(JSON_INCLUDE
        NAMES nlohmann/json.hpp
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${JSON_INSTALL_PATH}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
        FOUND_VAR
        json_FOUND
        REQUIRED_VARS
        JSON_INCLUDE
        )

if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message("json found in ${JSON_INSTALL_PATH}, and not force rebuild cann third_party")
    set(JSON_INCLUDE_DIR ${JSON_INSTALL_PATH}/include)
    add_library(json INTERFACE IMPORTED)
else()
    set(REQ_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip")

    include(ExternalProject)
    ExternalProject_Add(third_party_json
            URL ${REQ_URL}
            TLS_VERIFY OFF
            DOWNLOAD_DIR ${JSON_DOWNLOAD_PATH}
            DOWNLOAD_NO_EXTRACT TRUE
            SOURCE_DIR ${JSON_INSTALL_PATH}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND
                ${CMAKE_COMMAND} -E make_directory ${JSON_INSTALL_PATH} &&
                ${CMAKE_COMMAND} -E chdir ${JSON_INSTALL_PATH} ${CMAKE_COMMAND} -E tar xf "${JSON_DOWNLOAD_PATH}/include.zip" --format=zip
            UPDATE_COMMAND ""
    )

    ExternalProject_Get_Property(third_party_json SOURCE_DIR)
    ExternalProject_Get_Property(third_party_json BINARY_DIR)
    set(JSON_INCLUDE_DIR ${SOURCE_DIR}/include)
    add_library(json INTERFACE)
    target_include_directories(json INTERFACE ${JSON_INCLUDE_DIR})
    add_dependencies(json third_party_json)
endif()
