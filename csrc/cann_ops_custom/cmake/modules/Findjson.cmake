# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (json_FOUND)
    message(STATUS "Package json has been found.")
    return()
endif()

set(JSON_INCLUDE_SEARCH_PATHS
    ${CMAKE_SOURCE_DIR}/third_party/json/include
)

if(DEFINED ENV{ASCEND_3RD_LIB_PATH})
    list(APPEND JSON_INCLUDE_SEARCH_PATHS "$ENV{ASCEND_3RD_LIB_PATH}/json/include")
endif()

find_path(JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS ${JSON_INCLUDE_SEARCH_PATHS}
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
    FOUND_VAR
        json_FOUND
    REQUIRED_VARS
        JSON_INCLUDE_DIR
    )

if(json_FOUND)
    add_library(json INTERFACE IMPORTED)
    set_target_properties(json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE_DIR}")
    target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)
endif()