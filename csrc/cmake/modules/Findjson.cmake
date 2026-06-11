# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (json_FOUND)
  message(STATUS "Package json has been found.")
  return()
endif()

find_path(JSON_INCLUDE
  NAMES nlohmann/json.hpp
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
  FOUND_VAR
    json_FOUND
  REQUIRED_VARS
    JSON_INCLUDE
  )

if(json_FOUND)
  set(JSON_INCLUDE_DIR ${JSON_INCLUDE_DIR})

  add_library(json INTERFACE IMPORTED)
  set_target_properties(json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${JSON_INCLUDE_DIR}")
  target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)
endif()
