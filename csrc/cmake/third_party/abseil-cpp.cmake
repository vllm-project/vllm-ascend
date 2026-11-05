# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include(ExternalProject)
set(ABSEIL_VERSION_PKG abseil-cpp-20230802.1.tar.gz)

unset(abseil-cpp_FOUND CACHE)
unset(ABSL_SOURCE_DIR CACHE)

find_path(ABSL_SOURCE_DIR
        NAMES absl/log/absl_log.h
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(abseil-cpp
        FOUND_VAR
        abseil-cpp_FOUND
        REQUIRED_VARS
        ABSL_SOURCE_DIR)

set(ABSEIL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/abseil-cpp)
if(abseil-cpp_FOUND)
  message(STATUS "Found abseil-cpp in ${CANN_3RD_LIB_PATH}/abseil-cpp")
else()
  # 初始化可选参数列表
  if(EXISTS "${CANN_3RD_LIB_PATH}/abseil-cpp/${ABSEIL_VERSION_PKG}")
      set(REQ_URL "file://${CANN_3RD_LIB_PATH}/abseil-cpp/${ABSEIL_VERSION_PKG}")
      message(STATUS "[ThirdPartyLib][abseil-cpp] found in ${REQ_URL}.")
  elseif(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${ABSEIL_VERSION_PKG}")
      set(REQ_URL "file://${CANN_3RD_LIB_PATH}/pkg/${ABSEIL_VERSION_PKG}")
      message(STATUS "[ThirdPartyLib][abseil-cpp] found in ${REQ_URL}.")
  else()
    set(REQ_URL "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz")
    message(STATUS "[ThirdPartyLib][abseil-cpp] ${REQ_URL} not found, need download.")
  endif()

  ExternalProject_Add(abseil_build_transformer
                      URL ${REQ_URL}
                      DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}/pkg
                      PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/build/modules/patch/protobuf-hide_absl_symbols.patch
                      SOURCE_DIR ${ABSEIL_SOURCE_DIR}
                      CONFIGURE_COMMAND ""
                      BUILD_COMMAND ""
                      INSTALL_COMMAND ""
                      EXCLUDE_FROM_ALL TRUE 
  )

  ExternalProject_Get_Property(abseil_build_transformer SOURCE_DIR)
  set(ABSL_SOURCE_DIR ${SOURCE_DIR})
endif()