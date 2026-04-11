# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(nnopbase_FOUND)
  message(STATUS "nnopbase has been found")
  return()
endif()

set(nnopbase_FOUND ON)
include(FindPackageHandleStandardArgs)

set(NNOPBASE_ACLNN_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
  ${TOP_DIR}/ace/npuruntime/inc/external/            # compile with ci
)

set(NNOPBASE_OPDEV_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/aclnn
  ${TOP_DIR}/ace/npuruntime/inc/nnopbase/            # compile with ci
)

set(NNOPBASE_LIB_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}
)

find_path(NNOPBASE_ACLNN_INC_DIR
  NAMES aclnn/aclnn_base.h
  PATHS ${NNOPBASE_ACLNN_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if(NOT NNOPBASE_ACLNN_INC_DIR)
  set(nnopbase_FOUND OFF)
endif()

find_path(NNOPBASE_OPDEV_INC_DIR
  NAMES opdev/op_errno.h
  PATHS ${NNOPBASE_OPDEV_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)
if(NOT NNOPBASE_OPDEV_INC_DIR)
  set(nnopbase_FOUND OFF)
endif()

find_library(NNOPBASE_LIB_DIR
  NAME nnopbase 
  PATHS ${NNOPBASE_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

get_filename_component(NNOPBASE_ACLNN_INC_DIR ${NNOPBASE_ACLNN_INC_DIR} REALPATH)
get_filename_component(NNOPBASE_OPDEV_INC_DIR ${NNOPBASE_OPDEV_INC_DIR} REALPATH)

if(NNOPBASE_LIB_DIR)
  get_filename_component(NNOPBASE_LIB_DIR ${NNOPBASE_LIB_DIR} REALPATH)
  add_library(nnopbase SHARED IMPORTED)
  set_target_properties(nnopbase PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${NNOPBASE_OPDEV_INC_DIR}
    IMPORTED_LOCATION ${NNOPBASE_LIB_DIR}
  )
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library nnopbase")
  endif()
endif()

if(nnopbase_FOUND)
  set(NNOPBASE_INCLUDE_DIRS
    ${NNOPBASE_ACLNN_INC_DIR}
    ${NNOPBASE_OPDEV_INC_DIR}
  )
  message(STATUS "Found aclnn include dir:  ${NNOPBASE_ACLNN_INC_DIR}")
  message(STATUS "Found opdev include dir:  ${NNOPBASE_OPDEV_INC_DIR}")
endif()