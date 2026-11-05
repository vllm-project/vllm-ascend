# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(metadef_FOUND)
  message(STATUS "metadef has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

set(METADEF_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
  ${TOP_DIR}/metadef/inc/external/            # compile with ci
)

set(METADEF_LIB_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}
)

find_path(METADEF_INC_DIR
  NAMES register/register.h
  PATHS ${METADEF_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(REGISTER_LIB_DIR
  NAME register
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(REGISTER_LIB_DIR)
  get_filename_component(REGISTER_LIB_DIR ${REGISTER_LIB_DIR} REALPATH)
  add_library(register SHARED IMPORTED)
  set_target_properties(register PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}
    IMPORTED_LOCATION ${REGISTER_LIB_DIR}
  )
  message(STATUS "Found register library:${REGISTER_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library register")
  endif()
endif()

find_library(OPP_REGISTER_LIB_DIR
  NAME opp_registry
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(OPP_REGISTER_LIB_DIR)
  get_filename_component(OPP_REGISTER_LIB_DIR ${OPP_REGISTER_LIB_DIR} REALPATH)
  add_library(opp_registry SHARED IMPORTED)
  set_target_properties(opp_registry PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}
          IMPORTED_LOCATION ${OPP_REGISTER_LIB_DIR}
          )
  message(STATUS "Found opp_registry library:${OPP_REGISTER_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library opp_registry")
  endif()
endif()

find_library(EXEGRAPH_LIB_DIR
  NAME exe_graph
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(EXEGRAPH_LIB_DIR)
  get_filename_component(EXEGRAPH_LIB_DIR ${EXEGRAPH_LIB_DIR} REALPATH)
  add_library(exe_graph SHARED IMPORTED)
  set_target_properties(exe_graph PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}/exe_graph
    IMPORTED_LOCATION ${EXEGRAPH_LIB_DIR}
  )
  message(STATUS "Found exe_graph library:${EXEGRAPH_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library exe_graph")
  endif()
endif()

find_library(GRAPH_BASE_LIB_DIR
        NAME graph_base
        PATHS ${METADEF_LIB_SEARCH_PATHS}
        PATH_SUFFIXES lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        )

if(GRAPH_BASE_LIB_DIR)
  get_filename_component(GRAPH_BASE_LIB_DIR ${GRAPH_BASE_LIB_DIR} REALPATH)
  add_library(graph_base SHARED IMPORTED)
  set_target_properties(graph_base PROPERTIES
          IMPORTED_LOCATION ${GRAPH_BASE_LIB_DIR}
          )
  message(STATUS "Found graph_base library:${GRAPH_BASE_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library graph_base")
  endif()
endif()

find_library(ERROR_MANAGER_LIB_DIR
        NAME error_manager
        PATHS ${METADEF_LIB_SEARCH_PATHS}
        PATH_SUFFIXES lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        )

if(ERROR_MANAGER_LIB_DIR)
  get_filename_component(ERROR_MANAGER_LIB_DIR ${ERROR_MANAGER_LIB_DIR} REALPATH)
  add_library(error_manager SHARED IMPORTED)
  set_target_properties(error_manager PROPERTIES
          IMPORTED_LOCATION ${ERROR_MANAGER_LIB_DIR}
          )
  message(STATUS "Found error_manager library:${ERROR_MANAGER_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library error_manager")
  endif()
endif()

find_library(METADEF_LIB_DIR
        NAME metadef
        PATHS ${METADEF_LIB_SEARCH_PATHS}
        PATH_SUFFIXES lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        )

if(METADEF_LIB_DIR)
  get_filename_component(METADEF_LIB_DIR ${METADEF_LIB_DIR} REALPATH)
  add_library(metadef SHARED IMPORTED)
  set_target_properties(metadef PROPERTIES
          IMPORTED_LOCATION ${METADEF_LIB_DIR}
          )
  message(STATUS "Found metadef library:${METADEF_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library metadef")
  endif()
endif()

find_library(REGISTER_STATIC_LIB_DIR
  NAME librt2_registry.a
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(REGISTER_STATIC_LIB_DIR)
  get_filename_component(REGISTER_STATIC_LIB_DIR ${REGISTER_STATIC_LIB_DIR} REALPATH)
  add_library(rt2_registry_static STATIC IMPORTED)
  set_target_properties(rt2_registry_static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METADEF_INC_DIR}
    IMPORTED_LOCATION ${REGISTER_STATIC_LIB_DIR}
  )
  message(STATUS "Found rt2_registry library:${REGISTER_STATIC_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library rt2_registry")
  endif()
endif()

find_package_handle_standard_args(metadef
      REQUIRED_VARS METADEF_INC_DIR)

get_filename_component(METADEF_INC_DIR ${METADEF_INC_DIR} REALPATH)
if(metadef_FOUND)
  set(METADEF_INCLUDE_DIRS
    ${METADEF_INC_DIR}/
    ${METADEF_INC_DIR}/exe_graph
  )

  if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    set(METADEF_INCLUDE_DIRS ${METADEF_INC_DIR}/../ ${METADEF_INCLUDE_DIRS})
  endif()
  message(STATUS "Found source metadef include dir:  ${METADEF_INCLUDE_DIRS}")
endif()



set(METADEF_LIB_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}
)

find_library(GRAPH_LIB_DIR
  NAME graph
  PATHS ${METADEF_LIB_SEARCH_PATHS}
  PATH_SUFFIXES lib64
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(GRAPH_LIB_DIR)
  get_filename_component(GRAPH_LIB_DIR ${GRAPH_LIB_DIR} REALPATH)
  add_library(graph SHARED IMPORTED)
  set_target_properties(graph PROPERTIES
    IMPORTED_LOCATION ${GRAPH_LIB_DIR}
  )
  message(STATUS "Found graph library:${GRAPH_LIB_DIR}")
else()
  if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "Cannot find library graph")
  endif()
endif()