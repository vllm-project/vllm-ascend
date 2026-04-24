# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (securec_FOUND)
  message(STATUS "Package securec has been found.")
  return()
endif()

set(C_SEC_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
  ${TOP_DIR}/abl/libc_sec/include             # compile with ci
)

set(C_SEC_LIB_SEARCH_PATHS ${ASCEND_DIR}/${SYSTEM_PREFIX})

find_path(C_SEC_INCLUDE NAMES securec.h PATHS ${C_SEC_HEAD_SEARCH_PATHS})
if(WIN32)
  find_library(C_SEC_STATIC_LIBRARY NAMES libc_sec_static.lib)
  find_file(C_SEC_SHARED_DLL NAMES libc_sec.dll PATH_SUFFIXES lib)
  find_library(C_SEC_IMPLIB NAMES libc_sec.lib)
else()
  find_library(C_SEC_STATIC_LIBRARY NAMES libc_sec.a PATHS ${C_SEC_LIB_SEARCH_PATHS} PATH_SUFFIXES lib64)
  find_library(C_SEC_SHARED_LIBRARY NAMES libc_sec.so PATHS ${C_SEC_LIB_SEARCH_PATHS} PATH_SUFFIXES lib64)
endif()

include(FindPackageHandleStandardArgs)
if(WIN32)
  find_package_handle_standard_args(securec
    FOUND_VAR
      securec_FOUND
    REQUIRED_VARS
      C_SEC_INCLUDE
      C_SEC_STATIC_LIBRARY
      C_SEC_SHARED_DLL
      C_SEC_IMPLIB
    )
elseif("${ENABLE_SECUREC_SHARED}" STREQUAL "OFF")
  find_package_handle_standard_args(securec
    FOUND_VAR
      securec_FOUND
    REQUIRED_VARS
      C_SEC_INCLUDE
      C_SEC_STATIC_LIBRARY
    )
else()
  find_package_handle_standard_args(securec
    FOUND_VAR
      securec_FOUND
    REQUIRED_VARS
      C_SEC_INCLUDE
      C_SEC_STATIC_LIBRARY
      C_SEC_SHARED_LIBRARY
    )
endif()

if(securec_FOUND)
  set(C_SEC_INCLUDE_DIR ${C_SEC_INCLUDE})
  get_filename_component(C_SEC_LIBRARY_DIR ${C_SEC_STATIC_LIBRARY} DIRECTORY)

  add_library(c_sec_headers INTERFACE IMPORTED)
  target_include_directories(c_sec_headers INTERFACE ${C_SEC_INCLUDE})

  add_library(c_sec_static STATIC IMPORTED)
  set_target_properties(c_sec_static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${C_SEC_INCLUDE}"
    IMPORTED_LOCATION "${C_SEC_STATIC_LIBRARY}"
  )

  add_library(c_sec SHARED IMPORTED)
  if(WIN32)
    set_target_properties(c_sec PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${C_SEC_INCLUDE}"
      IMPORTED_IMPLIB "${C_SEC_IMPLIB}"
      IMPORTED_LOCATION "${C_SEC_SHARED_DLL}"
    )
  else()
    set_target_properties(c_sec PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${C_SEC_INCLUDE}"
      IMPORTED_LOCATION       "${C_SEC_SHARED_LIBRARY}"
    )
  endif()
endif()
