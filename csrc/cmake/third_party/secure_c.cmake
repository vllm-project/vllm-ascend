# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(_secure_c_url "")
if(CANN_PKG_SERVER)
  set(_secure_c_url "${CANN_PKG_SERVER}/libs/securec/v1.1.10.tar.gz")
endif()
include(ExternalProject)
ExternalProject_Add(secure_c
  URL               ${_secure_c_url}
                    https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.tar.gz
  URL_MD5           ae4865cec1bfb52f7dca03f5c05ac98a
  DOWNLOAD_DIR      download/secure_c
  PREFIX            third_party
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(secure_c SOURCE_DIR)
ExternalProject_Get_Property(secure_c BINARY_DIR)

set(SEC_C_SRCS
  ${SOURCE_DIR}/src/fscanf_s.c
  ${SOURCE_DIR}/src/fwscanf_s.c
  ${SOURCE_DIR}/src/gets_s.c
  ${SOURCE_DIR}/src/input.inl
  ${SOURCE_DIR}/src/memcpy_s.c
  ${SOURCE_DIR}/src/memmove_s.c
  ${SOURCE_DIR}/src/memset_s.c
  ${SOURCE_DIR}/src/output.inl
  ${SOURCE_DIR}/src/scanf_s.c
  ${SOURCE_DIR}/src/secinput.h
  ${SOURCE_DIR}/src/securecutil.c
  ${SOURCE_DIR}/src/securecutil.h
  ${SOURCE_DIR}/src/secureinput_a.c
  ${SOURCE_DIR}/src/secureinput_w.c
  ${SOURCE_DIR}/src/secureprintoutput_a.c
  ${SOURCE_DIR}/src/secureprintoutput.h
  ${SOURCE_DIR}/src/secureprintoutput_w.c
  ${SOURCE_DIR}/src/snprintf_s.c
  ${SOURCE_DIR}/src/sprintf_s.c
  ${SOURCE_DIR}/src/sscanf_s.c
  ${SOURCE_DIR}/src/strcat_s.c
  ${SOURCE_DIR}/src/strcpy_s.c
  ${SOURCE_DIR}/src/strncat_s.c
  ${SOURCE_DIR}/src/strncpy_s.c
  ${SOURCE_DIR}/src/strtok_s.c
  ${SOURCE_DIR}/src/swprintf_s.c
  ${SOURCE_DIR}/src/swscanf_s.c
  ${SOURCE_DIR}/src/vfscanf_s.c
  ${SOURCE_DIR}/src/vfwscanf_s.c
  ${SOURCE_DIR}/src/vscanf_s.c
  ${SOURCE_DIR}/src/vsnprintf_s.c
  ${SOURCE_DIR}/src/vsprintf_s.c
  ${SOURCE_DIR}/src/vsscanf_s.c
  ${SOURCE_DIR}/src/vswprintf_s.c
  ${SOURCE_DIR}/src/vswscanf_s.c
  ${SOURCE_DIR}/src/vwscanf_s.c
  ${SOURCE_DIR}/src/wcscat_s.c
  ${SOURCE_DIR}/src/wcscpy_s.c
  ${SOURCE_DIR}/src/wcsncat_s.c
  ${SOURCE_DIR}/src/wcsncpy_s.c
  ${SOURCE_DIR}/src/wcstok_s.c
  ${SOURCE_DIR}/src/wmemcpy_s.c
  ${SOURCE_DIR}/src/wmemmove_s.c
  ${SOURCE_DIR}/src/wscanf_s.c
)

add_library(c_sec SHARED ${SEC_C_SRCS})

add_dependencies(c_sec secure_c)

set_source_files_properties(
  ${SEC_C_SRCS}
  PROPERTIES
  GENERATED TRUE
)

target_include_directories(c_sec
  PUBLIC
  "${SOURCE_DIR}/include"
)

set(C_SEC_INCLUDE ${SOURCE_DIR}/include)

target_compile_options(c_sec
  PRIVATE
  -fstack-protector-strong -fPIC -Wall -D_FORTIFY_SOURCE=2 -O2
)

set_target_properties(c_sec
  PROPERTIES
  LINKER_LANGUAGE          C
  LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR}
)

cann_install(
  TARGET      c_sec
  FILES       $<TARGET_FILE:c_sec>
  DESTINATION "${CANN_ROOT}/lib"
)
