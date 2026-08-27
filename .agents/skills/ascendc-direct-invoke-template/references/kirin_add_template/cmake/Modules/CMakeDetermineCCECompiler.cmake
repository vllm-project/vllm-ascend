# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

find_program(CMAKE_CCE_COMPILER NAMES "ccec" PATHS "$ENV{PATH}" DOC "CCE Compiler")

mark_as_advanced(CMAKE_CCE_COMPILER)

message(STATUS "CMAKE_CCE_COMPILER: " ${CMAKE_CCE_COMPILER})
set(CMAKE_CCE_SOURCE_FILE_EXTENSIONS cce;cpp)
set(CMAKE_CCE_COMPILER_ENV_VAR "CCE")
message(STATUS "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})

# configure all variables set in this file
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCCECompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeCCECompiler.cmake
    @ONLY
)

message(STATUS "ASCEND_PRODUCT_TYPE:\n" "  ${ASCEND_PRODUCT_TYPE}")
message(STATUS "ASCEND_CORE_TYPE:\n" "  ${ASCEND_CORE_TYPE}")
message(STATUS "ASCEND_INSTALL_PATH:\n" "  ${ASCEND_INSTALL_PATH}")

if(DEFINED ASCEND_INSTALL_PATH)
    set(_CMAKE_ASCEND_INSTALL_PATH ${ASCEND_INSTALL_PATH})
else()
    message(FATAL_ERROR
        "no, installation path found, should passing -DASCEND_INSTALL_PATH=<PATH_TO_ASCEND_INSTALLATION> in cmake"
    )
    set(_CMAKE_ASCEND_INSTALL_PATH)
endif()


if(DEFINED ASCEND_PRODUCT_TYPE)
    set(_CMAKE_CCE_COMMON_COMPILE_OPTIONS "--cce-auto-sync -mllvm -api-deps-filter")
    if(ASCEND_PRODUCT_TYPE STREQUAL "")
        message(FATAL_ERROR "ASCEND_PRODUCT_TYPE must be non-empty if set.")
    elseif(ASCEND_PRODUCT_TYPE AND NOT ASCEND_PRODUCT_TYPE MATCHES "^Kirin[a-zA-Z0-9]")
        message(FATAL_ERROR
            "ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}\n"
            "is not one of the following: KirinX90, Kirin9030"
        )
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "KirinX90")
        if (ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-l300")
        elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-l300-eff")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS
            "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform")
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "Kirin9030")
        if (ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-l311")
        elseif(ASCEND_CORE_TYPE STREQUAL "VectorCore")
            set(_CMAKE_COMPILE_AS_CCE_FLAG "--cce-aicore-arch=dav-l311-eff")
        endif()
        set(_CMAKE_CCE_COMPILE_OPTIONS
            "-mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform")
    endif()
endif()

set(_CMAKE_CCE_HOST_IMPLICIT_LINK_DIRECTORIES
    ${_CMAKE_ASCEND_INSTALL_PATH}/runtime/lib64
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/simulator/${ASCEND_PRODUCT_TYPE}/lib
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${ASCEND_PRODUCT_TYPE}
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/platform/${ASCEND_PRODUCT_TYPE}/simulator
)

# link library
set(_CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES stdc++)
if(ASCEND_RUN_MODE STREQUAL "npu")
    list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES runtime)
elseif(ASCEND_RUN_MODE STREQUAL "sim")
    list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_DIRECTORIES )
    list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES runtime_camodel)
elseif(ASCEND_RUN_MODE STREQUAL "cpu")
    message(STATUS "RUN_MODE is cpu")
else()
    message(FATAL_ERROR
        "ASCEND_RUN_MODE: ${ASCEND_RUN_MODE}\n"
        "ASCEND_RUN_MODE must be one of the following: cpu, npu or sim"
    )
endif()
list(APPEND _CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES ascendcl)

set(__IMPLICIT_LINKS)
foreach(dir ${_CMAKE_CCE_HOST_IMPLICIT_LINK_DIRECTORIES})
  string(APPEND __IMPLICIT_LINKS " -L\"${dir}\"")
endforeach()
foreach(lib ${_CMAKE_CCE_HOST_IMPLICIT_LINK_LIBRARIES})
  if(${lib} MATCHES "/")
    string(APPEND __IMPLICIT_LINKS " \"${lib}\"")
  else()
    string(APPEND __IMPLICIT_LINKS " -l${lib}")
  endif()
endforeach()

set(_CMAKE_CCE_HOST_IMPLICIT_INCLUDE_DIRECTORIES
    ${_CMAKE_ASCEND_INSTALL_PATH}/acllib/include
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/impl
    ${_CMAKE_ASCEND_INSTALL_PATH}/compiler/tikcpp/tikcfw/interface
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikcpp/tikcfw
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikcpp/tikcfw/impl
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikcpp/tikcfw/interface
    ${_CMAKE_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/include
    ${_CMAKE_ASCEND_INSTALL_PATH}/x86_64-linux/include
    ${_CMAKE_ASCEND_INSTALL_PATH}/x86_64-linux/asc/include
)
set(__IMPLICIT_INCLUDES)
foreach(inc ${_CMAKE_CCE_HOST_IMPLICIT_INCLUDE_DIRECTORIES})
  string(APPEND __IMPLICIT_INCLUDES " -I\"${inc}\"")
endforeach()