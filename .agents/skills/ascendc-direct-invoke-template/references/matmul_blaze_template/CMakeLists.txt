# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(COMMAND uname -m
            OUTPUT_VARIABLE _ascend_processor
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    if("$ENV{USER}" STREQUAL "root")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Set ASCEND_HOME_PATH.")
        endif()
    else()
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()
message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")

set(CMAKE_PREFIX_PATH ${ASCEND_DIR}/)
find_package(ASC REQUIRED)

project(matmul_blaze LANGUAGES ASC CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/asc
    ${ASCEND_DIR}/asc/include
    ${ASCEND_DIR}/compiler/asc/include
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)

set(MATMUL_ASC_COMPILE_OPTS
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    $<$<COMPILE_LANGUAGE:ASC>:-w>
    $<$<COMPILE_LANGUAGE:ASC>:-O3>
)

set(MATMUL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/tensor_api
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/tensor_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# ============================================================================
# Target 1: Executable (direct invoke)
# ============================================================================

add_executable(matmul_blaze test_matmul_blaze.cpp op_kernel/matmul_blaze_kernel.cpp)
set_source_files_properties(test_matmul_blaze.cpp op_kernel/matmul_blaze_kernel.cpp PROPERTIES LANGUAGE ASC)
target_include_directories(matmul_blaze PRIVATE ${MATMUL_INCLUDE_DIRS})
target_include_directories(matmul_blaze PRIVATE ${ASCEND_INCLUDE_DIRS})
target_compile_options(matmul_blaze PRIVATE ${MATMUL_ASC_COMPILE_OPTS})
target_link_directories(matmul_blaze PRIVATE ${ASCEND_DIR}/lib64)
target_link_libraries(matmul_blaze PRIVATE m dl platform tiling_api register)

# ============================================================================
# Target 2: TORCH_LIBRARY module (libmatmul_blaze_ops.so)
# ============================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(matmul_blaze_ops SHARED
    op_kernel/matmul_blaze_kernel.cpp
    op_extension/matmul_blaze_torch.cpp
    op_extension/register.cpp
)
set_source_files_properties(op_kernel/matmul_blaze_kernel.cpp PROPERTIES LANGUAGE ASC)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
)

target_include_directories(matmul_blaze_ops PRIVATE
    ${MATMUL_INCLUDE_DIRS}
    ${ASCEND_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
)
target_compile_options(matmul_blaze_ops PRIVATE
    ${MATMUL_ASC_COMPILE_OPTS}
    ${TORCH_CXX_FLAGS}
)
target_link_directories(matmul_blaze_ops PRIVATE
    ${ASCEND_DIR}/lib64
    ${TORCH_NPU_LIBRARIES}
)
target_link_libraries(matmul_blaze_ops PRIVATE
    torch_npu
    m dl platform tiling_api register
    ascendcl ascendc_runtime graph_base unified_dlog
)

message(STATUS "TORCH_LIBRARY module 'libmatmul_blaze_ops.so' will be built")
