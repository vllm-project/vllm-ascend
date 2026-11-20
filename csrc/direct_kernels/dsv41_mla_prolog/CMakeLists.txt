# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(dsv41_mla_prolog_direct LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/catlass" CACHE PATH "CATLASS source root")
if(NOT CATLASS_ROOT OR NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR "Set CATLASS_ROOT to a CATLASS checkout (>= v1.6.1)")
endif()
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "Source the CANN set_env.sh first or pass -DASCEND_HOME_PATH")
endif()

set_source_files_properties(dsv41_mla_prolog_direct.cpp PROPERTIES LANGUAGE ASC)
add_library(
    vllm_ascend_mla_prolog_direct SHARED
    dsv41_mla_prolog_direct.cpp)

target_include_directories(
    vllm_ascend_mla_prolog_direct PRIVATE
    ${CATLASS_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../attention/dsv41_mla_prolog/op_kernel
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include)
# The ASC compiler needs the standard library headers of the host toolchain.
foreach(host_include IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    target_compile_options(vllm_ascend_mla_prolog_direct PRIVATE "SHELL:-isystem ${host_include}")
endforeach()
target_compile_definitions(
    vllm_ascend_mla_prolog_direct PRIVATE
    CATLASS_ARCH=2201
    _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(
    vllm_ascend_mla_prolog_direct PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
    "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-record-overflow=true>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>")

target_link_directories(
    vllm_ascend_mla_prolog_direct PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib)
target_link_libraries(
    vllm_ascend_mla_prolog_direct PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager
    ascendcl tiling_api nnopbase platform dl ascend_hal)
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libascendalog.so)
    target_link_libraries(vllm_ascend_mla_prolog_direct PRIVATE ascendalog)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so)
    target_link_libraries(vllm_ascend_mla_prolog_direct PRIVATE unified_dlog)
endif()
