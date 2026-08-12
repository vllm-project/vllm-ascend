# ============================================================
# tests/st/CMakeLists.txt — apace ST 顶层构建入口
# ============================================================
cmake_minimum_required(VERSION 3.16)

# ---- 查找 CANN 工具链 ----
if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        set(_ascend_processor "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()
    if(NOT _ascend_processor)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE _ascend_processor OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux" CACHE STRING "Ascend system prefix")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})
else()
    message(WARNING "Environment variable ASCEND_HOME_PATH is not set. Using default path.")
    if("$ENV{USER}" STREQUAL "root")
        if(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR /usr/local/Ascend/ascend-toolkit/latest)
        elseif(EXISTS /usr/local/Ascend/latest)
            set(ASCEND_DIR /usr/local/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Please set ASCEND_HOME_PATH.")
        endif()
    else()
        if(EXISTS $ENV{HOME}/Ascend/ascend-toolkit/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/ascend-toolkit/latest)
        elseif(EXISTS $ENV{HOME}/Ascend/latest)
            set(ASCEND_DIR $ENV{HOME}/Ascend/latest)
        else()
            message(FATAL_ERROR "Ascend toolkit not found. Please set ASCEND_HOME_PATH.")
        endif()
    endif()
endif()

message(STATUS "Using Ascend toolkit path: ${ASCEND_DIR}")
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")

if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${BISHENG}" CACHE FILEPATH "C compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "CXX compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_LINKER)
    set(CMAKE_LINKER "${BISHENG}" CACHE FILEPATH "Linker" FORCE)
endif()

# ---- ASC 编译器模块 ----
list(APPEND CMAKE_MODULE_PATH
    "${ASCEND_DIR}/${SYSTEM_PREFIX}/tikcpp/ascendc_kernel_cmake/asc_modules")
find_package(ASC REQUIRED)

project(apace_st LANGUAGES CXX ASC)

# ---- blaze / tensor_api（从 gitcode.com/cann/ops-tensor 拉取）----
# main.cpp 及 apace kernel 头文件依赖：
#   "blaze/gemm/..."           -> ops-tensor/include/
#   "tensor_api/tensor.h"      -> ops-tensor/include/tensor_api/include/
#   "include/tensor_api/..."   -> ops-tensor/include/tensor_api/
# tensor_api 子目录是 submodule（指向 asc-devkit），需要 --init --recursive。
# 复用主仓 ops-tensor（commit 锁定版，路径为 ${OPS_TRANSFORMER_ROOT}/third_party/ops-tensor）
get_filename_component(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." REALPATH)
if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${OPS_TRANSFORMER_ROOT}/third_party")
endif()
include(FetchContent)
include("${OPS_TRANSFORMER_ROOT}/cmake/third_party/ops-tensor.cmake")
set(OPS_TENSOR_ROOT "${OPTENSOR_SOURCE_PATH}")

find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git is required for ops-tensor submodule init")
endif()

# init submodule include/tensor_api（asc-devkit，提供 tensor.h / pointer.h）
if(NOT EXISTS "${OPS_TENSOR_ROOT}/include/tensor_api/include/tensor_api/tensor.h")
    message(STATUS "Initializing ops-tensor submodule include/tensor_api ...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive include/tensor_api
        WORKING_DIRECTORY "${OPS_TENSOR_ROOT}"
        RESULT_VARIABLE _sub_rc
        OUTPUT_VARIABLE _sub_out
        ERROR_VARIABLE _sub_err
    )
    if(NOT _sub_rc EQUAL 0)
        message(FATAL_ERROR "git submodule init failed:\n${_sub_err}")
    endif()
endif()

set(OPS_TENSOR_INCLUDE_DIR "${OPS_TENSOR_ROOT}/include")
if(NOT EXISTS "${OPS_TENSOR_INCLUDE_DIR}/blaze")
    message(FATAL_ERROR "blaze not found at: ${OPS_TENSOR_INCLUDE_DIR}/blaze")
endif()
if(NOT EXISTS "${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include/tensor_api/tensor.h")
    message(FATAL_ERROR "tensor_api/tensor.h not found at: ${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include/tensor_api")
endif()
message(STATUS "ops-tensor root: ${OPS_TENSOR_ROOT}")

# 三个 include 根，覆盖 main.cpp 与 apace kernel 头文件中所有 #include 形式：
#   blaze/...                -> ${OPS_TENSOR_INCLUDE_DIR}
#   tensor_api/tensor.h      -> ${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include
#   tensor_api/tensor/...    -> ${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include
#   include/tensor_api/...   -> ${OPS_TENSOR_INCLUDE_DIR}/tensor_api
add_library(apace_blaze_api INTERFACE)
target_include_directories(apace_blaze_api INTERFACE
    "${OPS_TENSOR_INCLUDE_DIR}"
    "${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include"
    "${OPS_TENSOR_INCLUDE_DIR}/tensor_api"
)

# ---- ST 用例 ----
add_subdirectory(all_to_all_quant_matmul)
add_subdirectory(all_gather_quant_matmul)
