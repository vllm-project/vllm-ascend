cmake_minimum_required(VERSION 3.16)
project(vllm-ascend LANGUAGE CXX)

include(CheckCXXcompilerFlag)
check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)


include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Torch REQUIRED)


set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/lastest")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "#{CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/pos_encoding_kernels.cpp)

ascendc_library(vllm_ascend_kernels STATIC
    add_custom.cpp
)

execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

file(GLOB TORCH_BINDINGS
${CMAKE_CURRENT_SOURCE_DIR}/torch_binding.cpp)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
)

set(LIBS
  vllm_ascend_kernels
  ascendcl
  platform)
# include_directories(
#   ${TORCH_INCLUDE_DIRS}
#   ${TORCH_NPU_INCLUDE_DIRS}
#   ${ASCEND_HOME_PATH}/include
#   ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
# )

# if("${RUN_MODE}" STREQUAL "cpu")
#   include(cmake/cpu_lib.cmake)
# elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
#   include(cmake/npu_lib.cmake)
# endif()

# set_target_properties()

define_gpu_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE CXX
    INCLUDE_DIRECTORIES ${INCLUDES}
    SOURCES ${VLLM_EXT_SRC}
    LIBRARIES ${LIBS}
    COMPILE_FLAGS ${CXX_COMPILE_FLAGS}
    USE_SABI 3
    WITH_SOABI
)

message(STATUS "Enabling C extension.")



