# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED HINTS "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
project(qwen35_routed_prefetch LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import os,torch_npu; print(os.path.dirname(torch_npu.__file__))"
                OUTPUT_VARIABLE NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
                OUTPUT_VARIABLE EXT_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
add_library(_qwen35_routed_prefetch SHARED extension.asc)
target_include_directories(_qwen35_routed_prefetch PRIVATE "${NPU_ROOT}/include"
    "${NPU_ROOT}/include/third_party/acl/inc" ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_link_directories(_qwen35_routed_prefetch PRIVATE "${NPU_ROOT}/lib")
target_link_libraries(_qwen35_routed_prefetch PRIVATE ${TORCH_LIBRARIES} torch_npu Python3::Python)
target_compile_options(_qwen35_routed_prefetch PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201> $<$<COMPILE_LANGUAGE:ASC>:-fPIC>)
set(CMAKE_ASC_FLAGS "${CMAKE_ASC_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_compile_definitions(_qwen35_routed_prefetch PRIVATE TORCH_EXTENSION_NAME=_qwen35_routed_prefetch)
set_target_properties(_qwen35_routed_prefetch PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
