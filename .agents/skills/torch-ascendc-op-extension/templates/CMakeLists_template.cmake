cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(<operator_name>_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)  # torch 要求 C++17，不可省略

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target 1: 可执行文件（原有，保持不变）
# ============================================================================
add_executable(<operator_name>_custom op_host/<operator_name>.asc)

target_link_libraries(<operator_name>_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(<operator_name>_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(<operator_name>_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
# ⬆ npu-arch 按实际芯片修改：A2/A3=dav-2201, A5=dav-3510

# ============================================================================
# Target 2: TORCH_LIBRARY 模块 (lib<operator_name>_ops.so)
#   Python 调用方式: torch.ops.load_library("build/lib<operator_name>_ops.so")
#                    torch.ops.npu.<operator_name>(x1, x2)
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

# 注意：xxx_kernel.asc 会被两个 Target 分别编译，这是预期行为。
# Target 1 通过 xxx.asc 的 #include 间接编译，Target 2 通过此源文件列表直接编译。
add_library(<operator_name>_ops SHARED
    op_kernel/<operator_name>_kernel.asc
    op_extension/<operator_name>_torch.cpp
    op_extension/register.cpp
)

target_include_directories(<operator_name>_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
)

target_link_libraries(<operator_name>_ops PRIVATE
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_link_directories(<operator_name>_ops PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    ${TORCH_NPU_LIBRARIES}
)

target_compile_options(<operator_name>_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

message(STATUS "TORCH_LIBRARY module 'lib<operator_name>_ops.so' will be built")
