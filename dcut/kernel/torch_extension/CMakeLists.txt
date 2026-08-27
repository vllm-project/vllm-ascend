cmake_minimum_required(VERSION 3.16)
project(dcut_torch_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED REPO_ROOT)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT DEFINED TORCH_NPU_PATH)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).resolve().parent)"
        OUTPUT_VARIABLE TORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()

if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
    else()
        set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

add_library(dcut_torch_ops SHARED
    dcut_torch_binding.cpp
    /vllm-workspace/vllm-ascend/csrc/aclnn_torch_adapter/NPUBridge.cpp
    /vllm-workspace/vllm-ascend/csrc/aclnn_torch_adapter/NPUStorageImpl.cpp
)
set_target_properties(dcut_torch_ops PROPERTIES PREFIX "" OUTPUT_NAME "dcut_torch_ops")
target_include_directories(dcut_torch_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include"
    "${ASCEND_HOME_PATH}/include"
    "/vllm-workspace/vllm-ascend/csrc"
)
target_link_directories(dcut_torch_ops PRIVATE
    ${TORCH_LIBRARY_DIRS}
    "${TORCH_NPU_PATH}/lib"
    "${ASCEND_HOME_PATH}/lib64"
)
target_link_libraries(dcut_torch_ops PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    tiling_api
    register
    platform
    ascendalog
    dl
    opapi
)
