cmake_minimum_required(VERSION 3.19)
project(flash_mla_a3 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE npu)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "A3 CANN SoC name, e.g. Ascend910_9392")
string(TOLOWER "${SOC_VERSION}" SOC_VERSION)
if(NOT SOC_VERSION MATCHES "^ascend910_93[0-9][0-9]$")
    message(FATAL_ERROR "Pass -DSOC_VERSION=<A3 SoC name>, for example Ascend910_9392")
endif()

find_file(ASCENDC_CMAKE ascendc.cmake
    HINTS "${ASCEND_CANN_PACKAGE_PATH}" "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux"
    PATH_SUFFIXES tools/tikcpp/ascendc_kernel_cmake compiler/tikcpp/ascendc_kernel_cmake
                  ascendc_devkit/tikcpp/samples/cmake
    NO_DEFAULT_PATH REQUIRED)
include("${ASCENDC_CMAKE}")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import importlib.util; print(next(iter(importlib.util.find_spec('torch_npu').submodule_search_locations)))"
    OUTPUT_VARIABLE torch_npu_dir OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_library(torch_npu_lib torch_npu PATHS "${torch_npu_dir}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ascendcl_lib ascendcl PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" NO_DEFAULT_PATH REQUIRED)

ascendc_library(flash_mla_a3_kernels STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/native/flash_mla_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/flash_decode_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native/metadata_kernel.cpp")

separate_arguments(torch_compile_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
function(add_mla_torch_library host_target kernel_target host_source)
    add_library(${host_target} SHARED ${host_source})
    add_dependencies(${host_target} ${kernel_target})
    target_include_directories(${host_target} PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel_target}"
        "${ASCEND_CANN_PACKAGE_PATH}/include"
        "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
        "${torch_npu_dir}/include" "${torch_npu_dir}/include/third_party/acl/inc"
        "${torch_npu_dir}/include/third_party/hccl/inc"
        ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
    target_compile_options(${host_target} PRIVATE ${torch_compile_flags})
    target_link_libraries(${host_target} PRIVATE ${kernel_target}
        "${torch_npu_lib}" "${ascendcl_lib}" ${TORCH_LIBRARIES})
    set_target_properties(${host_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endfunction()

add_mla_torch_library(flash_mla_a3_ops flash_mla_a3_kernels native/bindings.cpp)

ascendc_library(flash_mla_a3_grouped_kernels STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/native/grouped/grouped_kernel.cpp")
add_mla_torch_library(flash_mla_a3_grouped flash_mla_a3_grouped_kernels native/grouped/grouped_bindings.cpp)

add_library(flash_mla_a3_catlass_ops SHARED native/catlass_bindings.cpp)
target_include_directories(flash_mla_a3_catlass_ops PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include" "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${torch_npu_dir}/include" "${torch_npu_dir}/include/third_party/acl/inc"
    "${torch_npu_dir}/include/third_party/hccl/inc" ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_options(flash_mla_a3_catlass_ops PRIVATE ${torch_compile_flags})
target_link_libraries(flash_mla_a3_catlass_ops PRIVATE "${torch_npu_lib}" "${ascendcl_lib}" ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
