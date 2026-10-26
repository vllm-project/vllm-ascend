cmake_minimum_required(VERSION 3.20)
# This direct ASC extension is built separately from the aclnn vendor package.
if(PROJECT_NAME STREQUAL "cann_ops-transformer")
    return()
endif()
find_package(ASC REQUIRED)
project(compressor_serial LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(Torch_DIR "${Python3_SITELIB}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(TORCH_NPU_PATH "${Python3_SITELIB}/torch_npu")
add_library(compressor_serial_ops SHARED op_host/compressor_serial.cpp
    op_kernel/compressor_serial.asc)
target_include_directories(compressor_serial_ops PRIVATE ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include $ENV{ASCEND_HOME_PATH}/aarch64-linux/include)
target_link_directories(compressor_serial_ops PRIVATE ${TORCH_NPU_PATH}/lib $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64)
target_link_libraries(compressor_serial_ops PRIVATE ${TORCH_LIBRARIES} torch_npu ascendcl ascendc_runtime)
target_compile_options(compressor_serial_ops PRIVATE ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510;-O3>)
install(TARGETS compressor_serial_ops LIBRARY DESTINATION vllm_ascend)
