message(STATUS "TILING SINK TASK BEGIN")
message(STATUS "TARGET: ${TARGET}")
message(STATUS "OPTION: ${OPTION}")
message(STATUS "SRC: ${SRC}")
message(STATUS "VENDOR: ${VENDOR_NAME}")

set(CMAKE_CXX_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
set(CMAKE_C_COMPILER ${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc)

string(REPLACE " " ";" SRC "${SRC}")
add_library(${TARGET} ${OPTION}
    ${SRC}
)
target_compile_definitions(${TARGET} PRIVATE
    DEVICE_OP_TILING_LIB
    _FORTIFY_SOURCE=2
    google=ascend_private
)
target_include_directories(${TARGET} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_compile_options(${TARGET} PRIVATE
    -fPIC
    -fstack-protector-strong
    -fstack-protector-all
    -O2
    -std=c++11
    -fvisibility-inlines-hidden
    -fvisibility=hidden
)
target_link_libraries(${TARGET} PRIVATE
    -Wl,--whole-archive
    device_register
    c_sec
    mmpa
    tiling_api
    platform_static
    ascend_protobuf
    exe_meta_device
    -Wl,--no-whole-archive
)
target_link_directories(${TARGET} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/lib64
)
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME cust_opmaster
)
