set(SPLITQUANT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

ascendc_library(splitquant_vector STATIC
    op_kernel/scalar_round.cpp
    op_kernel/paged_silu.cpp
    op_kernel/compact_silu.cpp
    op_kernel/router_finish.cpp
    op_kernel/multi_quant.cpp
    op_kernel/multi_epilogue.cpp
    op_kernel/quant.cpp
    op_kernel/epilogue.cpp
    op_kernel/decode.cpp
    op_kernel/sparse_pack.cpp
    op_kernel/state_copy.cpp
    op_kernel/moe_vector.cpp
)
ascendc_library(splitquant_cube STATIC
    op_kernel/multi_cube.cpp
    op_kernel/cube.cpp
    op_kernel/rotation.cpp
    op_kernel/sparse_cube.cpp
    op_kernel/moe_cube.cpp
)
# MatmulImpl and TCubeTiling are provided by the public AscendC advanced API.
# The precompile target does not inherit the host target's include directories,
# so add the CANN device headers explicitly for all CANN layouts.
ascendc_include_directories(splitquant_cube PRIVATE
    "${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
)
ascendc_compile_definitions(splitquant_cube PRIVATE HAVE_WORKSPACE HAVE_TILING)
ascendc_library(splitquant_raw STATIC
    op_kernel/raw_cube.cpp
    op_kernel/raw_gate.cpp
)
ascendc_compile_definitions(splitquant_raw PRIVATE HAVE_WORKSPACE HAVE_TILING)
ascendc_compile_options(splitquant_raw PRIVATE -gline-tables-only --cce-auto-sync=off --cce-remove-auto-sync=false)
ascendc_library(splitquant_mix STATIC op_kernel/mixed_dense.cpp)
ascendc_compile_definitions(splitquant_mix PRIVATE HAVE_WORKSPACE HAVE_TILING)

add_library(vllm_ascend_splitquant_dense SHARED
    register.cpp
    op_host/paged_host.cpp
    op_host/compact_host.cpp
    op_host/router_host.cpp
    op_host/multi.cpp
    op_host/splitquant_dense_fused.cpp
    op_host/cube_host.cpp
    op_host/rotation_host.cpp
    op_host/decode_host.cpp
    op_host/sparse_host.cpp
    op_host/state_host.cpp
    op_host/moe_host.cpp
)
set_target_properties(vllm_ascend_splitquant_dense PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
target_link_directories(vllm_ascend_splitquant_dense PRIVATE
    "${TORCH_DIR}/lib"
    "${TORCH_NPU_PATH}/lib"
    "${ASCEND_HOME_PATH}/lib64"
)
target_link_libraries(vllm_ascend_splitquant_dense PRIVATE
    splitquant_vector
    splitquant_cube
    splitquant_raw
    splitquant_mix
    torch_npu
    ascendcl
    tiling_api
    nnopbase
    opapi
    register
    platform
    ascendalog
    dl
)
target_include_directories(vllm_ascend_splitquant_dense PRIVATE
    "${SPLITQUANT_ROOT}/include"
    "${SPLITQUANT_ROOT}/op_host"
    "${SPLITQUANT_ROOT}/op_kernel"
    "${TORCH_INCLUDE_DIRS}"
    "${TORCH_NPU_PATH}/include"
    "${TORCH_NPU_PATH}/include/third_party/acl/inc"
    "${TORCH_NPU_PATH}/include/third_party/hccl/inc"
    "${PYTHON_INCLUDE_PATH}"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/external"
    "${ASCEND_HOME_PATH}/include/experiment/platform"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
)
target_link_options(vllm_ascend_splitquant_dense PRIVATE
    "-Wl,-rpath,$ORIGIN:$ORIGIN/../../torch/lib"
)

install(TARGETS vllm_ascend_splitquant_dense
    LIBRARY DESTINATION "${VLLM_ASCEND_INSTALL_PATH}/lib"
)
