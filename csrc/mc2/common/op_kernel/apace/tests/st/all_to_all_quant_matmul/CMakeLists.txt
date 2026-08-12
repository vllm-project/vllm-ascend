# ============================================================
# CMakeLists.txt — apace AllToAllQuantMatmul ST 构建配置
#
# 顶层 tests/st/CMakeLists.txt 已通过 ascend.cmake 设置:
#   ASCEND_DIR / SYSTEM_PREFIX / BISHENG / CMAKE_CXX_COMPILER
# 并通过 tensor_api.cmake 提供 apace_blaze_api INTERFACE target。
# ============================================================

# ascend.cmake 已设置: ASCEND_DIR / SYSTEM_PREFIX / BISHENG / CMAKE_CXX_COMPILER

set(APACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")          # mc2/common/op_kernel/apace
set(OP_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")    # mc2/common/op_kernel
# blaze / tensor_api 来自顶层 clone 的 ops-tensor（apace_blaze_api INTERFACE 已提供 include）
set(BLAZE_COMMON "${OPS_TENSOR_INCLUDE_DIR}")

# kernel 模式（-xasc）include 路径：apace kernel + Blaze + hcomm
set(KERNEL_INCLUDES
    ${OP_KERNEL_ROOT}                                                       # apace/ 头文件
    ${APACE_ROOT}
    ${APACE_ROOT}/kernel/all_to_all_quant_matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils                                    # utils.h
    ${BLAZE_COMMON}                                                         # blaze/（apace_blaze_api 也提供，此处冗余但无害）
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include                              # basic_api/ adv_api/
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api            # kernel_tiling/kernel_tiling.h
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include
)

# host 模式 include 路径：apace tiling + CANN tiling（platform_ascendc.h + soc_spec.h）
set(HOST_INCLUDES
    ${APACE_ROOT}
    ${APACE_ROOT}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling                       # platform_ascendc.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include                                  # hccl/ acl/
)

set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")
set(HOST_OPTS "-std=c++17" "-w" "-O2")

# hccl_fwk 提供 HcclChannelAcquire / HcclCommMemReg 等 HCCL 内部符号
# 使用 LINK_LIBRARIES 确保链接顺序：.o 在前，libhccl_fwk 在后
set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++)

add_executable(apace_a2a_qmm_udma_st
    src/main.cpp
)

# main.cpp: kernel 模式编译
set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

# libhccl_fwk.so 在 ${SYSTEM_PREFIX}/lib64，需显式指定链接目录
target_link_directories(apace_a2a_qmm_udma_st PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(apace_a2a_qmm_udma_st PRIVATE ${COMMON_LIBS} apace_blaze_api)

# libhccl_fwk.so 提供 HCCL 内部符号（HcclChannelAcquire 等），
# --no-as-needed 防止链接器丢弃，放在 OBJECT 之后确保符号解析
target_link_libraries(apace_a2a_qmm_udma_st PRIVATE "-Wl,--no-as-needed" hccl_fwk "-Wl,--as-needed")
