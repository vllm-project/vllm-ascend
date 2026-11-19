# -----------------------------------------------------------------------------------------------------------
#
# k2q_csr：父目录挂载独立命名阶段算子（Meta/Hist/RowPrefix/TilePrefix/Scatter）
# -----------------------------------------------------------------------------------------------------------

# opc 按算子目录独立拷贝 op_kernel；内核 #include "common/..." 依赖各阶段
# op_kernel/common/（由 k2q_csr_common/op_kernel 同步，且被 .gitignore）。
# 在 configure 阶段自动 vendor，保证 `build.sh --pkg --ops=k2q_csr` 无需手工 rsync。
set(_K2Q_CSR_COMMON_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/k2q_csr_common/op_kernel")
set(_K2Q_CSR_STAGE_OPS
    k2q_csr_meta
    k2q_csr_hist
    k2q_csr_row_prefix
    k2q_csr_tile_prefix
    k2q_csr_scatter
)
if(EXISTS "${_K2Q_CSR_COMMON_KERNEL}")
    foreach(_op ${_K2Q_CSR_STAGE_OPS})
        set(_dst "${CMAKE_CURRENT_SOURCE_DIR}/${_op}/op_kernel/common")
        file(MAKE_DIRECTORY "${_dst}")
        file(COPY "${_K2Q_CSR_COMMON_KERNEL}/"
             DESTINATION "${_dst}"
             FILES_MATCHING
             PATTERN "*.h"
             PATTERN "*.cpp"
             PATTERN "CMakeLists.txt" EXCLUDE)
    endforeach()
    # opc src_copy 用 .done 戳缓存；若此前拷过空 common，需清戳强制重拷
    file(GLOB _k2q_stale_src_dones
         "${CMAKE_BINARY_DIR}/binary/*/src/k2q_csr_*/*.done")
    if(_k2q_stale_src_dones)
        file(REMOVE ${_k2q_stale_src_dones})
    endif()
    message(STATUS "k2q_csr: vendored k2q_csr_common/op_kernel -> */op_kernel/common")
else()
    message(WARNING "k2q_csr: missing ${_K2Q_CSR_COMMON_KERNEL}; kernel build may fail")
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
# 非算子目录 / 接入层文件
list(REMOVE_ITEM CURRENT_DIRS framework docs examples k2q_csr_common build tmp kernel_meta
     k2q_csr_torch_adpt.h README.md)

foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/CMakeLists.txt")
        add_subdirectory(${SUB_DIR}/op_host)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
