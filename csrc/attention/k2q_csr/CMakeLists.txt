# -----------------------------------------------------------------------------------------------------------
#
# k2q_csr：父目录挂载独立命名阶段算子（Meta/Hist/RowPrefix/TilePrefix/Scatter）
# -----------------------------------------------------------------------------------------------------------

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
