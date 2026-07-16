set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)
set(solve_tri_depends
    third_party/catlass/include/catlass
    third_party/catlass/include/tla
    CACHE INTERNAL "Dependencies for solve_tri")

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/solve_tri_def.cpp
    )
endif()

add_modules_sources(OPTYPE solve_tri ACLNNTYPE aclnn_exclude)
add_ops_tiling_keys(
    OP_NAME SolveTri
    TILING_KEYS 1
)

add_ops_compile_options(OP_NAME SolveTri OPTIONS
    --cce-auto-sync=off
    -I${CATLASS_INCLUDE_DIR_ABS}
)
