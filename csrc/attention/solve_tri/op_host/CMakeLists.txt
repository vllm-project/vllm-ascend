add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/solve_tri_def.cpp
    )
endif()

add_modules_sources(OPTYPE solve_tri ACLNNTYPE aclnn_exclude)

add_ops_compile_options(OP_NAME SolveTri OPTIONS
    --cce-auto-sync=off
)
