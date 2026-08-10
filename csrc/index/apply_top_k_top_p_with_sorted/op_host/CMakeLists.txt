add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        apply_top_k_top_p_with_sorted_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ApplyTopKTopPWithSorted
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE apply_top_k_top_p_with_sorted ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()