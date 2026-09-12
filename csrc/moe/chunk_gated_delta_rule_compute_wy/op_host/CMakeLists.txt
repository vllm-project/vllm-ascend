set(CURRENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_compute_wy_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleComputeWy
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_gated_delta_rule_compute_wy ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
