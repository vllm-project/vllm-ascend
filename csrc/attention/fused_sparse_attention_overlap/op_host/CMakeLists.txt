add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(fused_sparse_attention_overlap_depends attention/common CACHE INTERNAL
        "Dependencies for fused_sparse_attention_overlap")
    target_sources(op_host_aclnn PRIVATE
        fused_sparse_attention_overlap_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedSparseAttentionOverlap
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_sparse_attention_overlap ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
