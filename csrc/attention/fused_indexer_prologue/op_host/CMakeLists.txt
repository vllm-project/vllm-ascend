add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(fused_indexer_prologue_depends
        mla_preprocess
        CACHE INTERNAL "Dependencies for fused_indexer_prologue"
    )
    target_sources(op_host_aclnn PRIVATE
        fused_indexer_prologue_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedIndexerPrologue
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_indexer_prologue ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
