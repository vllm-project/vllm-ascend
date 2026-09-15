add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        fused_scatter_copy_sparse_flash_attention_first_fill_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedScatterCopySparseFlashAttentionFirstFill
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_scatter_copy_sparse_flash_attention_first_fill ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
