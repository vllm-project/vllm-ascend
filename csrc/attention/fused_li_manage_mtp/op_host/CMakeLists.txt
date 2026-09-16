add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE fused_li_manage_mtp_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME FusedLiManageMtp
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -mllvm -cce-vf-remove-membar=false
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_li_manage_mtp ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
