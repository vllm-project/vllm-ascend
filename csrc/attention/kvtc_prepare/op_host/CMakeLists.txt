add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        kvtc_prepare_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME KvtcPrepare
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -mllvm -cce-aicore-hoist-movemask=false
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE kvtc_prepare
        ACLNNTYPE aclnn
    )

    add_tiling_modules()

    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kvtc_prepare_tiling.cpp
    )
endif()
