add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_dispatch_normal.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MoeDispatchNormal
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_dispatch_normal ACLNNTYPE aclnn_inner)
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        moe_dispatch_normal.cpp
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_SOURCE_DIR}/utils/inc/kernel
    )
endif()
