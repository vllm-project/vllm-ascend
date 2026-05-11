add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_layout.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DispatchLayout
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_layout ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
        dispatch_layout.cpp
    )
endif()
