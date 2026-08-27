add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_grouped_matmul_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MoeGroupedMatmulCustom
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_grouped_matmul ACLNNTYPE aclnn_inner)
    if (TARGET ${OPHOST_NAME}_tiling_obj)
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
            moe_grouped_matmul_cpu.cpp
        )
        target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

