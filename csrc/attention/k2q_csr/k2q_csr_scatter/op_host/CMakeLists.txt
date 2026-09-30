# -----------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        k2q_csr_scatter_def.cpp
    )
    add_ops_compile_options(
        OP_NAME K2qCsrScatter
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME K2qCsrScatter
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
endif()

add_modules_sources_with_soc(OPTYPE k2q_csr_scatter ACLNNTYPE aclnn)
