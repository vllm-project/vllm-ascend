if(NOT "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE gmm_dequant_situ_quant_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME GmmDequantSituQuant
    OPTIONS -Wno-deprecated-declarations
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE gmm_dequant_situ_quant ACLNNTYPE aclnn)
endif()
