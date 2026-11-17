# The MX A5 prototype and ACLNN ABI stay in grouped_matmul_situ_quant.
if(NOT "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()
add_op_to_compiled_list()
target_sources(op_host_aclnn PRIVATE grouped_matmul_situ_quant_v2_def.cpp)
add_ops_compile_options(
    OP_NAME GroupedMatmulSituQuantV2
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)
add_modules_sources_with_soc(OPTYPE grouped_matmul_situ_quant_v2 ACLNNTYPE aclnn)
