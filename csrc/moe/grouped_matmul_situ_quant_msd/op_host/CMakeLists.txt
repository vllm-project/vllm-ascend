if(NOT "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE grouped_matmul_situ_quant_msd_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME GroupedMatmulSituQuantMsd
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

add_ops_tiling_keys(
    OP_NAME GroupedMatmulSituQuantMsd
    TILING_KEYS 2 3 4
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE grouped_matmul_situ_quant_msd ACLNNTYPE aclnn)
endif()
