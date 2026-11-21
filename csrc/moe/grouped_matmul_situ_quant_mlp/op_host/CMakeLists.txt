if(NOT "ascend910b" IN_LIST ASCEND_COMPUTE_UNIT AND NOT "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    set(grouped_matmul_situ_quant_mlp_depends moe/grouped_matmul_situ_quant CACHE INTERNAL
        "Kernel sources shared by the SiTU operators")
    target_sources(op_host_aclnn PRIVATE grouped_matmul_situ_quant_mlp_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME GroupedMatmulSituQuantMlp
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

add_ops_tiling_keys(
    OP_NAME GroupedMatmulSituQuantMlp
    TILING_KEYS 0 1
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE grouped_matmul_situ_quant_mlp ACLNNTYPE aclnn)
endif()
