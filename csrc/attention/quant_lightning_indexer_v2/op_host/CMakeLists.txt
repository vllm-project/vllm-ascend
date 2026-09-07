add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        quant_lightning_indexer_v2_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantLightningIndexerV2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quant_lightning_indexer_v2 ACLNNTYPE aclnn)
endif()
