add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE flash_mla_bf16_prepare_def.cpp)
    add_ops_compile_options(OP_NAME FlashMlaBf16Prepare OPTIONS --cce-auto-sync=off
        -Wno-deprecated-declarations -Werror)
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE flash_mla_bf16_prepare ACLNNTYPE aclnn_inner)
endif()
