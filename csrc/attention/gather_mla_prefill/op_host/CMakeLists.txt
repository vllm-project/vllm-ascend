add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE gather_mla_prefill_def.cpp)
    add_ops_compile_options(OP_NAME GatherMlaPrefill OPTIONS --cce-auto-sync=off
        -Wno-deprecated-declarations -Werror)
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE gather_mla_prefill ACLNNTYPE aclnn_inner)
endif()
