# Reuse the normal operator dependency/source packaging mechanism.
set(mla_prolog_dcp_c8_depends "attention/mla_prolog_v3" CACHE INTERNAL "MLAPO shared implementation")
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE mla_prolog_dcp_c8_def.cpp)
    add_ops_compile_options(OP_NAME MlaPrologDcpC8 OPTIONS --cce-auto-sync=off
        -Wno-deprecated-declarations -Werror -mllvm -cce-aicore-hoist-movemask=false)
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mla_prolog_dcp_c8 ACLNNTYPE aclnn_inner)
endif()
