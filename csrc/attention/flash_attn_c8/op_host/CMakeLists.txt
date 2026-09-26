set(flash_attn_c8_depends "attention/a5_mla_common" CACHE INTERNAL "C8 attention shared pipeline")
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE flash_attn_c8_def.cpp)
    add_ops_compile_options(OP_NAME FlashAttnC8 OPTIONS --cce-auto-sync=off
        -DENABLE_CV_COMM_VIA_SSBUF=true -Wno-deprecated-declarations -Werror)
    # Offline adapters search the same relative common path as installed JIT
    # adapters. Keep the generated package independent of the build directory.
    configure_file(${OPS_TRANSFORMER_DIR}/common/include/cann_compat.h
        ${CMAKE_BINARY_DIR}/binary/ascend950/src/ascendc/common/cann_compat.h COPYONLY)
endif()
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE flash_attn_c8 ACLNNTYPE aclnn_inner)
endif()
