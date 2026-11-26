# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

if (NOT BUILD_OPEN_PROJECT OR BUILD_OPS_RTY_KERNEL)
    return()
endif()

add_op_to_compiled_list()

# The tiling calls HcPreTiling on the host. Kernel sources also use headers
# from both dependencies when building only hc_post_pre_norm.
set(hc_post_pre_norm_depends
    moe/hc_pre
    moe/rms_norm_cast
    CACHE INTERNAL "Dependencies for hc_post_pre_norm"
)

target_sources(op_host_aclnn PRIVATE hc_post_pre_norm_def.cpp)

add_ops_compile_options(
    OP_NAME HcPostPreNorm
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)

add_modules_sources(OPTYPE hc_post_pre_norm ACLNNTYPE aclnn)
