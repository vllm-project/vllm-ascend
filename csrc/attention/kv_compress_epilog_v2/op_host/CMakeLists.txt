# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE kv_compress_epilog_v2_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME KvCompressEpilogV2
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_compress_epilog_v2 ACLNNTYPE aclnn)
endif()
