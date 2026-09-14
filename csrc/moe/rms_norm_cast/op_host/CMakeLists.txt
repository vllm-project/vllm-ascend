# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE rms_norm_cast_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME RmsNormCast
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE rms_norm_cast ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
