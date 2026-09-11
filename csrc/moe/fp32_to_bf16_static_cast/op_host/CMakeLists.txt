# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        fp32_to_bf16_static_cast_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Fp32ToBf16StaticCast
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fp32_to_bf16_static_cast ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
