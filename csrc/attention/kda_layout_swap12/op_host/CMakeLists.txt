# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License"). Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        kda_layout_swap12_def.cpp
    )
endif()

add_modules_sources(OPTYPE kda_layout_swap12 ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME KdaLayoutSwap12
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
