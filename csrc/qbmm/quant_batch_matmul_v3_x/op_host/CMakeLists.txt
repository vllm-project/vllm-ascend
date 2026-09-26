# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        quant_batch_matmul_v3_x_def.cpp
    )
endif()

add_modules_sources(OPTYPE quant_batch_matmul_v3_x ACLNNTYPE aclnn)

# The tiling helper sources are not covered by add_modules_sources' globs
# (which only pick up *_tiling*.cpp / *fallback*.cpp / op_tiling/*.cpp), so
# compile them into the tiling object explicitly. Otherwise liboptiling.so
# links with an undefined cost-model symbol and the tbe dlopen fails.
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    tiling_cost_model.cpp
)
target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
