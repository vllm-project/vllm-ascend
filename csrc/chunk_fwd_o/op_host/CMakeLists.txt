# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 获取当前CMakeLists.txt所在目录的绝对路径
set(CURRENT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# 构造catlass的绝对路径（基于当前CMake目录）
set(CATLASS_INCLUDE_DIR "${CURRENT_CMAKE_DIR}/../../third_party/common")
# 转成绝对路径（避免任何相对路径问题）
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)



# ===================== 编译选项：用绝对路径 =====================
add_ops_compile_options(
        OP_NAME ChunkFwdO
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -I${CATLASS_INCLUDE_DIR_ABS}  # 用绝对路径，不是相对路径
)

# Host （aclnn）
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
           chunk_fwd_o_def.cpp
    )

    # Tiling
    target_sources(optiling PRIVATE
           chunk_fwd_o_tiling.cpp
    )

message(STATUS "Catlass绝对路径：${CATLASS_INCLUDE_DIR_ABS}")

    # 额外保障：给tiling模块也添加绝对路径的头文件目录
    target_include_directories(optiling PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CATLASS_INCLUDE_DIR_ABS}  # 同步添加绝对路径
    )

endif()
