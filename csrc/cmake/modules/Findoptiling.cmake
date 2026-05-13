# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

unset(OPTILING_SO_LIB_DIR)

file(GLOB_RECURSE LIB_OP_TILING_SO
    ${ASCEND_DIR}/../liboptiling.so
)

list(FILTER LIB_OP_TILING_SO INCLUDE REGEX "lib/linux/${CMAKE_SYSTEM_PROCESSOR}")

if(LIB_OP_TILING_SO)
    list(GET LIB_OP_TILING_SO 0 LIB_OP_TILING_SO_PATH)
    get_filename_component(LIB_OP_TILING_SO_PATH ${LIB_OP_TILING_SO_PATH} DIRECTORY)
    message(STATUS "Found optiling so lib:${LIB_OP_TILING_SO_PATH}")
else()
    message(STATUS "Cannot find library optiling so")
endif()

