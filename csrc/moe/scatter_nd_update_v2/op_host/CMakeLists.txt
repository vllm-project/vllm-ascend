# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

#optiling and opapi
#message(STATUS "=== Debug: start ops.index.scatter_nd_update_v2.CMakeLists.txt ")
#add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE scatter_nd_update_v2 ACLNNTYPE aclnn_exclude)


add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        scatter_nd_update_v2_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME ScatterNdUpdateV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)


if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE scatter_nd_update_v2 ACLNNTYPE aclnn_exclude)
endif()