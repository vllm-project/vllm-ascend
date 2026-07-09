# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        scatter_nd_update_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ScatterNdUpdate
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    set(SUPPORTED_ARCHS arch32 arch35)
    add_modules_sources(
        OPTYPE scatter_nd_update
        ACLNNTYPE aclnn_exclude
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    )
    add_tiling_modules()

    foreach(ARCH ${ARCH_DIRECTORY})
        if(ARCH IN_LIST SUPPORTED_ARCHS)
            if(ARCH STREQUAL arch32)
                target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/scatter_nd_update_tiling.cpp
                )
            elseif(ARCH STREQUAL arch35)
                target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/scatter_nd_update_tiling_arch35.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/scatter_nd_update_tiling_regbase.cpp
                )
            endif()
            target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}
                ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/${ARCH}
            )
        endif()
    endforeach()
endif()
