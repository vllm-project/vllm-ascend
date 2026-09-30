# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(CUSTOM_MULS_SUPPORTED_SOCS ascend910b ascend910_93)
set(CUSTOM_MULS_ENABLED FALSE)
foreach(SOC_VERSION ${ASCEND_COMPUTE_UNIT})
    if(SOC_VERSION IN_LIST CUSTOM_MULS_SUPPORTED_SOCS)
        set(CUSTOM_MULS_ENABLED TRUE)
    endif()
endforeach()
if(NOT CUSTOM_MULS_ENABLED)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        custom_muls_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME customMuls
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE custom_muls ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
