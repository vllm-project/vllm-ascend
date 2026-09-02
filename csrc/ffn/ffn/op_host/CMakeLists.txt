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
    target_sources(op_host_aclnnExc PRIVATE
        ffn_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
            OP_NAME FFN
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_modules_sources(OPTYPE "ffn" ACLNNTYPE "aclnn_exclude")
    # mat_mul_v3 host tiling
    set(FFN_OFFICIAL_MM_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.cpp
    )
    file(GLOB FFN_OFFICIAL_COMMON_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/common/op_host/op_tiling/*.cpp
    )
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${FFN_OFFICIAL_MM_TILING_SRCS}
        ${FFN_OFFICIAL_COMMON_TILING_SRCS}
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/common/inc
    )
    target_link_libraries(${OPHOST_NAME}_tiling_obj INTERFACE $<$<TARGET_EXISTS:opsbase>:opsbase>)
endif()
