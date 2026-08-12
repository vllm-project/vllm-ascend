# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

 if (BUILD_OPEN_PROJECT)
    #打桩文件
    file(GLOB MC2_DZ)
    if(MC2_DZ)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${MC2_DZ})
    endif()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ops_legacy/op_tiling)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ops_legacy/op_api)

    if(MC2_OPAPI_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${MC2_OPAPI_SRCS})
    endif()

    if(UT_TEST_ALL OR OP_HOST_UT)
        file(GLOB MC2_3RD_BLUE_COMM
            # mm  stub
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ut/framework_normal/common/op_cache_tiling.cpp
        )
    else()
        file(GLOB MC2_3RD_BLUE_COMM
            # mm  stub
            ${CMAKE_CURRENT_SOURCE_DIR}/ops_legacy/op_tiling/op_cache_tiling.cpp
        )
    endif()

    if(MC2_3RD_BLUE_COMM)
        add_tiling_modules()
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MC2_3RD_BLUE_COMM})
    endif()
endif()

file(GLOB MC2_3RD_SRCS
    # mm
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/mat_mul_v3_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/arch35/*.cpp
    # bmm
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/batch_mat_mul_v3_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/op_tiling/arch35/*.cpp
    # qbmm
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/arch35/*.cpp
    # wqbmm
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_quant_batch_matmul_v2/op_host/op_tiling/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/*.cpp
    # mm common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/op_host/op_tiling/*.cpp
    # gmm
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul/op_tiling/*.cpp
)

# MC2_COMPILE used to avoid tefusion_llt
# when blue use SUB_MC2_COMPILE when rty use MC2_COMPILE
if(MC2_3RD_SRCS AND (MC2_COMPILE OR SUB_MC2_COMPILE OR UT_TEST OR OP_HOST_UT))
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MC2_3RD_SRCS})
endif()

add_subdirectory(common/op_host)
