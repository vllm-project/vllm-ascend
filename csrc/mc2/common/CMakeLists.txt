# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(CONDITION_UNIT_COMMON ${ASCEND_COMPUTE_UNIT})
if (BUILD_OPEN_PROJECT)
    file(GLOB OPTILING_SRCS
        utils/context_transfer.cpp
        op_host/op_tiling/hccl_formulaic_tiling.cpp
        op_host/op_tiling/hccl_performance.cpp
        op_host/op_tiling/hccl_performance_arch35.cpp
        op_host/op_tiling/matmul_formulaic_tiling.cpp
        op_host/op_tiling/matmul_performance.cpp
        op_host/op_tiling/mc2_fit_based_balance_tiling.cpp
        utils/mc2_hcom_topo_info.cpp
        utils/mc2_log.cpp
        op_host/op_tiling/mc2_matmul_tiling_cfg.cpp
        utils/mc2_moe_utils.cpp
        op_host/op_tiling/mc2_tiling_utils.cpp
        op_host/op_tiling/moe_tiling_base.cpp
        op_host/op_tiling/one_calc_two_comm_tiling.cpp
    )

    if(ENABLE_BUILT_IN)
        list(APPEND OPTILING_SRCS op_host/mc2_common_infershape.cpp)
    endif()

    if(UT_TEST_ALL OR OP_HOST_UT)
        list(FILTER OPTILING_SRCS EXCLUDE REGEX "utils/mc2_hcom_topo_info\\.cpp")
        file(GLOB MC2_TILING_UT_STUB ${UT_COMMON_INC}/mc2_hcom_topology_mocker.cpp ${UT_COMMON_INC}/op_cache_tiling.cpp ${UT_COMMON_INC}/rt_soc_spec_mocker.cpp)
        add_library(mc2_tiling_ut_stub SHARED ${MC2_TILING_UT_STUB})
        target_include_directories(mc2_tiling_ut_stub PRIVATE
            ${OPS_TRANSFORMER_DIR}/mc2/common/utils
            ${OPS_TRANSFORMER_DIR}/mc2/common/op_host/op_tiling
            ${OP_TILING_INCLUDE}
            ${ASCEND_DIR}/pkg_inc
        )
    endif()
    if(OPTILING_SRCS)
        add_tiling_modules()
        target_compile_options(${OPHOST_NAME}_tiling_obj PRIVATE
            -include${OPS_TRANSFORMER_DIR}/mc2/mega_moe/mega_moe_build_compat.h)
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
        if(UT_TEST_ALL OR OP_HOST_UT)
            target_link_directories(${OPHOST_NAME}_tiling_obj PRIVATE mc2_tiling_ut_stub)
        endif()
    endif()
    if(NOT ENABLE_BUILT_IN)
        set(MC2_INFER_SRCS
            op_host/mc2_common_infershape.cpp
        )
        if(MC2_INFER_SRCS)
            add_infer_modules()
            target_compile_options(${OPHOST_NAME}_infer_obj PRIVATE
                -include${OPS_TRANSFORMER_DIR}/mc2/mega_moe/mega_moe_build_compat.h)
            target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${MC2_INFER_SRCS})
        endif()
    endif()
    file(GLOB COMMON_UTILS_SRCS
        utils/mc2_platform_info.cpp
    )
    file(GLOB GENTASK_SRCS
        op_graph/mc2_gen_task_ops_utils.cpp
        op_graph/matmul_all_reduce_gen_task_ops_utils.cpp
        op_graph/mc2_moe_gen_task_ops_utils.cpp
    )
    if("${CONDITION_UNIT_COMMON}" STREQUAL "ascend950")
        list(APPEND GENTASK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_graph/mc2_gen_task_ops_utils_arch35.cpp)
    else()
        list(APPEND GENTASK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_graph/mc2_gen_task_ops_utils_arch35_stub.cpp)
    endif()
    # vLLM Ascend's custom-op build does not provide the ops-transformer
    # graph-mode gentask target. ACLNN execution does not use these sources.
    if(GENTASK_SRCS AND COMMAND add_gentask_modules)
        add_gentask_modules()
        target_sources(${OPGRAPH_NAME}_gentask_obj PRIVATE
            ${COMMON_UTILS_SRCS}
            ${GENTASK_SRCS}
        )
    endif()
    file(GLOB MC2_GRAPH_PLUGIN_SRCS
        op_graph/mc2_moe_graph_context.cpp
    )
    if(MC2_GRAPH_PLUGIN_SRCS)
        add_graph_plugin_modules()
        target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${MC2_GRAPH_PLUGIN_SRCS})
    endif()
    if(NOT ENABLE_TEST)
        file(GLOB OPPROTO_SRCS)
        if(OPPROTO_SRCS)
            add_infer_modules()
            target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPPROTO_SRCS})
        endif()
        file(GLOB OPAPI_SRC)
        if(OPAPI_SRC)
            add_opapi_modules()
            target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRC})
        endif()
    endif()
endif()

file(GLOB NEW_MC2_MM_SRC op_host/op_tiling/new_mc2_*.cpp)
# MC2_COMPILE used to avoid tefusion_llt
# when blue use SUB_MC2_COMPILE when rty use MC2_COMPILE
if(MC2_COMPILE OR SUB_MC2_COMPILE OR UT_TEST OR OP_HOST_UT)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${NEW_MC2_MM_SRC})
endif()

file(GLOB OPAPI_SRC
    op_api/mc2_aclnn_util.cpp
    op_api/mc2_context.cpp
)
if(OPAPI_SRC)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRC})
endif()
