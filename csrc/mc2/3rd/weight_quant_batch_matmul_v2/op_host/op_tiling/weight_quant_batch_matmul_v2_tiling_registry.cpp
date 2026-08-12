/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"
#include "weight_quant_batch_matmul_v2_tiling_custom.h"
#include "weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h"
#include "weight_quant_batch_matmul_v2_tiling_fixpipe.h"
#include "weight_quant_batch_matmul_v2_tiling_msd.h"
#include "weight_quant_batch_matmul_v2_tiling_msd_group.h"
#include "weight_quant_batch_matmul_v2_tiling_splitk.h"
#include "weight_quant_batch_matmul_v2_weight_nz_tiling.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "platform/platform_infos_def.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {

constexpr int32_t SPLIT_K_PRIORITY = 0;
constexpr int32_t MSD_GROUP_PRIORITY = 1;
constexpr int32_t MSD_PRIORITY = 2;
constexpr int32_t CUSTOM_SPLITK_PRIORITY = 3;
constexpr int32_t CUSTOM_PRIORITY = 4;
constexpr int32_t FIXPIPE_PRIORITY = 5;
constexpr int32_t WEIGHT_NZ_PRIORITY = 6;
constexpr int32_t ADAPTIVE_SPLIT_PRIORITY = 7;
constexpr int32_t ANTI_REG_PRIORITY = 8;
constexpr int32_t ASW_PRIORITY = 9;

REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingSplitK, SPLIT_K_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingMsdGroup,
                             MSD_GROUP_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2Msd, MSD_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2CustomNzSplitK,
                             CUSTOM_SPLITK_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingCustom, CUSTOM_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingFixpipe, FIXPIPE_PRIORITY);
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2WeightNz, WEIGHT_NZ_PRIORITY);

static ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingFunc(gert::TilingContext *context)
{
    platform_ascendc::SocVersion socVersion;
    bool supportMmadS8S4 = false;
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Mc2WeightQuantBatchMatmulV2", "tilingContext is null");
    auto compileInfoPtr = reinterpret_cast<const Mc2WeightQuantBatchMatmulV2CompileInfo *>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
        std::string mmad;
        bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
        supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;
    } else {
        socVersion = compileInfoPtr->socVersion;
        supportMmadS8S4 = compileInfoPtr->supportMmadS8S4;
    }
    if (socVersion != platform_ascendc::SocVersion::ASCEND310P) {
        OP_LOGI(context->GetNodeName(), "Platform support Intrinsic_fix_pipe_l0c2out");
        if (supportMmadS8S4) {
            OP_LOGI(context->GetNodeName(), "Platform support Intrinsic_mmad s8s4");
            Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4 wqbmmv2Checker(context);
            OP_TILING_CHECK(
                wqbmmv2Checker.Check() != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "parameters",
                                                      std::to_string(static_cast<int>(wqbmmv2Checker.Check())).c_str(),
                                                      "The value of parameters must be valid."),
                return ge::GRAPH_FAILED);
            std::vector<int32_t> registerList = {ASW_PRIORITY};
            return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
        } else {
            OP_LOGI(context->GetNodeName(), "Platform not support Intrinsic_mmad s8s4");
            OP_TILING_CHECK(Mc2CheckPara(context, socVersion) != ge::GRAPH_SUCCESS,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                context->GetNodeName(), "parameters",
                                std::to_string(static_cast<int>(Mc2CheckPara(context, socVersion))).c_str(),
                                "The value of parameters must be valid."),
                            return ge::GRAPH_FAILED);
            if (socVersion != platform_ascendc::SocVersion::ASCEND910B) {
                OP_LOGI(context->GetNodeName(), "Platform support Intrinsic_data_move_l12bt bf16");
                std::vector<int32_t> registerList = {ADAPTIVE_SPLIT_PRIORITY, ANTI_REG_PRIORITY};
                return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
            } else {
                OP_LOGI(context->GetNodeName(), "Platform not support Intrinsic_data_move_l12bt bf16");
                return TilingRegistry::GetInstance().DoTilingImpl(context);
            }
        }
    } else {
        OP_LOGI(context->GetNodeName(), "Platform not support Intrinsic_fix_pipe_l0c2out");
        std::vector<int32_t> registerList = {WEIGHT_NZ_PRIORITY};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

static ge::graphStatus Mc2TilingParseForWeightQuantBatchMatmulV2(gert::TilingParseContext *context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Mc2WeightQuantBatchMatmulV2", "tilingParseContext is null");
    auto platformInfoPtr = context->GetPlatformInfo();
    // 在tilingParse获取不到GetPlatformInfo时，返回成功。会在之后的InitCompileInfo阶段设置compileInfo信息。
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_SUCCESS, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<Mc2WeightQuantBatchMatmulV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    OP_LOGE_IF(compileInfoPtr->aicNum == 0, ge::GRAPH_FAILED, context->GetNodeName(), "aicNum is 0");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    std::string mmad;
    bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    compileInfoPtr->supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;

    TilingPrepareForOpCache(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Mc2WeightQuantBatchMatmulV2)
    .Tiling(Mc2WeightQuantBatchMatmulV2TilingFunc)
    .TilingParse<Mc2WeightQuantBatchMatmulV2CompileInfo>(
        Mc2TilingParseForWeightQuantBatchMatmulV2); // 向框架注册入口函数
} // namespace optiling
