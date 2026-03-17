/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swi_glu_dynamic_quant_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "swi_glu_dynamic_quant_tiling_utils.h"


namespace optiling {
constexpr uint32_t DQ_BLOCK_SIZE = 32;
constexpr uint32_t DQ_L2_CACHE_LINE_SIZE = 512;

constexpr uint32_t DQ_SINGLE_UB_SIZE = 25;

static std::map<const ge::DataType, const uint32_t> dq_x_dTypeLen = { { ge::DT_FLOAT16, 2 },
    { ge::DT_BF16, 2 },
    { ge::DT_FLOAT, 4 } };

inline static ge::graphStatus SetTilingDataForSwiGluDynamicQuant(gert::TilingContext *context,
    SwiGluDynamicQuantTilingData &tilingData)
{
    OP_LOGD(context, "SetTilingDataForSwiGluDynamicQuant start.");
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(context, "SetTilingDataForSwiGluDynamicQuant end.");
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus DQGetCompileInfo(gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo)
{
    OP_LOGD(context, "GetCompileInfo start.");
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);

    if (totalCoreNum == 0 || ubSize <= 0) {
        OP_LOGD(context, "GetCompileInfo Failed, coreNum:%d, ubSize:%d.", totalCoreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    compileInfo.totalCore = totalCoreNum;
    compileInfo.ubSize = ubSize;
    OP_LOGD(context, "GetCompileInfo end.");
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus DQGetTillingData(gert::TilingContext *context, SwiGluDynamicQuantCompileInfo &compileInfo,
    SwiGluDynamicQuantTilingParam &tilingParam, SwiGluDynamicQuantTilingData &tilingData)
{
    OP_LOGD(context, "GetTillingData start.");
    if (DQCheckOpParams(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto xDtype = context->GetInputDesc(DQ_INPUT_X_INDEX)->GetDataType();
    uint32_t tilingKey = DQCalculateTilingKey(xDtype, compileInfo);

    tilingData.set_tilingKey(tilingKey);
    tilingData.set_groupLen(compileInfo.groupLength);
    compileInfo.inputDataByte = dq_x_dTypeLen[xDtype];
    compileInfo.dataNumSingleUb = compileInfo.ubSize / DQ_SINGLE_UB_SIZE;

    compileInfo.block_num = DQ_BLOCK_SIZE / compileInfo.inputDataByte;
    compileInfo.cacheLineLen = DQ_L2_CACHE_LINE_SIZE / compileInfo.inputDataByte;

    auto inputShape = context->GetInputTensor(DQ_INPUT_X_INDEX)->GetStorageShape();
    if (!DQSetTotalShape(inputShape, context, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    DQCalTilingData(context, compileInfo, tilingParam, tilingData);
    OP_LOGD(context, "GetTillingData end.");
    DQSetTilingData(compileInfo, tilingParam, tilingData);
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus Tiling4SwiGluDynamicQuant(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4SwiGluDynamicQuant start.");
    SwiGluDynamicQuantCompileInfo compileInfo;
    SwiGluDynamicQuantTilingParam tilingParam;
    SwiGluDynamicQuantTilingData tilingData;
    if (DQGetCompileInfo(context, compileInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DQGetTillingData(context, compileInfo, tilingParam, tilingData);
    SetTilingDataForSwiGluDynamicQuant(context, tilingData);
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t *workspaces = context->GetWorkspaceSizes(1);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaces[0] = sysWorkspaceSize + tilingData.get_realCoreNum() * DQ_BLOCK_SIZE;
    OP_LOGD(context, "Tiling4SwiGluDynamicQuant end.");
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingPrepare4SwiGluDynamicQuant(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4SwiGluDynamicQuant start and end.");
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SwiGluDynamicQuant).Tiling(Tiling4SwiGluDynamicQuant).TilingParse<SwiGluDynamicQuantCoreCompileInfo>(TilingPrepare4SwiGluDynamicQuant);
} // namespace optiling
