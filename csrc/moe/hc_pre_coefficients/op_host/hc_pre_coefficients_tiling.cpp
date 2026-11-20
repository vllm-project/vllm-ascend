/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "../../hc_pre/op_host/hc_pre_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(HcPreCoefficients, HcPreTilingData)

namespace {
constexpr int64_t HC_MULT = 4;
constexpr int64_t HIDDEN_SIZE = 5120;
constexpr int64_t PROJECTION_SIZE = 24;
constexpr int64_t VECTORS_PER_CUBE = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t CUBE_M_ALIGN = 16;
constexpr int64_t CORE_K_ALIGN = 256;
constexpr int64_t L1_K_ALIGN = 128;
constexpr int64_t L1_ACTIVATION_ELEMENTS = 128 * 256;
constexpr int64_t MAX_M_L1_ROWS = L1_ACTIVATION_ELEMENTS / L1_K_ALIGN;
constexpr int64_t CV_K = 1024;
constexpr int64_t WORKSPACE_ALIGN = 512;
constexpr int64_t SYSTEM_WORKSPACE = 16 * 1024 * 1024;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t Align(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

ge::graphStatus TilingForHcPreCoefficients(gert::TilingContext *context)
{
    const auto shape = context->GetInputShape(0)->GetStorageShape();
    const int64_t rows = shape.GetDim(0);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t cubeCores = std::min<int64_t>(platform.GetCoreNumAic(),
        platform.GetCoreNumAiv() / VECTORS_PER_CUBE);
    OPS_ERR_IF(rows <= 0,
        OPS_LOG_E(context->GetNodeName(), "HcPreCoefficients requires a positive row count"),
        return ge::GRAPH_FAILED);

    // Match the shared projection's (M, K) block mapping and L1 tile bounds.
    constexpr int64_t width = HC_MULT * HIDDEN_SIZE;
    const int64_t mGroups = std::min(cubeCores, CeilDiv(rows, MAX_M_L1_ROWS));
    const int64_t rowsPerMGroup = Align(CeilDiv(rows, mGroups), CUBE_M_ALIGN);
    const int64_t rowsL1 = std::min(rowsPerMGroup, MAX_M_L1_ROWS);
    const int64_t splitK = Align(CeilDiv(width, cubeCores / mGroups), CORE_K_ALIGN);
    const int64_t kBlocks = CeilDiv(width, splitK);
    const int64_t rowsPerCore = CeilDiv(rows, cubeCores * VECTORS_PER_CUBE);
    const int64_t usedVectorCores = CeilDiv(rows, rowsPerCore);
    const int64_t tailRows = rows - (usedVectorCores - 1) * rowsPerCore;
    const int64_t projectionBlocks = CeilDiv(rows, rowsPerMGroup) * kBlocks;
    const int64_t launchBlocks = std::max(projectionBlocks, CeilDiv(usedVectorCores, VECTORS_PER_CUBE));
    uint64_t ubBytes;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t bufferBytesPerRow = 2 * CV_K * (sizeof(uint16_t) + sizeof(float));

    auto attrs = context->GetAttrs();
    HcPreTilingData data;
    data.set_bs(rows);
    data.set_hcMix(PROJECTION_SIZE);
    data.set_hcMult(HC_MULT);
    data.set_hcMultAlign(Align(HC_MULT, BLOCK_SIZE / sizeof(float)));
    data.set_k(width);
    data.set_cubeBlockDimK(kBlocks);
    data.set_cubeCoreNum(launchBlocks);
    data.set_multCoreSplitMSize(rowsPerMGroup);
    data.set_multCoreSplitKSize(splitK);
    data.set_mL1Size(rowsL1);
    data.set_cvLoopKSize(CV_K);
    data.set_kL1Size(std::min(CV_K, L1_ACTIVATION_ELEMENTS / rowsL1) / L1_K_ALIGN * L1_K_ALIGN);
    data.set_stage1MFactor(std::min<int64_t>(
        CeilDiv(std::min(rows, rowsL1), VECTORS_PER_CUBE), ubBytes / bufferBytesPerRow));
    // Process one coefficient row per UB tile.
    data.set_rowOfFormerBlock(rowsPerCore);
    data.set_rowOfTailBlock(tailRows);
    data.set_rowLoopOfFormerBlock(rowsPerCore);
    data.set_rowLoopOfTailBlock(tailRows);
    data.set_tailRowFactorOfFormerBlock(1);
    data.set_tailRowFactorOfTailBlock(1);
    data.set_stage2RowFactor(1);
    data.set_secondUsedCoreNum(usedVectorCores);
    data.set_iterTimes(*attrs->GetAttrPointer<int64_t>(0));
    data.set_hcEps(*attrs->GetAttrPointer<float>(1));
    data.set_normEps(*attrs->GetAttrPointer<float>(2));

    const uint64_t projectionInputBytes = launchBlocks * 2 * rowsL1 * CV_K * sizeof(float);
    const uint64_t projectionOutputBytes = Align(kBlocks * rows * L1_K_ALIGN * sizeof(float), WORKSPACE_ALIGN);
    const uint64_t squareSumBytes = Align(
        kBlocks * Align(rows, CUBE_M_ALIGN) * CUBE_M_ALIGN * sizeof(float), WORKSPACE_ALIGN);

    context->SetBlockDim(launchBlocks);
    context->SetTilingKey(2);
    context->GetWorkspaceSizes(1)[0] = SYSTEM_WORKSPACE + projectionInputBytes + projectionOutputBytes + squareSumBytes;
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareHcPreCoefficients(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(HcPreCoefficients)
    .Tiling(TilingForHcPreCoefficients)
    .TilingParse<HcPreCompileInfo>(PrepareHcPreCoefficients);
} // namespace optiling
