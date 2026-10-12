/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gmm_dequant_situ_quant_tiling.h"

namespace optiling {
namespace {
struct GmmDequantSituQuantCompileInfo {};

// Mirrors GmsqFusedAivKernel256::Init(rawInt32=true). All rows are already
// 32-byte aligned; metadata and the scalar reduction outputs are rounded up.
// Sigmoid explicitly reuses aBuf_, so it requires no hidden UB stack space.
static int64_t MsdUbBytes(int64_t experts, int64_t N2)
{
    const auto align32 = [](int64_t bytes) { return (bytes + 31) / 32 * 32; };
    const int64_t packing = 2 * 4096 + 16384 + 2 * 8192;
    const int64_t rawAndScale = 4 * N2 + std::max<int64_t>(4 * N2, 8192);
    const int64_t rows = 4 * 4 * N2 + 2 * N2 + N2;
    const int64_t scalarAndReduction = 32 + 32 + 4096 + align32(((N2 + 1023) / 1024) * 4);
    return packing + rawAndScale + rows + scalarAndReduction + align32((3 * experts + 4) * 4);
}

} // namespace

ge::graphStatus TilingGmmDequantSituQuant(gert::TilingContext *context)
{
    const auto *x = context->GetDynamicInputShape(0, 0);
    const auto *weight = context->GetDynamicInputShape(1, 0);
    if (x == nullptr || weight == nullptr) {
        return ge::GRAPH_FAILED;
    }
    GmmDequantSituQuantTilingData data{};
    data.experts = context->GetIrInputInstanceInfo(1)->GetInstanceNum();
    data.k = x->GetOriginShape().GetDim(1);
    data.capacity = x->GetOriginShape().GetDim(0);
    data.n = weight->GetOriginShape().GetDim(1) * 8;
    const auto *attrs = context->GetAttrs();
    data.beta = *attrs->GetAttrPointer<float>(0);
    data.linearBeta = *attrs->GetAttrPointer<float>(1);
    data.hasLinear = *attrs->GetAttrPointer<bool>(2);
    data.groupListType = *attrs->GetAttrPointer<int64_t>(3);
    data.weightNz = *attrs->GetAttrPointer<bool>(4);
    data.invBeta = 1.0f / data.beta;
    data.invLinearBeta = data.hasLinear ? 1.0f / data.linearBeta : 1.0f;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCores = platform.GetCoreNumAic();
    if (aicCores == 0 || platform.GetCoreNumAiv() == 0) {
        OP_LOGE(context->GetNodeName(), "failed to query core numbers");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t requiredUb = MsdUbBytes(data.experts, data.n / 2);
    if (ubBytes == 0 || static_cast<uint64_t>(requiredUb) > ubBytes) {
        OP_LOGE(context->GetNodeName(), "unsupported N/E for the row epilogue: requires %ld UB bytes, device provides %lu",
                requiredUb, ubBytes);
        return ge::GRAPH_FAILED;
    }

    // Same eight scratch slots and 272 rows per slot as the direct launcher.
    constexpr uint64_t msdSlots = 8;
    constexpr uint64_t msdRawRows = 2 * 128 + 16;
    data.packedBytes = msdSlots * msdRawRows * data.k / 2;
    const uint64_t rawBytes = msdSlots * msdRawRows * data.n * sizeof(int32_t);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + data.packedBytes + rawBytes;
    context->SetBlockDim(aicCores);
    context->SetTilingKey(0);
    return context->GetRawTilingData()->Append(data);
}

ge::graphStatus TilingPrepareGmmDequantSituQuant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GmmDequantSituQuant)
    .Tiling(TilingGmmDequantSituQuant)
    .TilingParse<GmmDequantSituQuantCompileInfo>(TilingPrepareGmmDequantSituQuant);
} // namespace optiling
