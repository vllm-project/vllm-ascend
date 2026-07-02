/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglustep_tiling.cpp
 * \brief SwigluStep tiling: two-level split (inter-core 512B cache line align + intra-core UB align)
 */
#include "swiglustep_tiling.h"

#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int64_t CACHE_LINE_BYTE        = 512;                 // inter-core cache line align
constexpr int64_t UB_ALIGN_BYTE          = 32;                  // intra-core UB align
constexpr uint64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;    // elementwise workspace: 16 MB
// UB_SIZE_LIMIT is queried from the platform at tiling time (see GetCoreMemSize)

static int64_t GetDtypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:   return 4;
        case ge::DT_FLOAT16: return 2;
        case ge::DT_BF16:    return 2;
        default:             return 2;
    }
}

static ge::graphStatus TilingForSwiglustep(gert::TilingContext* context)
{
    // ---- totalLength = total elements of gate (flattened M*N) ----
    auto* gateShapeDesc = context->GetInputShape(0);
    auto gateShape = gateShapeDesc->GetStorageShape();
    int64_t totalLength = 1;
    for (size_t i = 0; i < gateShape.GetDimNum(); ++i) {
        totalLength *= gateShape.GetDim(i);
    }

    // empty input guard: single core, zero tiling, avoid div-by-zero in usedCoreNum
    if (totalLength == 0) {
        SwiglustepTilingData tilingData;
        tilingData.set_totalLength(0);
        tilingData.set_formerNum(0);
        tilingData.set_formerLength(0);
        tilingData.set_tailNum(0);
        tilingData.set_tailLength(0);
        tilingData.set_tileLength(0);
        tilingData.set_limit(7.0f);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t dtypeSize = GetDtypeSize(context->GetInputDesc(0)->GetDataType());

    // ---- platform: core count + real per-core UB size (queried per-SoC, not hardcoded) ----
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum == 0) {
        totalCoreNum = 1;
    }
    // Query the real per-core UB size and reserve 10% margin for InitBuffer
    // alignment/metadata overhead, so tileLength never sizes right up to the limit.
    uint64_t ubSizeTotal = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTotal);
    const int64_t UB_SIZE_LIMIT = static_cast<int64_t>(ubSizeTotal * 9 / 10);

    // ---- tileLength: UB level ----
    // bufferCoefficient = UB bytes per tileLength element (op def: bf16/fp16, dtypeSize=2):
    //   3 queues x BUFFER_NUM(2) x dtypeSize(2) = 12,  plus 5 fp32 TBufs x 4B = 20  -> 32 B/element
    // (revisit if fp32 is added to the op def: 3x2x4 + 20 = 44)
    constexpr int64_t bufferCoefficient = 32;
    // tileLength = UB bytes / (B/element); bufferCoefficient already embeds dtype —
    // do NOT multiply by dtypeSize again (prior bug halved tileLength: 6144 -> 3072).
    const int64_t maxTileElements   = UB_SIZE_LIMIT / bufferCoefficient;
    const int64_t ubAlignElements   = UB_ALIGN_BYTE / dtypeSize;
    const int64_t tileLength        = (maxTileElements / ubAlignElements) * ubAlignElements;

    // require 32B-aligned totalLength so every DataCopy (incl. the tail core's
    // tail tile) stays aligned. MoE FFN guarantees this: totalLength = M*N, where
    // N (=moe_intermediate, e.g. 1280) is a multiple of 16 elems = 32B for bf16/fp16.
    // Reject other shapes explicitly rather than crash on an unaligned DataCopy.
    if (totalLength % ubAlignElements != 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- block level: 512B cache line align, former/tail core load balance ----
    const int64_t alignElements        = CACHE_LINE_BYTE / dtypeSize;
    const int64_t totalLengthCore      = (totalLength + totalCoreNum - 1) / totalCoreNum;
    const int64_t totalLengthCoreAlign =
        ((totalLengthCore + alignElements - 1) / alignElements) * alignElements;
    int64_t usedCoreNum =
        (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    const int64_t formerNum    = usedCoreNum - 1;
    const int64_t formerLength = totalLengthCoreAlign;
    const int64_t tailNum      = 1;
    const int64_t tailLength   = totalLength - formerNum * formerLength;

    // ---- limit: Attr 0 (REQUIRED Float) ----
    float limit = 7.0f;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* limitAttr = attrs->GetAttrPointer<float>(0);
        if (limitAttr != nullptr) {
            limit = *limitAttr;
        }
    }

    // ---- fill tiling and write back ----
    SwiglustepTilingData tilingData;
    tilingData.set_totalLength(totalLength);
    tilingData.set_formerNum(formerNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailNum(tailNum);
    tilingData.set_tailLength(tailLength);
    tilingData.set_tileLength(tileLength);
    tilingData.set_limit(limit);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYSTEM_WORKSPACE_SIZE;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Swiglustep).Tiling(TilingForSwiglustep);

}  // namespace optiling
