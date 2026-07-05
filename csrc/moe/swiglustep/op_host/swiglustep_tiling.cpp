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
#include "error/ops_error.h"

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
    // ---- x: [M, 2N] row-major. M = rows, N = gate/up width = last dim / 2 ----
    auto* xShapeDesc = context->GetInputShape(0);
    auto xShape = xShapeDesc->GetStorageShape();
    const int64_t dimNum  = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t lastDim = xShape.GetDim(dimNum - 1);          // = 2N
    const int64_t N       = lastDim / 2;
    int64_t M = 1;
    for (int64_t i = 0; i < dimNum - 1; ++i) {
        M *= xShape.GetDim(i);
    }
    const int64_t totalLength = M;                              // rows

    const int64_t dtypeSize = GetDtypeSize(context->GetInputDesc(0)->GetDataType());

    // require x last dim even + N 32B-aligned (so every per-row DataCopy is aligned)
    const int64_t ubAlignElements = UB_ALIGN_BYTE / dtypeSize;  // 16 for bf16/fp16
    OPS_ERR_IF((lastDim % 2 != 0 || N % ubAlignElements != 0),
               OPS_LOG_E(context, "swiglustep: x last dim must be 2N with N %ld-aligned, got lastDim=%ld",
                         ubAlignElements, lastDim),
               return ge::GRAPH_FAILED);

    // empty input guard
    if (M == 0) {
        SwiglustepTilingData tilingData;
        tilingData.set_totalLength(0);
        tilingData.set_N(N);
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

    // ---- platform: core count + real per-core UB size (queried per-SoC, not hardcoded) ----
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (totalCoreNum == 0) {
        totalCoreNum = 1;
    }
    // reserve a fixed 16 KB for InitBuffer metadata/alignment (compressor_metadata style)
    uint64_t ubSizeTotal = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeTotal);
    constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
    const int64_t UB_SIZE_LIMIT = static_cast<int64_t>(ubSizeTotal) - UB_RESERVED_BYTES;

    // ---- tileM (rows per UB tile): UB bytes per out-element ----
    //   inQueueX  x BUFFER_NUM(2) [tileM,2N] = 2 * (2N/N) * dtypeSize = 8   (bf16/fp16)
    //   5 fp32 TBufs [tileM,N]               = 5 * 4                    = 20
    //   outQueue x BUFFER_NUM(2) [tileM,N]   = 2 * dtypeSize            = 4
    //   total = 32 B/out-element
    constexpr int64_t bufferCoefficient = 32;
    const int64_t maxTileElements = UB_SIZE_LIMIT / bufferCoefficient;   // out-elements
    // single row needs 32*N bytes (per out-element 32B); row-internal tiling is not
    // implemented, so N must fit one row in UB. Real MoE shapes (Step-3.7 N=1280) are far
    // below this cap; reject oversized N up front instead of crashing in InitBuffer.
    OPS_ERR_IF((N > maxTileElements),
               OPS_LOG_E(context, "swiglustep: N=%ld exceeds single-row UB capacity (%ld out-elements); "
                         "row-internal tiling not implemented", N, maxTileElements),
               return ge::GRAPH_FAILED);
    // N <= maxTileElements (asserted above) => tileM >= 1, no clamp needed
    int64_t tileM = maxTileElements / N;
    // N already 32B-aligned => any tileM keeps per-row DataCopy aligned; no extra round-up
    const int64_t tileLength = tileM;                                    // rows per tile

    // ---- block level: split by ROW (whole rows per core) ----
    // x is [M,2N] row-major; cutting mid-row would split the gate/up pair of one row,
    // so cores always split on row boundaries. rowBytes = 2*N*dtypeSize:
    //   - MoE N>=1280 (bf16): rowBytes=5120B = a multiple of the 512B cache line, so every
    //     core starts cache-aligned (best GM throughput).
    //   - Small N (e.g. N=160 single-card after TP8, or tiny test shapes): rowBytes may not
    //     be a 512B multiple -> non-tail cores start at non-cache-aligned addresses. This is
    //     correct (rowBytes is always 32B-aligned since N%16==0), only slightly slower on GM
    //     access; small-N decode shapes aren't where A2's contiguous-elimination wins anyway.
    // So no hard cache-line rejection: correctness only needs the 32B UB alignment already
    // guaranteed by the N % ubAlignElements check above.
    const int64_t rowsPerCore = (M + totalCoreNum - 1) / totalCoreNum;
    int64_t usedCoreNum       = (M + rowsPerCore - 1) / rowsPerCore;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    const int64_t formerNum    = usedCoreNum - 1;
    const int64_t formerLength = rowsPerCore;                  // rows
    const int64_t tailNum      = 1;
    const int64_t tailLength   = M - formerNum * rowsPerCore;  // rows

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
    tilingData.set_totalLength(totalLength);   // M rows
    tilingData.set_N(N);
    tilingData.set_formerNum(formerNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_tailNum(tailNum);
    tilingData.set_tailLength(tailLength);
    tilingData.set_tileLength(tileLength);     // tileM rows
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
