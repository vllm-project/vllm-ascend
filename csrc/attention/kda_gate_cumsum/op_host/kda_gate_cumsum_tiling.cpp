/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_gate_cumsum_tiling.h"
#include <algorithm>
#include <cstring>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_G_IDX = 0;
constexpr size_t INPUT_A_LOG_IDX = 1;
constexpr size_t INPUT_DT_BIAS_IDX = 2;
constexpr size_t INPUT_CU_SEQLENS_IDX = 3;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;
constexpr size_t ATTR_USE_GATE_IDX = 1;
constexpr size_t ATTR_SAFE_GATE_IDX = 2;
constexpr size_t ATTR_LOWER_BOUND_IDX = 3;
constexpr size_t ATTR_LAYOUT_IDX = 4;
constexpr int64_t MAX_K_DIM = 256;

enum class KdaGateLayout : int64_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    NTD = 3,
};
} // namespace

ge::graphStatus Tiling4KdaGateCumsum(gert::TilingContext *context)
{
    KdaGateCumsumTilingData tiling;
    auto gShape = context->GetOptionalInputShape(INPUT_G_IDX)->GetStorageShape();
    auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    if (gDesc == nullptr || (gShape.GetDimNum() != 3 && gShape.GetDimNum() != 4)) {
        return ge::GRAPH_FAILED;
    }

    int64_t rank = static_cast<int64_t>(gShape.GetDimNum());
    const char *layoutAttr = context->GetAttrs()->GetAttrPointer<char>(ATTR_LAYOUT_IDX);
    if (layoutAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    KdaGateLayout layout;
    if (std::strcmp(layoutAttr, "BSND") == 0) {
        layout = KdaGateLayout::BSND;
    } else if (std::strcmp(layoutAttr, "BNSD") == 0) {
        layout = KdaGateLayout::BNSD;
    } else if (std::strcmp(layoutAttr, "TND") == 0) {
        layout = KdaGateLayout::TND;
    } else if (std::strcmp(layoutAttr, "NTD") == 0) {
        layout = KdaGateLayout::NTD;
    } else {
        return ge::GRAPH_FAILED;
    }
    if ((rank == 4 && (layout == KdaGateLayout::TND || layout == KdaGateLayout::NTD)) ||
        (rank == 3 && (layout == KdaGateLayout::BSND || layout == KdaGateLayout::BNSD))) {
        return ge::GRAPH_FAILED;
    }
    int64_t batch = (rank == 4) ? gShape.GetDim(0) : 1;
    int64_t t = (layout == KdaGateLayout::BNSD) ? gShape.GetDim(2) :
                ((layout == KdaGateLayout::NTD) ? gShape.GetDim(1) :
                 ((rank == 4) ? gShape.GetDim(1) : gShape.GetDim(0)));
    int64_t hv = (layout == KdaGateLayout::BNSD) ? gShape.GetDim(1) :
                 ((layout == KdaGateLayout::NTD) ? gShape.GetDim(0) :
                  ((rank == 4) ? gShape.GetDim(2) : gShape.GetDim(1)));
    int64_t k = (rank == 4) ? gShape.GetDim(3) : gShape.GetDim(2);
    if (k > MAX_K_DIM) {
        return ge::GRAPH_FAILED;
    }

    int64_t chunkSize = *context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX);
    bool useGate = *context->GetAttrs()->GetAttrPointer<bool>(ATTR_USE_GATE_IDX);
    bool safeGate = *context->GetAttrs()->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX);
    float lowerBound = *context->GetAttrs()->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX);

    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    int64_t hasCuSeqlens = (cuShape != nullptr) ? 1 : 0;
    int64_t seqNum = hasCuSeqlens ? (cuShape->GetStorageShape().GetDim(0) - 1) : batch;
    int64_t maxChunks = (t + chunkSize - 1) / chunkSize;
    // Dense input keeps chunk-level parallelism. Varlen owns one (sequence, head) pair and
    // iterates only that sequence's real chunks, avoiding a rectangular grid of empty tasks.
    int64_t taskCount = hasCuSeqlens ? seqNum * hv : batch * hv * maxChunks;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskCount, coreNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.set_batch(batch);
    tiling.set_t(t);
    tiling.set_hv(hv);
    tiling.set_k(k);
    tiling.set_rank(rank);
    tiling.set_layout(static_cast<int64_t>(layout));
    tiling.set_chunkSize(chunkSize);
    tiling.set_seqNum(seqNum);
    tiling.set_hasCuSeqlens(hasCuSeqlens);
    tiling.set_hasALog(context->GetOptionalInputDesc(INPUT_A_LOG_IDX) != nullptr ? 1 : 0);
    tiling.set_hasDtBias(context->GetOptionalInputDesc(INPUT_DT_BIAS_IDX) != nullptr ? 1 : 0);
    int64_t dataType = 0;
    if (gDesc->GetDataType() == ge::DT_FLOAT) {
        dataType = 2;
    } else if (gDesc->GetDataType() == ge::DT_BF16) {
        dataType = 1;
    }
    tiling.set_dataType(dataType);
    tiling.set_useGateInKernel(useGate ? 1 : 0);
    tiling.set_safeGate(safeGate ? 1 : 0);
    tiling.set_lowerBound(lowerBound);
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KdaGateCumsum(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KdaGateCumsum)
    .Tiling(Tiling4KdaGateCumsum)
    .TilingParse<KdaGateCumsumCompileInfo>(TilingPrepare4KdaGateCumsum);

} // namespace optiling
