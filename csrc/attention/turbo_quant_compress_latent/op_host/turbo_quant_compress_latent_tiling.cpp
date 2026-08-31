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
 * \file turbo_quant_compress_latent_tiling.cpp
 * \brief
 */
#include "turbo_quant_compress_latent_tiling.h"

namespace optiling {
constexpr size_t TQ_COMPRESS_LATENT_DIM_NUM = 2;
constexpr size_t TQ_COMPRESS_DIM_TOKEN = 0;
constexpr size_t TQ_COMPRESS_DIM_HEAD = 1;

static ge::graphStatus TilingFuncForTurboQuantCompressLatent(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin to do TilingForTurboQuantCompressLatent");
    const gert::StorageShape* latentShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    const gert::StorageShape* centroidsShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, centroidsShape);
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outputMode = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMode);
    if (*outputMode != TQ_COMPRESS_OUTPUT_PADDED && *outputMode != TQ_COMPRESS_OUTPUT_COMPACT_CORRECTED) {
        OP_LOGE(context->GetNodeName(), "output_mode only supports 0 or 1, but got %ld", *outputMode);
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& latent = latentShape->GetStorageShape();
    if (latent.GetDimNum() != TQ_COMPRESS_LATENT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "latent must be 2-dimensional [numTokens, headDim], but got %zu-d",
                latent.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t numTokens = latent.GetDim(TQ_COMPRESS_DIM_TOKEN);
    int64_t headDim = latent.GetDim(TQ_COMPRESS_DIM_HEAD);
    if (numTokens < 0) {
        OP_LOGE(context->GetNodeName(), "numTokens must be non-negative, but got %ld", numTokens);
        return ge::GRAPH_FAILED;
    }
    if (headDim != TQ_COMPRESS_SUPPORTED_HEAD_DIM) {
        OP_LOGE(context->GetNodeName(), "headDim only supports %ld for now, but got %ld",
                TQ_COMPRESS_SUPPORTED_HEAD_DIM, headDim);
        return ge::GRAPH_FAILED;
    }

    int64_t centCount = centroidsShape->GetStorageShape().GetShapeSize();
    if (centCount != TQ_COMPRESS_N_CENT) {
        OP_LOGE(context->GetNodeName(), "centroids must hold exactly %ld elements, but got %ld", TQ_COMPRESS_N_CENT,
                centCount);
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum < 1) {
        coreNum = 1;
    }

    // A batch of tokens is fully resident in UB, so the split across cores is purely over tokens.
    uint32_t tokens = static_cast<uint32_t>(numTokens < 1 ? 1 : numTokens);
    uint32_t tokensPerCore = (tokens + coreNum - 1) / coreNum;
    uint32_t blockDim = (tokens + tokensPerCore - 1) / tokensPerCore;

    int64_t slotSize = TqCompressOutputSlotSize(headDim, *outputMode);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t budget = static_cast<int64_t>(ubSize) - TqCompressFixedBytes(headDim) - TQ_COMPRESS_UB_RESERVE;
    int64_t bytesPerToken = TqCompressBytesPerToken(headDim, slotSize);
    int64_t maxByUb = (budget > 0 && bytesPerToken > 0) ? budget / bytesPerToken : 1;

    // Batching only pays off up to what a core actually owns; below that it would idle cores instead.
    int64_t tokensPerBatch = static_cast<int64_t>(tokensPerCore);
    if (tokensPerBatch > maxByUb) {
        tokensPerBatch = maxByUb;
    }
    if (tokensPerBatch > TQ_COMPRESS_MAX_TOKENS_PER_BATCH) {
        tokensPerBatch = TQ_COMPRESS_MAX_TOKENS_PER_BATCH;
    }
    if (tokensPerBatch < 1) {
        tokensPerBatch = 1;
    }

    TurboQuantCompressLatentTilingData tilingData;
    tilingData.set_numTokens(static_cast<uint32_t>(numTokens));
    tilingData.set_tokensPerCore(tokensPerCore);
    tilingData.set_headDim(static_cast<uint32_t>(headDim));
    tilingData.set_slotSize(static_cast<uint32_t>(slotSize));
    tilingData.set_tokensPerBatch(static_cast<uint32_t>(tokensPerBatch));
    tilingData.set_outputMode(static_cast<uint32_t>(*outputMode));

    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        OP_LOGE(context->GetNodeName(), "tiling data size %zu exceeds capacity %zu", tilingData.GetDataSize(),
                context->GetRawTilingData()->GetCapacity());
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGD(context, "End to do TilingForTurboQuantCompressLatent");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForTurboQuantCompressLatent(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantCompressLatent)
    .Tiling(TilingFuncForTurboQuantCompressLatent)
    .TilingParse<TurboQuantCompressLatentCompileInfo>(TilingPrepareForTurboQuantCompressLatent);
} // namespace optiling
