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
 * \file turbo_quant_compress_latent_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "turbo_quant_compress_latent_tiling.h"

using namespace ge;
namespace ops {
constexpr size_t TQ_COMPRESS_LATENT_DIM_NUM = 2;
constexpr size_t TQ_COMPRESS_DIM_TOKEN = 0;
constexpr size_t TQ_COMPRESS_DIM_HEAD = 1;
constexpr int64_t TQ_COMPRESS_UNKNOWN_DIM = -1;

static ge::graphStatus InferShapeForTurboQuantCompressLatent(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForTurboQuantCompressLatent");
    const gert::Shape* latentShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, latentShape);
    gert::Shape* slotShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotShape);

    if (Ops::Base::IsUnknownRank(*latentShape)) {
        OP_LOGD(context, "latent shape is UnknownRank, set slot shape to (-2, )");
        Ops::Base::SetUnknownRank(*slotShape);
        return ge::GRAPH_SUCCESS;
    }

    if (latentShape->GetDimNum() != TQ_COMPRESS_LATENT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "latent must be 2-dimensional [numTokens, headDim], but got %zu-d",
                latentShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t numTokens = latentShape->GetDim(TQ_COMPRESS_DIM_TOKEN);
    int64_t headDim = latentShape->GetDim(TQ_COMPRESS_DIM_HEAD);
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outputMode = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputMode);
    if (*outputMode != optiling::TQ_COMPRESS_OUTPUT_PADDED &&
        *outputMode != optiling::TQ_COMPRESS_OUTPUT_COMPACT_CORRECTED) {
        OP_LOGE(context->GetNodeName(), "output_mode only supports 0 or 1, but got %ld", *outputMode);
        return ge::GRAPH_FAILED;
    }

    slotShape->SetDimNum(TQ_COMPRESS_LATENT_DIM_NUM);
    slotShape->SetDim(TQ_COMPRESS_DIM_TOKEN, numTokens);
    if (headDim == TQ_COMPRESS_UNKNOWN_DIM) {
        slotShape->SetDim(TQ_COMPRESS_DIM_HEAD, TQ_COMPRESS_UNKNOWN_DIM);
    } else {
        slotShape->SetDim(TQ_COMPRESS_DIM_HEAD, optiling::TqCompressOutputSlotSize(headDim, *outputMode));
    }

    OP_LOGD(context, "End to do InferShapeForTurboQuantCompressLatent");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForTurboQuantCompressLatent(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeForTurboQuantCompressLatent");
    context->SetOutputDataType(0, ge::DT_UINT8);
    OP_LOGD(context, "End to do InferDataTypeForTurboQuantCompressLatent");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantCompressLatent)
    .InferShape(InferShapeForTurboQuantCompressLatent)
    .InferDataType(InferDataTypeForTurboQuantCompressLatent);
} // namespace ops
