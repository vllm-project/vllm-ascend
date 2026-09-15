/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;

ge::graphStatus InferShapeVllmFusedInferAttentionScore(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    if (softmaxLseShape != nullptr) {
        OPS_ERR_IF(queryShape->GetDimNum() != 3,
                   OPS_LOG_E("VllmFusedInferAttentionScore", "query must be 3D for TND layout"),
                   return ge::GRAPH_FAILED);
        softmaxLseShape->SetDimNum(3);
        softmaxLseShape->SetDim(0, queryShape->GetDim(0));
        softmaxLseShape->SetDim(1, queryShape->GetDim(1));
        softmaxLseShape->SetDim(2, 1);
    }

    gert::Shape *sparseStatsShape = context->GetOutputShape(2);
    if (sparseStatsShape != nullptr) {
        sparseStatsShape->SetDimNum(1);
        sparseStatsShape->SetDim(0, 16);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeVllmFusedInferAttentionScore(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("VllmFusedInferAttentionScore", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(VllmFusedInferAttentionScore)
    .InferShape(InferShapeVllmFusedInferAttentionScore)
    .InferDataType(InferDataTypeVllmFusedInferAttentionScore);
} // namespace ops
