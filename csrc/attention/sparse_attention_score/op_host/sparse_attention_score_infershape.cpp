/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t KEY_INDEX = 1;
static constexpr uint32_t VALUE_INDEX = 2;
static constexpr uint32_t SELECT_IDX_INDEX = 3;
static constexpr uint32_t BLOCK_TABLE_INDEX = 4;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

static constexpr uint32_t TND_DIM_T = 0;
static constexpr uint32_t TND_DIM_N = 1;
static constexpr uint32_t TND_DIM_D = 2;
static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr uint32_t LSE_DIM_D = 1;

static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeSparseAttentionScore(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("SparseAttentionScore", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Enter SparseAttentionScore InferShape impl.");

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    const gert::Shape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    if (queryShape->GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "SparseAttentionScore only supports TND layout, queryDims(%zu) must be 3!",
                queryShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    *attentionOutShape = *queryShape;

    softmaxLseShape->SetDimNum(TND_DIM_NUM);
    (*softmaxLseShape)[TND_DIM_T] = queryShape->GetDim(TND_DIM_T);
    (*softmaxLseShape)[TND_DIM_N] = queryShape->GetDim(TND_DIM_N);
    (*softmaxLseShape)[TND_DIM_D] = LSE_DIM_D;

    OP_LOGD(context->GetNodeName(), "SparseAttentionScore InferShape success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, ge::DT_FLOAT16);
    } else {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    }
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseAttentionScore)
    .InferShape(InferShapeSparseAttentionScore)
    .InferDataType(InferDataTypeSparseAttentionScore);

}  // namespace ops
