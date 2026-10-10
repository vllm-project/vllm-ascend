/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include <string>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t KEY_INDEX = 1;
static constexpr uint32_t VALUE_INDEX = 2;
static constexpr uint32_t BLOCK_TABLE_INDEX = 15;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// Keep attr indices in sync with OpDef / tiling.
static constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 1;
static constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 2;
static constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 11;

static constexpr uint32_t TND_DIM_T = 0;
static constexpr uint32_t TND_DIM_N = 1;
static constexpr uint32_t TND_DIM_D = 2;
static constexpr uint32_t TND_DIM_NUM = 3;

// PA_BBND: [blockNum, blockSize, Nkv, D]
static constexpr uint32_t PA_BBND_DIM_KV_HEAD = 2;
static constexpr uint32_t PA_BBND_DIM_D = 3;
static constexpr uint32_t PA_BBND_DIM_NUM = 4;
static constexpr uint32_t LSE_DIM_D = 1;

static constexpr int32_t UNKNOWN_DIMS = -2;

static bool IsUnknownRankShape(const gert::Shape *shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == UNKNOWN_DIMS;
}

static void SetUnknownOutShapes(gert::Shape *attentionOutShape, gert::Shape *softmaxLseShape)
{
    attentionOutShape->SetDimNum(1);
    (*attentionOutShape)[0] = UNKNOWN_DIMS;
    softmaxLseShape->SetDimNum(1);
    (*softmaxLseShape)[0] = UNKNOWN_DIMS;
}

static void SetEmptyLseShape(gert::Shape *softmaxLseShape)
{
    // return_softmax_lse=0 → placeholder {0}.
    softmaxLseShape->SetDimNum(1);
    (*softmaxLseShape)[0] = 0;
}

static ge::graphStatus InferShapeGenericBlockSparseAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("GenericBlockSparseAttention", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Enter GenericBlockSparseAttention InferShape impl.");

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

    // Any of Q/K/V unknown-rank → propagate unknown outs.
    if (IsUnknownRankShape(queryShape) || IsUnknownRankShape(keyShape) || IsUnknownRankShape(valueShape)) {
        SetUnknownOutShapes(attentionOutShape, softmaxLseShape);
        return ge::GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
    const char *layoutKvPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_KV_INDEX);
    const std::string layoutQ = (layoutQPtr != nullptr) ? layoutQPtr : "TND";
    const std::string layoutKv = (layoutKvPtr != nullptr) ? layoutKvPtr : "TND";

    // Layout shape checks are independent of the SoC capability checks in tiling.
    const bool isPaged = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    if (layoutQ != "TND" || (isPaged ? layoutKv != "PA_BBND" : layoutKv != "TND")) {
        OP_LOGE(context->GetNodeName(),
                "Q must be TND; KV must be PA_BBND with blockTable, or TND without blockTable.");
        return ge::GRAPH_FAILED;
    }
    if (queryShape->GetDimNum() != TND_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "TND query must have rank 3.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t kvRank = isPaged ? PA_BBND_DIM_NUM : TND_DIM_NUM;
    const uint32_t kvHeadAxis = isPaged ? PA_BBND_DIM_KV_HEAD : TND_DIM_N;
    const uint32_t kvDAxis = isPaged ? PA_BBND_DIM_D : TND_DIM_D;
    if (keyShape->GetDimNum() != kvRank || valueShape->GetDimNum() != kvRank) {
        OP_LOGE(context->GetNodeName(), "KV layout %s requires rank %u.", layoutKv.c_str(), kvRank);
        return ge::GRAPH_FAILED;
    }
    for (uint32_t axis = 0; axis < kvRank; ++axis) {
        if (keyShape->GetDim(axis) != valueShape->GetDim(axis)) {
            OP_LOGE(context->GetNodeName(), "Key/value shapes must match.");
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t qHeads = queryShape->GetDim(TND_DIM_N);
    const int64_t kvHeads = keyShape->GetDim(kvHeadAxis);
    if (keyShape->GetDim(kvDAxis) != queryShape->GetDim(TND_DIM_D) ||
        (kvHeads >= 0 && (kvHeads == 0 || (qHeads >= 0 && qHeads % kvHeads != 0)))) {
        OP_LOGE(context->GetNodeName(), "Q/KV head dimensions or head counts do not match.");
        return ge::GRAPH_FAILED;
    }
    *attentionOutShape = *queryShape;

    const int64_t *lseFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_SOFTMAX_LSE_INDEX);
    const int64_t returnSoftmaxLse = (lseFlagPtr != nullptr) ? *lseFlagPtr : 0;
    if (returnSoftmaxLse != 0 && returnSoftmaxLse != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported return_softmax_lse=%ld, only 0 or 1 are supported.",
                returnSoftmaxLse);
        return ge::GRAPH_FAILED;
    }

    if (returnSoftmaxLse == 1) {
        softmaxLseShape->SetDimNum(TND_DIM_NUM);
        (*softmaxLseShape)[TND_DIM_T] = queryShape->GetDim(TND_DIM_T);
        (*softmaxLseShape)[TND_DIM_N] = queryShape->GetDim(TND_DIM_N);
        (*softmaxLseShape)[TND_DIM_D] = LSE_DIM_D;
    } else {
        SetEmptyLseShape(softmaxLseShape);
    }

    OP_LOGD(context->GetNodeName(), "GenericBlockSparseAttention InferShape success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeGenericBlockSparseAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        // FP8 input: derive output dtype from user-provided attentionOut tensor.
        dtype = context->GetOutputDataType(ATTENTION_OUT_INDEX);
        if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
            OP_LOGE(context->GetNodeName(),
                    "When query is float8_e4m3fn, attentionOut dtype must be float16 or bfloat16, "
                    "but got %d.",
                    static_cast<int32_t>(dtype));
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GenericBlockSparseAttention)
    .InferShape(InferShapeGenericBlockSparseAttention)
    .InferDataType(InferDataTypeGenericBlockSparseAttention);

} // namespace ops
