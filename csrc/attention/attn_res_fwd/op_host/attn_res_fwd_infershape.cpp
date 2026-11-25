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
 * \file attn_res_fwd_infershape.cpp
 * \brief AttnResFwd infer shape implementation
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

constexpr size_t PREFIX_SUM_INDEX = 0;
constexpr size_t BLOCK_RESIDUAL_INDEX = 1;
constexpr size_t PROJ_WEIGHT_INDEX = 2;
constexpr size_t NORM_WEIGHT_INDEX = 3;
constexpr size_t HIDDEN_STATES_INDEX = 0;
constexpr size_t INV_RMS_INDEX = 1;
constexpr size_t PROBS_INDEX = 2;

constexpr size_t PREFIX_SUM_DIM = 2;
constexpr size_t BLOCK_RESIDUAL_DIM = 3;
constexpr size_t WEIGHT_1D_DIM = 1;
constexpr size_t WEIGHT_2D_DIM = 2;
constexpr size_t ATTR_NEED_BACKWARD_INDEX = 1;

static ge::graphStatus InferShapeAttnResFwd(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("AttnResFwd", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto prefixShape = context->GetInputShape(PREFIX_SUM_INDEX);
    auto blockShape = context->GetInputShape(BLOCK_RESIDUAL_INDEX);
    auto projShape = context->GetInputShape(PROJ_WEIGHT_INDEX);
    auto normShape = context->GetInputShape(NORM_WEIGHT_INDEX);
    auto outShape = context->GetOutputShape(HIDDEN_STATES_INDEX);
    if (prefixShape == nullptr || blockShape == nullptr || projShape == nullptr || normShape == nullptr ||
        outShape == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(prefixShape->GetDimNum() != PREFIX_SUM_DIM,
                OP_LOGE(opName, "prefix_sum dim num should be %zu, got %zu", PREFIX_SUM_DIM, prefixShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape->GetDimNum() != BLOCK_RESIDUAL_DIM,
                OP_LOGE(opName, "block_residual dim num should be %zu, got %zu", BLOCK_RESIDUAL_DIM,
                        blockShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(projShape->GetDimNum() != WEIGHT_1D_DIM && projShape->GetDimNum() != WEIGHT_2D_DIM,
                OP_LOGE(opName, "proj_weight dim num should be 1 or 2, got %zu", projShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape->GetDimNum() != WEIGHT_1D_DIM,
                OP_LOGE(opName, "norm_weight dim num should be %zu, got %zu", WEIGHT_1D_DIM, normShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = prefixShape->GetDim(0);
    const int64_t hiddenSize = prefixShape->GetDim(1);
    const int64_t numBlocks = blockShape->GetDim(1);
    const int64_t blockCount = numBlocks + 1;

    outShape->SetDimNum(PREFIX_SUM_DIM);
    outShape->SetDim(0, numTokens);
    outShape->SetDim(1, hiddenSize);

    bool needBackward = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
        if (needBackwardPtr != nullptr) {
            needBackward = *needBackwardPtr;
        }
    }

    if (needBackward) {
        auto invRmsShape = context->GetOutputShape(INV_RMS_INDEX);
        auto probsShape = context->GetOutputShape(PROBS_INDEX);
        if (invRmsShape != nullptr) {
            invRmsShape->SetDimNum(PREFIX_SUM_DIM);
            invRmsShape->SetDim(0, numTokens);
            invRmsShape->SetDim(1, blockCount);
        }
        if (probsShape != nullptr) {
            probsShape->SetDimNum(PREFIX_SUM_DIM);
            probsShape->SetDim(0, numTokens);
            probsShape->SetDim(1, blockCount);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAttnResFwd(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("AttnResFwd", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    auto prefixDtype = context->GetInputDataType(PREFIX_SUM_INDEX);
    context->SetOutputDataType(HIDDEN_STATES_INDEX, prefixDtype);

    bool needBackward = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
        if (needBackwardPtr != nullptr) {
            needBackward = *needBackwardPtr;
        }
    }
    if (needBackward) {
        context->SetOutputDataType(INV_RMS_INDEX, ge::DT_FLOAT);
        context->SetOutputDataType(PROBS_INDEX, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttnResFwd)
    .InferShape(InferShapeAttnResFwd)
    .InferDataType(InferDataTypeAttnResFwd);
} // namespace ops
