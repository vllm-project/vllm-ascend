/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t HIDDEN_STATES_INPUT_INDEX = 0;
constexpr uint32_t WK_INPUT_INDEX = 1;
constexpr uint32_t GATE_WEIGHT_INPUT_INDEX = 2;

constexpr uint32_t APE_INPUT_INDEX = 3;
constexpr uint32_t STATE_CACHE_INPUT_INDEX = 4;
constexpr uint32_t CACHE_BLOCK_TABLE_INPUT_INDEX = 5;
constexpr uint32_t START_POS_INPUT_INDEX = 6;
constexpr uint32_t CU_SEQLENS_INPUT_INDEX = 11;

// ATTR
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
// OUTPUT
constexpr uint32_t POOLED_KEY_OUTPUT_INDEX = 0;

// ATTR DEFAULT VALUE
constexpr uint32_t CMP_RATIO_VALUE = 4;

struct KeyPoolProtoShapeParam {
    bool isBsMerge{false};
    int64_t B{0};
    int64_t Sr{0};
    int64_t D{0};
};

constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;

ge::graphStatus GetKeyPoolShapeDim(const gert::InferShapeContext *context, KeyPoolProtoShapeParam &shapeParam)
{
    auto hiddenStatesShape = context->GetRequiredInputShape(HIDDEN_STATES_INPUT_INDEX); // (B, S, H) | (T, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenStatesShape);
    auto wkShape = context->GetRequiredInputShape(WK_INPUT_INDEX); // (D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wkShape);
    auto gateWeightShape = context->GetRequiredInputShape(GATE_WEIGHT_INPUT_INDEX); // (D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, gateWeightShape);
    // (block_num, block_size, 2 * D)
    auto stateCacheShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateCacheShape);

    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX); // (r, D)
    OP_CHECK_NULL_WITH_CONTEXT(context, apeShape);

    auto cacheBlockTableShape = context->GetRequiredInputShape(CACHE_BLOCK_TABLE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheBlockTableShape);
    if (hiddenStatesShape->GetDimNum() == DIM_NUM_2) {
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQLENS_INPUT_INDEX); // (B+1,)
        OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    }
    auto startPosShape = context->GetRequiredInputShape(START_POS_INPUT_INDEX); // (B,)
    OP_CHECK_NULL_WITH_CONTEXT(context, startPosShape);

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    OP_CHECK_IF(cmpRatio == 0, OP_LOGE(context->GetNodeName(), "cmp_ratio must not be zero"), return ge::GRAPH_FAILED);
    if (hiddenStatesShape->GetDimNum() == DIM_NUM_3) { // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = hiddenStatesShape->GetDim(DIM_INDEX_0);
        shapeParam.Sr = (hiddenStatesShape->GetDim(DIM_INDEX_1) + cmpRatio - 1) / cmpRatio;
    } else { // T
        shapeParam.isBsMerge = true;
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQLENS_INPUT_INDEX);
        shapeParam.B = cuSeqlensShape->GetDim(DIM_INDEX_0) - 1;
        const int64_t tokenSize = hiddenStatesShape->GetDim(DIM_INDEX_0);
        shapeParam.Sr = std::min(tokenSize, tokenSize / cmpRatio + shapeParam.B);
    }

    shapeParam.D = wkShape->GetDim(DIM_INDEX_0);

    return GRAPH_SUCCESS;
}

ge::graphStatus SetKeyPoolShapeDim(const KeyPoolProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto pooledKeyShape = context->GetOutputShape(POOLED_KEY_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledKeyShape);
    if (shapeParam.isBsMerge) {
        pooledKeyShape->SetDimNum(DIM_NUM_2);
        pooledKeyShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        pooledKeyShape->SetDim(DIM_INDEX_1, shapeParam.D);
    } else {
        pooledKeyShape->SetDimNum(DIM_NUM_3);
        pooledKeyShape->SetDim(DIM_INDEX_0, shapeParam.B);
        pooledKeyShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        pooledKeyShape->SetDim(DIM_INDEX_2, shapeParam.D);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKeyPool(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "Context is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter KeyPool inferDataType impl.");

    context->SetOutputDataType(POOLED_KEY_OUTPUT_INDEX, context->GetRequiredInputDataType(HIDDEN_STATES_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeKeyPool(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "Context is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter KeyPool infershape impl.");

    KeyPoolProtoShapeParam shapeParam{};
    auto apiRet = GetKeyPoolShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetKeyPoolShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KeyPool).InferShape(InferShapeKeyPool).InferDataType(InferDataTypeKeyPool);
} // namespace ops
