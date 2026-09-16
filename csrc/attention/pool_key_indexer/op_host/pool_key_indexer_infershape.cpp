/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t POOL_KEY_INDEX = 1;
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_POOL_SIZE_INDEX = 1;
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 2;
constexpr uint32_t ATTR_LAYOUT_K_INDEX = 3;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 5;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;

// topk / poolSize constraints (ref aclnn doc)
constexpr int64_t TOPK_MIN = 1;
constexpr int64_t TOPK_MAX_CONTINUOUS = 2048;
constexpr int64_t POOL_SIZE_MIN = 1;
constexpr int64_t POOL_SIZE_MAX = 128;

static ge::graphStatus InferShapePoolKeyIndexer(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PoolKeyIndexer", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(POOL_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQPtr);
    const char *layoutKPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKPtr);
    const int64_t *topkPtr = attrs->GetInt(ATTR_TOPK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkPtr);
    const int64_t *poolSizePtr = attrs->GetInt(ATTR_POOL_SIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolSizePtr);
    const int64_t *maskModePtr = attrs->GetInt(ATTR_MASK_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskModePtr);
    const int64_t *quantModePtr = attrs->GetInt(ATTR_QUANT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    const bool *returnValuePtr = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnValuePtr);

    std::string layoutQ(layoutQPtr);
    std::string layoutK(layoutKPtr);
    int64_t topk = *topkPtr;
    int64_t poolSize = *poolSizePtr;
    int64_t maskMode = *maskModePtr;
    int64_t quantMode = *quantModePtr;
    bool returnValue = *returnValuePtr;

    OP_CHECK_IF(layoutQ != "BSND" && layoutQ != "TND",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "layout_q", layoutQ.c_str(),
                                                      "layout_q should be BSND or TND"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(layoutK != "BSND" && layoutK != "TND" && layoutK != "PA_BBND",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "layout_k", layoutK.c_str(),
                                                      "layout_k should be BSND, TND or PA_BBND"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(maskMode != 0 && maskMode != 3,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "mask_mode", std::to_string(maskMode).c_str(),
                                                      "mask_mode should be 0 or 3"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(quantMode != -1 && quantMode != 0 && quantMode != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "quant_mode", std::to_string(quantMode).c_str(),
                                                      "quant_mode should be -1, 0 or 1"),
                return ge::GRAPH_FAILED);

    // poolSize constraints: [1, 128]
    OP_CHECK_IF(poolSize < POOL_SIZE_MIN || poolSize > POOL_SIZE_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "pool_size", std::to_string(poolSize).c_str(),
                                                      "pool_size must be in [1, 128]"),
                return ge::GRAPH_FAILED);

    // topk constraints: [1, 2048] or {3072, 4096, 5120, 6144, 7168, 8192}
    bool topkValid = (topk >= TOPK_MIN && topk <= TOPK_MAX_CONTINUOUS) || topk == 3072 || topk == 4096 ||
                     topk == 5120 || topk == 6144 || topk == 7168 || topk == 8192;
    OP_CHECK_IF(!topkValid,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "PoolKeyIndexer", "topk", std::to_string(topk).c_str(),
                    "topk must be in [1, 2048] or one of {3072, 4096, 5120, 6144, 7168, 8192}"),
                return ge::GRAPH_FAILED);

    // topk must be divisible by poolSize
    OP_CHECK_IF(poolSize == 0 || topk % poolSize != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "topk", std::to_string(topk).c_str(),
                                                      "topk must be divisible by pool_size"),
                return ge::GRAPH_FAILED);

    int64_t outLastDim = topk + poolSize - 1;
    int64_t valuesLastDim = topk / poolSize;

    if (layoutQ == "BSND") {
        OP_CHECK_IF(queryShape->GetDimNum() != 4,
                    OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDimNum(3);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1));
        sparseIndicesShape->SetDim(2, outLastDim);
    } else {
        OP_CHECK_IF(queryShape->GetDimNum() != 3,
                    OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDimNum(2);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));
        sparseIndicesShape->SetDim(1, outLastDim);
    }

    if (returnValue) {
        if (layoutQ == "BSND") {
            sparseValuesShape->SetDimNum(3);
            sparseValuesShape->SetDim(0, queryShape->GetDim(0));
            sparseValuesShape->SetDim(1, queryShape->GetDim(1));
            sparseValuesShape->SetDim(2, valuesLastDim);
        } else {
            sparseValuesShape->SetDimNum(2);
            sparseValuesShape->SetDim(0, queryShape->GetDim(0));
            sparseValuesShape->SetDim(1, valuesLastDim);
        }
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }

    OP_LOGI(context->GetNodeName(), "PoolKeyIndexer InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePoolKeyIndexer(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PoolKeyIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGI(context->GetNodeName(), "PoolKeyIndexer InferDataType end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PoolKeyIndexer).InferShape(InferShapePoolKeyIndexer).InferDataType(InferDataTypePoolKeyIndexer);
} // namespace ops
