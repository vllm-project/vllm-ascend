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
* \file sparse_flash_mla_proto.cpp
* \brief
*/

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t ORI_KV_INPUT_INDEX = 1;
constexpr uint32_t CMP_KV_INPUT_INDEX = 2;
constexpr uint32_t RETURN_SOFTMAX_INDEX = 9;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 7;

int64_t GetKvHeadNum(const gert::Shape *kvShape, const std::string &layoutKv)
{
    if (layoutKv == "TND") {
        return kvShape->GetDim(DIM_INDEX_1);
    }
    return kvShape->GetDim(DIM_INDEX_2);
}

const gert::Shape *GetOptionalStorageShape(gert::InferShapeContext *context, uint32_t inputIndex)
{
    const gert::StorageShape *storageShape = context->GetOptionalInputShape(inputIndex);
    if (storageShape == nullptr) {
        return nullptr;
    }
    return &storageShape->GetStorageShape();
}

ge::graphStatus InferShapeSparseFlashMla(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashMla", "InferShapeContext is nullptr"),
            return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    const gert::Shape *oriKvShape = GetOptionalStorageShape(context, ORI_KV_INPUT_INDEX);
    const gert::Shape *cmpKvShape = GetOptionalStorageShape(context, CMP_KV_INPUT_INDEX);
    const gert::Shape *kvShape = (oriKvShape != nullptr) ? oriKvShape : cmpKvShape;
    OPS_LOG_E_IF_NULL(context, kvShape, return ge::GRAPH_FAILED)

    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, softmaxLseShape, return ge::GRAPH_FAILED)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_INDEX);
    const char *layoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKvPtr);
    std::string layoutKv = std::string(layoutKvPtr);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    int64_t kvHeadNum = GetKvHeadNum(kvShape, layoutKv);

    if (returnSoftmaxLse) {
        if (queryShape->GetDimNum() == DIM_NUM_3) {
            softmaxLseShape->SetDimNum(DIM_NUM_3);
            softmaxLseShape->SetDim(DIM_INDEX_0, kvHeadNum);
            softmaxLseShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
            softmaxLseShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / kvHeadNum);
        } else {
            softmaxLseShape->SetDimNum(DIM_NUM_4);
            softmaxLseShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxLseShape->SetDim(DIM_INDEX_1, kvHeadNum);
            softmaxLseShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxLseShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / kvHeadNum);
        }
    } else {
        softmaxLseShape->SetDimNum(DIM_NUM_1);
        softmaxLseShape->SetDim(DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashMla(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashMla", "InferShapeContext is nullptr"),
            return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashMla).InferShape(InferShapeSparseFlashMla).InferDataType(InferDataTypeSparseFlashMla);
} // namespace ops
