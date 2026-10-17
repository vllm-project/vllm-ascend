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
 * \file sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

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

static std::vector<int64_t> ToVectorFunc(const gert::Shape *shape)
{
    size_t shapeSize = shape->GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape->GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringFunc(const gert::Shape *shape)
{
    std::ostringstream oss;
    auto v = ToVectorFunc(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

static int64_t GetKvHeadNum(const gert::Shape *kvShape, const std::string &layoutKv)
{
    if (layoutKv == "TND") {
        return kvShape->GetDim(DIM_INDEX_1);
    }
    return kvShape->GetDim(DIM_INDEX_2);
}

const gert::Shape *GetOptionalStorageShape(const gert::InferShapeContext *context, uint32_t inputIndex)
{
    return context->GetOptionalInputShape(inputIndex);
}

ge::graphStatus InferShapeSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMla", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *oriKvShape = GetOptionalStorageShape(context, ORI_KV_INPUT_INDEX);
    const gert::Shape *cmpKvShape = GetOptionalStorageShape(context, CMP_KV_INPUT_INDEX);
    const gert::Shape *kvShape = (oriKvShape != nullptr) ? oriKvShape : cmpKvShape;
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_INDEX);
    const char *layoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKvPtr);
    std::string layoutKv = std::string(layoutKvPtr);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    int64_t kvHeadNum = GetKvHeadNum(kvShape, layoutKv);

    OP_CHECK_IF(kvHeadNum <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    "SparseFlashMla", "ori_kv or cmp_kv", ToStringFunc(kvShape).c_str(),
                    "The head num of ori_kv or cmp_kv should be greater than 0 but got " + std::to_string(kvHeadNum)),
                return ge::GRAPH_FAILED);

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
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashMla", "InferShapeContext is nullptr"), return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashMla).InferShape(InferShapeSparseFlashMla).InferDataType(InferDataTypeSparseFlashMla);
} // namespace ops
