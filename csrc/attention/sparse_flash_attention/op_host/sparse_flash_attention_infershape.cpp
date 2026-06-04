/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;

constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t LAYOUT_KEY_ATTR_INDEX = 3;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 8;

constexpr uint32_t OUTPUT_INDEX_0 = 0;
constexpr uint32_t OUTPUT_INDEX_1 = 1;
constexpr uint32_t OUTPUT_INDEX_2 = 2;

ge::graphStatus InferShapeSparseFlashAttention(gert::InferShapeContext *context)
{  
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);

    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(LAYOUT_KEY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag);
    bool return_softmax_lse = (lse_flag != nullptr)? *lse_flag : false;

    if(return_softmax_lse){
        if(queryShape->GetDimNum() == DIM_NUM_3){
            if (inputLayoutKeyPtrStr == "PA_BSND") {
                softmaxMaxShape->SetDimNum(DIM_NUM_3);
                softmaxMaxShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_2));
                softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_2));

                softmaxSumShape->SetDimNum(DIM_NUM_3);
                softmaxSumShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_2));
                softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_2));
            } else {
                softmaxMaxShape->SetDimNum(DIM_NUM_3);
                softmaxMaxShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_1));
                softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_1));

                softmaxSumShape->SetDimNum(DIM_NUM_3);
                softmaxSumShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_1));
                softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_1));
            }
        } else {
            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_1, keyShape->GetDim(DIM_INDEX_2));
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxMaxShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyShape->GetDim(DIM_INDEX_2));

            softmaxSumShape->SetDimNum(DIM_NUM_4);
            softmaxSumShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_1, keyShape->GetDim(DIM_INDEX_2));
            softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxSumShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyShape->GetDim(DIM_INDEX_2));
        }
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(DIM_INDEX_0, 0);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SparseFlashAttention", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX_0, inputDataType);
    context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashAttention)
    .InferShape(InferShapeSparseFlashAttention)
    .InferDataType(InferDataTypeSparseFlashAttention);
} // namespace ops
  
