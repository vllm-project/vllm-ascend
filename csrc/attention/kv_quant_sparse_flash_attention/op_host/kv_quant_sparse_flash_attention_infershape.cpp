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
 * \file kv_quant_sparse_flash_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

#ifdef OP_LOGE_WITH_INVALID_INPUT
#undef OP_LOGE_WITH_INVALID_INPUT
#endif
#define OP_LOGE_WITH_INVALID_INPUT(opname, param) \
    OP_LOGE(opname, "Invalid input: %s.", param)

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 13;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t OUTPUT_INDEX_0 = 0;
constexpr uint32_t OUTPUT_INDEX_1 = 1;
constexpr uint32_t OUTPUT_INDEX_2 = 2;

ge::graphStatus InferShapeKvQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("KvQuantSparseFlashAttention", "InferShapeContext"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    gert::Shape *softmaxMaxShape = context->GetOutputShape(OUTPUT_INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    gert::Shape *softmaxSumShape = context->GetOutputShape(OUTPUT_INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const char *inputLayoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKvPtr);
    std::string inputLayoutKvPtrStr = std::string(inputLayoutKvPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lse_flag);
    bool return_softmax_lse = (lse_flag != nullptr) ? *lse_flag : false;

    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(DIM_NUM_4);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2)); // 2:dim2
        if(queryShape->GetDim(DIM_INDEX_3) != -1){
            attentionOutShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_3) - ropeHeadDim); // 3:dim3
        }
    } else { // TND
        attentionOutShape->SetDimNum(DIM_NUM_3);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        if(queryShape->GetDim(DIM_INDEX_2) != -1){
            attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2) - ropeHeadDim); // 2:dim2
        }
    }

    if (return_softmax_lse) {
        if (queryShape->GetDimNum() == DIM_NUM_3) {
            if (inputLayoutKvPtrStr == "PA_BSND") {
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

ge::graphStatus InferDataTypeKvQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("KvQuantSparseFlashAttention", "InferShapeContext"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX_0, inputDataType);
    context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttention)
    .InferShape(InferShapeKvQuantSparseFlashAttention)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttention);
} // namespace ops
