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
 * \file kv_quant_sparse_flash_attention_cann_v2_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
// Inputs Index
constexpr size_t QSFA_QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
// Attributes Index
constexpr uint32_t QSFA_LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t QSFA_ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 13;
// Dim Index
constexpr uint32_t QSFA_DIM_INDEX_0 = 0;
constexpr uint32_t QSFA_DIM_INDEX_1 = 1;
constexpr uint32_t QSFA_DIM_INDEX_2 = 2;
constexpr uint32_t QSFA_DIM_INDEX_3 = 3;
constexpr uint32_t QSFA_DIM_NUM_3 = 3;
constexpr uint32_t QSFA_DIM_NUM_4 = 4;
constexpr uint32_t DIM_NUM_1 = 1;

// 设置attention_out的shape: query shape去掉rope_head_dim维度
inline void SetQSFAAttentionOutShape(const gert::Shape *queryShape, gert::Shape *attentionOutShape,
                                     const std::string &inputLayoutQueryPtrStr, const int64_t ropeHeadDim)
{
    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(QSFA_DIM_NUM_4);
        attentionOutShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_1));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_2));
        if (queryShape->GetDim(QSFA_DIM_INDEX_3) != -1) {
            attentionOutShape->SetDim(QSFA_DIM_INDEX_3, queryShape->GetDim(QSFA_DIM_INDEX_3) - ropeHeadDim);
        }
    } else { // TND
        attentionOutShape->SetDimNum(QSFA_DIM_NUM_3);
        attentionOutShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
        attentionOutShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_1));
        if (queryShape->GetDim(QSFA_DIM_INDEX_2) != -1) {
            attentionOutShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_2) - ropeHeadDim);
        }
    }
}

ge::graphStatus InferShapeKvQuantSparseFlashAttentionCannV2(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvQuantSparseFlashAttentionCannV2", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QSFA_QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(QSFA_LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const char *inputLayoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKvPtr);
    std::string inputLayoutKvPtrStr = std::string(inputLayoutKvPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(QSFA_ROPE_HEAD_DIM_ATTR_INDEX);
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;

    SetQSFAAttentionOutShape(queryShape, attentionOutShape, inputLayoutQueryPtrStr, ropeHeadDim);

    if (returnSoftmaxLse) {
        int64_t kvHeadNum = 0;
        if (inputLayoutKvPtrStr == "TND") {
            kvHeadNum = keyShape->GetDim(QSFA_DIM_INDEX_1);
        } else {
            kvHeadNum = keyShape->GetDim(QSFA_DIM_INDEX_2);
        }
        OP_CHECK_IF(kvHeadNum == 0,
                    OP_LOGE(context->GetNodeName(), "kv head num of key must not be 0 when return_softmax_lse is true"),
                    return ge::GRAPH_FAILED);
        int64_t queryHeadNum = (inputLayoutQueryPtrStr == "BSND") ? queryShape->GetDim(QSFA_DIM_INDEX_2) :
                                                                    queryShape->GetDim(QSFA_DIM_INDEX_1);
        // 动态shape(-1)场景g保持-1, 正常场景整除计算group数
        int64_t g = -1;
        if (kvHeadNum > 0 && queryHeadNum != -1) {
            g = queryHeadNum / kvHeadNum;
        }
        if (inputLayoutQueryPtrStr == "BSND") {
            softmaxMaxShape->SetDimNum(QSFA_DIM_NUM_4);
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_1, kvHeadNum);
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_1));
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_3, g);

            softmaxSumShape->SetDimNum(QSFA_DIM_NUM_4);
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_0, queryShape->GetDim(QSFA_DIM_INDEX_0));
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_1, kvHeadNum);
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_2, queryShape->GetDim(QSFA_DIM_INDEX_1));
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_3, g);
        } else {
            softmaxMaxShape->SetDimNum(QSFA_DIM_NUM_3);
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_0, kvHeadNum);
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_0));
            softmaxMaxShape->SetDim(QSFA_DIM_INDEX_2, g);

            softmaxSumShape->SetDimNum(QSFA_DIM_NUM_3);
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_0, kvHeadNum);
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_1, queryShape->GetDim(QSFA_DIM_INDEX_0));
            softmaxSumShape->SetDim(QSFA_DIM_INDEX_2, g);
        }
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(QSFA_DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(QSFA_DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseFlashAttentionCannV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvQuantSparseFlashAttentionCannV2", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QSFA_QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT); // 2：设置输出数据类型，第三个输出为浮点型
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionCannV2)
    .InferShape(InferShapeKvQuantSparseFlashAttentionCannV2)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttentionCannV2);
} // namespace ops
