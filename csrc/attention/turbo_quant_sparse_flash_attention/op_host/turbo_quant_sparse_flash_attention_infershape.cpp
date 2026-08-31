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
 * \file turbo_quant_sparse_flash_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

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
constexpr uint32_t OUTPUT_INDEX_1 = 1;
constexpr uint32_t OUTPUT_INDEX_2 = 2;

ge::graphStatus InferShapeTurboQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "InferShapeContext invalid"),
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
    const int64_t *ropeHeadDimPtr = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeHeadDimPtr);
    const int64_t ropeHeadDim = *ropeHeadDimPtr;
    const bool *lse_flag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    bool return_softmax_lse = (lse_flag != nullptr) ? *lse_flag : false;

    // 图推导须与 host tiling 保持同一边界，否则非法配置会在推导阶段成功、
    // 得到错误的输出 shape，直到运行期才失败。
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND",
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "layout_query invalid, only TND is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputLayoutKvPtrStr != "PA_BSND",
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "layout_kv invalid, only PA_BSND is supported"),
        return ge::GRAPH_FAILED);

    // query 为 TND 的 3 维、KV 为 PA_BSND 的 4 维；下面的 dim 访问与除法均依赖于此。
    OP_CHECK_IF(
        queryShape->GetDimNum() != DIM_NUM_3,
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "query shape dim num invalid, TND requires 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        keyShape->GetDimNum() != DIM_NUM_4,
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "key shape dim num invalid, PA_BSND requires 4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeHeadDim < 0 || ropeHeadDim >= queryShape->GetDim(DIM_INDEX_2),
                OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "rope_head_dim invalid"),
                return ge::GRAPH_FAILED);

    *attentionOutShape = *queryShape;
    attentionOutShape->SetDimNum(DIM_NUM_3);
    attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
    attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
    if (queryShape->GetDim(DIM_INDEX_2) != -1) {
        attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2) - ropeHeadDim); // 2:dim2
    }

    if (return_softmax_lse) {
        // KV 为 PA_BSND，N2 位于 dim2；query 已校验为 3 维，故不再有 4 维 LSE 分支。
        int64_t kvHeadDim = keyShape->GetDim(DIM_INDEX_2);
        OP_CHECK_IF(
            kvHeadDim <= 0,
            OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "kv head num should be greater than 0"),
            return ge::GRAPH_FAILED);
        softmaxMaxShape->SetDimNum(DIM_NUM_3);
        softmaxMaxShape->SetDim(DIM_INDEX_0, kvHeadDim);
        softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
        softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / kvHeadDim);
        softmaxSumShape->SetDimNum(DIM_NUM_3);
        softmaxSumShape->SetDim(DIM_INDEX_0, kvHeadDim);
        softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
        softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / kvHeadDim);
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeTurboQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "InferShapeContext invalid"),
                return ge::GRAPH_FAILED);
    // 算子仅支持 BFLOAT16；此处若照搬 query 的 dtype，非法 dtype 会先得到一份看似
    // 合法的图推导结果，直到 tiling 阶段才失败。
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    OP_CHECK_IF(inputDataType != ge::DT_BF16,
                OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention",
                                            "query dtype invalid, only BFLOAT16 is supported"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantSparseFlashAttention)
    .InferShape(InferShapeTurboQuantSparseFlashAttention)
    .InferDataType(InferDataTypeTurboQuantSparseFlashAttention);
} // namespace ops
