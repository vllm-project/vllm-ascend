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
 * \file fused_quant_scatter_copy_sparse_flash_attention_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "../op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"

namespace ops {
namespace {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t HBM_KV_INDEX = 1;
constexpr uint32_t HBM_KV_OUT_INDEX = 0;
constexpr uint32_t ATTENTION_OUT_INDEX = 1;
}

ge::graphStatus InferShapeFusedQuantScatterCopySparseFlashAttention(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(QUERY_INDEX) == nullptr ||
        context->GetInputShape(HBM_KV_INDEX) == nullptr || context->GetOutputShape(HBM_KV_OUT_INDEX) == nullptr ||
        context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    if (queryShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(HBM_KV_OUT_INDEX) = *context->GetInputShape(HBM_KV_INDEX);
    *context->GetOutputShape(ATTENTION_OUT_INDEX) = *queryShape;
    context->GetOutputShape(ATTENTION_OUT_INDEX)->SetDim(
        2, fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFusedQuantScatterCopySparseFlashAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(HBM_KV_OUT_INDEX, ge::DT_INT8);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, context->GetInputDataType(QUERY_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedQuantScatterCopySparseFlashAttention)
    .InferShape(InferShapeFusedQuantScatterCopySparseFlashAttention)
    .InferDataType(InferDataTypeFusedQuantScatterCopySparseFlashAttention);
} // namespace ops
