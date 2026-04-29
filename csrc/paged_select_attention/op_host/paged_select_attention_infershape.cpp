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
 * \file paged_select_attention_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShapePagedSelectAttention(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_LOG_E("PagedSelectAttention", "InferShape context is null"),
        return ge::GRAPH_FAILED);

    auto queryShape = context->GetInputShape(0);
    auto attentionOutShape = context->GetOutputShape(0);
    auto softmaxLseShape = context->GetOutputShape(1);
    OPS_ERR_IF(queryShape == nullptr || attentionOutShape == nullptr || softmaxLseShape == nullptr,
        OPS_LOG_E("PagedSelectAttention", "InferShape tensor shape is null"),
        return ge::GRAPH_FAILED);

    *attentionOutShape = *queryShape;
    softmaxLseShape->SetDimNum(1);
    softmaxLseShape->SetDim(0, 0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePagedSelectAttention(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PagedSelectAttention)
    .InferShape(InferShapePagedSelectAttention)
    .InferDataType(InferDataTypePagedSelectAttention);
} // namespace ops
