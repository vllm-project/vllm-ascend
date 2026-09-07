/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vector_paged_attention_infershape.cpp
 * \brief InferShape implementation for VectorPagedAttention
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {
namespace {
constexpr size_t QUERY_INDEX = 0;
constexpr size_t ATTN_OUT_INDEX = 0;
constexpr size_t QUERY_DIM_NUM = 3;
}  // namespace

// The output carries the query's [batch, numHeads, headDim] shape: one row of
// attention output per query row.
static ge::graphStatus InferShape4VectorPagedAttention(gert::InferShapeContext* context)
{
    const gert::Shape* query = context->GetInputShape(QUERY_INDEX);
    gert::Shape* attnOut = context->GetOutputShape(ATTN_OUT_INDEX);
    if (query == nullptr || attnOut == nullptr || query->GetDimNum() != QUERY_DIM_NUM) {
        return GRAPH_FAILED;
    }
    *attnOut = *query;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4VectorPagedAttention(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(ATTN_OUT_INDEX, context->GetInputDataType(QUERY_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VectorPagedAttention)
    .InferShape(InferShape4VectorPagedAttention)
    .InferDataType(InferDataType4VectorPagedAttention);
}  // namespace ops
