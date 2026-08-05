/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {

// attn_out has the same shape as query: [batch, num_heads, head_dim]
static ge::graphStatus InferShapeTurboquantPagedAttentionV310(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TurboquantPagedAttentionV310", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    auto qShape = context->GetInputShape(0);
    auto outShape = context->GetOutputShape(0);
    if (qShape == nullptr || outShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }
    *outShape = *qShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTurboquantPagedAttentionV310(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboquantPagedAttentionV310)
    .InferShape(InferShapeTurboquantPagedAttentionV310)
    .InferDataType(InferDataTypeTurboquantPagedAttentionV310);
}  // namespace ops
