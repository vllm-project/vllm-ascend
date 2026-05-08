/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

using namespace ge;

namespace ops {

static graphStatus InferShapeForGumbelSample(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // sampled.shape = [num_tokens]，取 logits（输入 0）的 dim0
    const gert::Shape* logitsShape = context->GetInputShape(0);
    if (logitsShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (logitsShape->GetDimNum() < 2) {
        return GRAPH_FAILED;
    }

    gert::Shape* sampledShape = context->GetOutputShape(0);
    if (sampledShape == nullptr) {
        return GRAPH_FAILED;
    }
    sampledShape->SetDimNum(1);
    sampledShape->SetDim(0, logitsShape->GetDim(0));   // sampled.shape = [num_tokens]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForGumbelSample(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, DT_INT64);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GumbelSample)
    .InferShape(InferShapeForGumbelSample)
    .InferDataType(InferDataTypeForGumbelSample);

}  // namespace ops
