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
 * \file swiglustep_infershape.cpp
 * \brief SwigluStep shape/dtype infer: x[M,2N] -> y[M,N], y dtype = x dtype
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape4Swiglustep(gert::InferShapeContext* context)
{
    // x: [M, 2N] -> y: [M, N] (halve the last dim)
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    int64_t lastDim = xShape->GetDim(xShape->GetDimNum() - 1);
    yShape->SetDim(yShape->GetDimNum() - 1, lastDim / 2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4Swiglustep(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Swiglustep)
    .InferShape(InferShape4Swiglustep)
    .InferDataType(InferDtype4Swiglustep);
}  // namespace ops
