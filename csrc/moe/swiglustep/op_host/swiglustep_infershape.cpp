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
 * \brief SwigluStep shape/dtype infer: y shape = gate shape, y dtype = gate dtype
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape4Swiglustep(gert::InferShapeContext* context)
{
    const gert::Shape* gateShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *gateShape;
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
