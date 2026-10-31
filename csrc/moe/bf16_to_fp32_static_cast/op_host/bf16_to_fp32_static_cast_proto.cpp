/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "error/ops_error.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int32_t kInputX = 0;
constexpr int32_t kOutputY = 0;
}

static ge::graphStatus InferShape4Bf16ToFp32StaticCast(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kInputX);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(kOutputY);
    OPS_LOG_E_IF_NULL(context, yShape, return ge::GRAPH_FAILED);

    const size_t dimNum = xShape->GetDimNum();
    yShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4Bf16ToFp32StaticCast(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutputY, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Bf16ToFp32StaticCast)
    .InferShape(InferShape4Bf16ToFp32StaticCast)
    .InferDataType(InferDtype4Bf16ToFp32StaticCast);
} // namespace ops
