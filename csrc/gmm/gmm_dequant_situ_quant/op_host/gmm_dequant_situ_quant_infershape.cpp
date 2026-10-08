/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "runtime/infer_shape_context.h"

namespace ops {
ge::graphStatus InferShapeGmmDequantSituQuant(gert::InferShapeContext *context)
{
    const auto *x = context->GetDynamicInputShape(0, 0);
    const auto *weight = context->GetDynamicInputShape(1, 0);
    auto *y = context->GetOutputShape(0);
    auto *scale = context->GetOutputShape(1);
    if (x == nullptr || weight == nullptr || y == nullptr || scale == nullptr ||
        x->GetDimNum() != 2 || weight->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    // Each INT32 carrier packs eight INT4 weights; SiTU halves the GMM width.
    const int64_t packedN = weight->GetDim(1);
    *y = gert::Shape({x->GetDim(0), packedN < 0 ? packedN : packedN * 4});
    *scale = gert::Shape({x->GetDim(0)});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeGmmDequantSituQuant(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GmmDequantSituQuant)
    .InferShape(InferShapeGmmDequantSituQuant)
    .InferDataType(InferDataTypeGmmDequantSituQuant);
} // namespace ops
