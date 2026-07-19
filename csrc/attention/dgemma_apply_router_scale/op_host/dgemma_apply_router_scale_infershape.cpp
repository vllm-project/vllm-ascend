/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
using namespace gert;
namespace ops {
namespace { constexpr size_t IN_WEIGHTS = 0, OUT = 0; }
static ge::graphStatus InferShapeDgemmaApplyRouterScale(InferShapeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto in = context->GetInputShape(IN_WEIGHTS);
    auto out = context->GetOutputShape(OUT);
    if (in == nullptr || out == nullptr) { return ge::GRAPH_FAILED; }
    out->SetDimNum(in->GetDimNum());
    for (size_t d = 0; d < in->GetDimNum(); ++d) { out->SetDim(d, in->GetDim(d)); }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeDgemmaApplyRouterScale(gert::InferDataTypeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    context->SetOutputDataType(OUT, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DgemmaApplyRouterScale)
    .InferShape(InferShapeDgemmaApplyRouterScale)
    .InferDataType(InferDataTypeDgemmaApplyRouterScale);
} // namespace ops
