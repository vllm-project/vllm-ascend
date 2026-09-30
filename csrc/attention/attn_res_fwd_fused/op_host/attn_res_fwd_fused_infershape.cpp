// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *prefix = context->GetInputShape(0);
    if (prefix == nullptr) return ge::GRAPH_FAILED;
    *context->GetOutputShape(0) = *prefix;
    *context->GetOutputShape(1) = *prefix;
    *context->GetOutputShape(2) = *prefix;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    context->SetOutputDataType(2, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AttnResFwdFused).InferShape(InferShape).InferDataType(InferDataType);
}
