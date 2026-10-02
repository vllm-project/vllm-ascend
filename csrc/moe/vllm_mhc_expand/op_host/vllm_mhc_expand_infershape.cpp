// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    auto* y = context->GetOutputShape(0);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    OP_CHECK_NULL_WITH_CONTEXT(context, y);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* mult = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, mult);
    if (x->GetDimNum() != 2 || *mult <= 0) {
        return ge::GRAPH_FAILED;
    }
    y->SetDimNum(3);
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, *mult);
    y->SetDim(2, x->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(VllmMhcExpand).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
