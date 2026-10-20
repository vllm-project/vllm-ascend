// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    if (x == nullptr || x->GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto* heads = context->GetAttrs()->GetInt(0);
    if (heads == nullptr || *heads <= 0) return ge::GRAPH_FAILED;
    if (x->GetDim(1) % *heads || x->GetDim(1) / *heads <= 64) return ge::GRAPH_FAILED;
    *context->GetOutputShape(0) = gert::Shape({x->GetDim(0),*heads,x->GetDim(1)/ *heads-64});
    *context->GetOutputShape(1) = gert::Shape({x->GetDim(0),*heads,64});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0,ge::DT_BF16);
    context->SetOutputDataType(1,ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MlaSerialQPrepare).InferShape(Infer).InferDataType(InferType);
}
