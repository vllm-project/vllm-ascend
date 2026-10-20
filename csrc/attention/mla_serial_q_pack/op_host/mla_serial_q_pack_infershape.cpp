// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext* context)
{
    const auto* x=context->GetInputShape(0);
    const auto* attrs=context->GetAttrs();
    if(x==nullptr || x->GetDimNum()!=2 || attrs==nullptr) return ge::GRAPH_FAILED;
    const auto* heads=attrs->GetInt(0);
    if(heads==nullptr || *heads<=0 || *heads>128 || x->GetDim(1)% *heads || x->GetDim(1)/ *heads<=64)
        return ge::GRAPH_FAILED;
    *context->GetOutputShape(0)=gert::Shape({*heads,x->GetDim(0),x->GetDim(1)/ *heads-64});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0,ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MlaSerialQPack).InferShape(Infer).InferDataType(InferType);
}
