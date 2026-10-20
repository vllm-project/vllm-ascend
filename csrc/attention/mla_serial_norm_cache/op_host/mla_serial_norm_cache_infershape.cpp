// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    if (x == nullptr || x->GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto* blocks = context->GetAttrs()->GetInt(2);
    if (blocks == nullptr || *blocks <= 0) return ge::GRAPH_FAILED;
    *context->GetOutputShape(0) = gert::Shape({*blocks,128,1,512});
    *context->GetOutputShape(1) = gert::Shape({*blocks,128,1,64});
    *context->GetOutputShape(2) = gert::Shape({x->GetDim(0),x->GetDim(1)-576});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0,ge::DT_BF16);
    context->SetOutputDataType(1,ge::DT_BF16);
    context->SetOutputDataType(2,ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MlaSerialNormCache).InferShape(Infer).InferDataType(InferType);
}
