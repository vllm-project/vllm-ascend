#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t T = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    const int32_t T_LSE = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    gert::Shape* out_shape = context->GetOutputShape(0);
    out_shape->SetDimNum(0);
    out_shape->AppendDim(B);
    out_shape->AppendDim(T);
    out_shape->AppendDim(H);
    out_shape->AppendDim(128);
    gert::Shape* lse_shape = context->GetOutputShape(1);
    lse_shape->SetDimNum(0);
    lse_shape->AppendDim(B);
    lse_shape->AppendDim(H);
    lse_shape->AppendDim(T_LSE);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnShortFwdM128Kernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
