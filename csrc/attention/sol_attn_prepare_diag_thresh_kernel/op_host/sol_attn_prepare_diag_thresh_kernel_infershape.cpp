#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    gert::Shape* thresh_bh_shape = context->GetOutputShape(0);
    thresh_bh_shape->SetDimNum(0);
    thresh_bh_shape->AppendDim(B);
    thresh_bh_shape->AppendDim(H);
    thresh_bh_shape->AppendDim(NT);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnPrepareDiagThreshKernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
