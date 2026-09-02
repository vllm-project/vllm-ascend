#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus InferShape4AdnRmsNorm(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4AdnRmsNorm(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdnRmsNorm)
    .InferShape(InferShape4AdnRmsNorm)
    .InferDataType(InferDataType4AdnRmsNorm);

}  // namespace ops
