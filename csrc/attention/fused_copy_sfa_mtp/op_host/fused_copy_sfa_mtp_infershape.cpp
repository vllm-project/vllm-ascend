#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {

ge::graphStatus InferShapeFusedCopySfaMtp(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedCopySfaMtp",
                         "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED)
    *outputShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtypeFusedCopySfaMtp(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedCopySfaMtp",
                         "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedCopySfaMtp)
    .InferShape(InferShapeFusedCopySfaMtp)
    .InferDataType(InferDtypeFusedCopySfaMtp);

}  // namespace ops
