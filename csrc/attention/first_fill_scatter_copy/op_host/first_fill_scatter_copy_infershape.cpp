#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {

ge::graphStatus InferShapeFirstFillScatterCopy(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FirstFillScatterCopy",
                         "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *hbmRopeShape = context->GetInputShape(0);
    const gert::Shape *hbmKvShape = context->GetInputShape(1);
    gert::Shape *hbmRopeOutShape = context->GetOutputShape(0);
    gert::Shape *hbmKvOutShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, hbmRopeShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, hbmKvShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, hbmRopeOutShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(context, hbmKvOutShape, return ge::GRAPH_FAILED)
    *hbmRopeOutShape = *hbmRopeShape;
    *hbmKvOutShape = *hbmKvShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtypeFirstFillScatterCopy(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FirstFillScatterCopy",
                         "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FirstFillScatterCopy)
    .InferShape(InferShapeFirstFillScatterCopy)
    .InferDataType(InferDtypeFirstFillScatterCopy);

}  // namespace ops
