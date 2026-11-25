#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

namespace ops {
static ge::graphStatus InferShape4CustomMuls(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    gert::Shape* y = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y);
    *y = *x;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CustomMuls(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    if (dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(customMuls).InferShape(InferShape4CustomMuls).InferDataType(InferDataType4CustomMuls);
} // namespace ops
