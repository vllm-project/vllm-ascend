#include <register/op_impl_registry.h>
#include "runtime/infer_shape_context.h"

namespace ops {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x = context->GetInputShape(0);
    auto* y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr || x->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(GroupedMatmulSituQuantMlp).InferShape(InferShape).InferDataType(InferDtype);

} // namespace ops
