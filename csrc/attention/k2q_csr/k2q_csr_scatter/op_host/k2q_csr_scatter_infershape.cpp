/**
 * K2qCsrScatter infer shape.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeK2qCsrScatter(gert::InferShapeContext *context)
{
    const gert::Shape *q2kShape = context->GetInputShape(0);
    gert::Shape *qIndShape = context->GetOutputShape(0);
    gert::Shape *slotShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, q2kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qIndShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotShape);

    int64_t H = q2kShape->GetDim(0);
    int64_t T = q2kShape->GetDim(1);
    int64_t topk = q2kShape->GetDim(2);

    qIndShape->SetDimNum(2);
    qIndShape->SetDim(0, H);
    qIndShape->SetDim(1, T * topk);

    slotShape->SetDimNum(2);
    slotShape->SetDim(0, H);
    slotShape->SetDim(1, T * topk);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(K2qCsrScatter).InferShape(InferShapeK2qCsrScatter);
} // namespace ops
