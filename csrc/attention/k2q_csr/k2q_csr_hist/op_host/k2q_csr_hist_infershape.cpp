/**
 * K2qCsrHist infer shape（scratch inplace）.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeK2qCsrHist(gert::InferShapeContext *context)
{
    const gert::Shape *scratchInput = context->GetInputShape(1);
    gert::Shape *scratchOut = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scratchInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, scratchOut);
    *scratchOut = *scratchInput;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(K2qCsrHist).InferShape(InferShapeK2qCsrHist);
} // namespace ops
