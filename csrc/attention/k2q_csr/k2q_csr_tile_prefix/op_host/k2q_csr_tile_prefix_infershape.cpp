/**
 * K2qCsrTilePrefix infer shape（scratch inplace）.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeK2qCsrTilePrefix(gert::InferShapeContext *context)
{
    const gert::Shape *scratchInput = context->GetInputShape(0);
    gert::Shape *scratchOut = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scratchInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, scratchOut);
    *scratchOut = *scratchInput;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(K2qCsrTilePrefix).InferShape(InferShapeK2qCsrTilePrefix);
} // namespace ops
