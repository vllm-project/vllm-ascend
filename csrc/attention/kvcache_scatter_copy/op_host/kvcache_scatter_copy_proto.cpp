/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr int32_t INPUT_HBM_K_ROPE = 0;
constexpr int32_t INPUT_HBM_KV_CACHE = 1;

static ge::graphStatus InferShape4KvcacheScatterCopy(gert::InferShapeContext* context)
{
    const gert::Shape* ropeShape = context->GetInputShape(INPUT_HBM_K_ROPE);
    OPS_LOG_E_IF_NULL(context, ropeShape, return ge::GRAPH_FAILED);
    *context->GetOutputShape(0) = *ropeShape;

    const gert::Shape* kvShape = context->GetInputShape(INPUT_HBM_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, kvShape, return ge::GRAPH_FAILED);
    *context->GetOutputShape(1) = *kvShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4KvcacheScatterCopy(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(INPUT_HBM_K_ROPE));
    context->SetOutputDataType(1, context->GetInputDataType(INPUT_HBM_KV_CACHE));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvcacheScatterCopy)
    .InferShape(InferShape4KvcacheScatterCopy)
    .InferDataType(InferDtype4KvcacheScatterCopy);
} // namespace ops
