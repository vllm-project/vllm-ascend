/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferShape4RmsNormCast(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* y_fp32_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_fp32_shape);
    *y_shape = *x_shape;
    *y_fp32_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RmsNormCast(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormCast)
    .InferShape(InferShape4RmsNormCast)
    .InferDataType(InferDataType4RmsNormCast);
}  // namespace ops
