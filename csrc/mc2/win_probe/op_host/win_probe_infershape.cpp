/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferShape4WinProbe(gert::InferShapeContext* context)
{
    gert::Shape* addrs_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, addrs_shape);
    // PoC is fixed to 2 ranks.
    addrs_shape->SetDimNum(1);
    addrs_shape->SetDim(0, 2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4WinProbe(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WinProbe)
    .InferShape(InferShape4WinProbe)
    .InferDataType(InferDataType4WinProbe);
}  // namespace ops
