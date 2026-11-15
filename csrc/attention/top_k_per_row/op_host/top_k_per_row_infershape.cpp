/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"

namespace ops {
constexpr size_t IDX_IN_LOGITS = 0;
constexpr size_t IDX_IN_ROWS = 1;
constexpr size_t IDX_OUT_INDICES = 0;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t ATTR_TOPK = 0;

ge::graphStatus InferShape4TopKPerRow(gert::InferShapeContext* context)
{
    const gert::Shape* logitsShape = context->GetInputShape(IDX_IN_LOGITS);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* topk = attrs->GetInt(ATTR_TOPK);
    OP_CHECK_NULL_WITH_CONTEXT(context, topk);

    const int64_t numRows = (logitsShape->GetDimNum() >= 1) ? logitsShape->GetDim(0) : UNKNOWN_DIM;

    gert::Shape* outShape = context->GetOutputShape(IDX_OUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(2);
    outShape->SetDim(0, numRows);
    outShape->SetDim(1, *topk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4TopKPerRow(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(IDX_OUT_INDICES, ge::DT_INT32);
}

IMPL_OP_INFERSHAPE(TopKPerRow)
    .InferShape(InferShape4TopKPerRow)
    .InferDataType(InferDtype4TopKPerRow);
}  // namespace ops
