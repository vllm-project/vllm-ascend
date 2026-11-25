/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferShape4QsaUnionInputs(gert::InferShapeContext* context)
{
    const gert::Shape* compact = context->GetInputShape(0);
    const gert::Shape* row_map = context->GetInputShape(2);
    gert::Shape* union_idx = context->GetOutputShape(0);
    gert::Shape* union_count = context->GetOutputShape(1);
    gert::Shape* exact_mask = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, compact);
    OP_CHECK_NULL_WITH_CONTEXT(context, row_map);
    OP_CHECK_NULL_WITH_CONTEXT(context, union_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, union_count);
    OP_CHECK_NULL_WITH_CONTEXT(context, exact_mask);
    OP_CHECK_IF(compact->GetDimNum() != 2 || row_map->GetDimNum() != 2,
                OP_LOGE(context, "QsaUnionInputs expects rank-2 inputs"),
                return ge::GRAPH_FAILED);
    const int64_t input_groups = compact->GetDim(1) - 3;
    const int64_t topk = input_groups + (input_groups < 512 ? 1 : 0);
    const int64_t groups = row_map->GetDim(0);
    const int64_t width = 8 * topk;
    *union_idx = gert::Shape({1, groups, width});
    *union_count = gert::Shape({1, groups});
    *exact_mask = gert::Shape({1, groups, width * 4});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QsaUnionInputs(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QsaUnionInputs)
    .InferShape(InferShape4QsaUnionInputs)
    .InferDataType(InferDataType4QsaUnionInputs);
}  // namespace ops
