// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../mla_prolog_v3/op_host/mla_prolog_v3_infershape.h"

namespace ops {
static ge::graphStatus InferDcpC8Shape(gert::InferShapeContext *context)
{
    const auto *x = context->GetRequiredInputShape(0);
    const auto *w = context->GetRequiredInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x);
    OP_CHECK_NULL_WITH_CONTEXT(context, w);
    auto query = *x;
    query.SetDim(query.GetDimNum() - 1, w->GetDim(0));
    query.AppendDim(w->GetDim(2));
    auto rope = query;
    rope.SetDim(rope.GetDimNum() - 1, 64);
    *context->GetOutputShape(0) = query;
    *context->GetOutputShape(1) = rope;
    *context->GetOutputShape(2) = *context->GetRequiredInputShape(9);
    *context->GetOutputShape(3) = *context->GetRequiredInputShape(10);
    for (int i = 4; i <= 6; ++i) {
        context->GetOutputShape(i)->SetDimNum(1);
        context->GetOutputShape(i)->SetDim(0, 0);
    }
    *context->GetOutputShape(7) = query;
    *context->GetOutputShape(8) = rope;
    auto *scale = context->GetOutputShape(9);
    scale->SetDimNum(3);
    scale->SetDim(0, x->GetShapeSize() / x->GetDim(x->GetDimNum() - 1));
    scale->SetDim(1, w->GetDim(0));
    scale->SetDim(2, 1);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDcpC8Type(gert::InferDataTypeContext *context)
{
    const ge::DataType types[] = {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
        ge::DT_FLOAT, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
        ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT};
    for (int i = 0; i < 10; ++i) context->SetOutputDataType(i, types[i]);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MlaPrologDcpC8).InferShape(InferDcpC8Shape).InferDataType(InferDcpC8Type);
}  // namespace ops
