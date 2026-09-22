// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeFlashAttnC8(gert::InferShapeContext *context)
{
    const auto *query = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    if (!query || !attrs || query->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *query;
    auto *lse = context->GetOutputShape(1);
    if (*attrs->GetAttrPointer<bool>(4)) {
        lse->SetDimNum(2);
        lse->SetDim(0, query->GetDim(1));
        lse->SetDim(1, query->GetDim(0));
    } else {
        lse->SetDimNum(1);
        lse->SetDim(0, 0);
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferTypeFlashAttnC8(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(FlashAttnC8).InferShape(InferShapeFlashAttnC8).InferDataType(InferTypeFlashAttnC8);
}
