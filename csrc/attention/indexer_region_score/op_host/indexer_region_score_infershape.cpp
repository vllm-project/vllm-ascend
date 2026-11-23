// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"
namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *q = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    auto *output = context->GetOutputShape(0);
    auto *ids = context->GetOutputShape(1);
    if (!q || !attrs || !output || !ids || q->GetDimNum() != 4) return ge::GRAPH_FAILED;
    const auto *regions = attrs->GetAttrPointer<int64_t>(0);
    const auto *topK = attrs->GetAttrPointer<int64_t>(1);
    const auto *cacheMode = attrs->GetAttrPointer<int64_t>(4);
    const auto *maskMode = attrs->GetAttrPointer<int64_t>(5);
    const auto *sortMode = attrs->GetAttrPointer<int64_t>(6);
    if (!regions || !topK || *regions < 0 || *topK <= 0 || *topK > 512) return ge::GRAPH_FAILED;
    if (!cacheMode || !maskMode || !sortMode || *cacheMode != 1 || *maskMode != 3 || *sortMode != 1)
        return ge::GRAPH_FAILED;
    const auto *returnScore = attrs->GetAttrPointer<bool>(7);
    if (returnScore && !*returnScore) {
        output->SetDimNum(1);
        output->SetDim(0, 0);
    } else {
        const int64_t capacity = output->GetDimNum() == 3 && output->GetDim(0) == q->GetDim(1) &&
            output->GetDim(1) == q->GetDim(0) && output->GetDim(2) >= *regions ? output->GetDim(2) : *regions;
        output->SetDimNum(3);
        output->SetDim(0, q->GetDim(1));
        output->SetDim(1, q->GetDim(0));
        output->SetDim(2, capacity);
    }
    ids->SetDimNum(3);
    ids->SetDim(0, q->GetDim(1));
    ids->SetDim(1, q->GetDim(0));
    ids->SetDim(2, *topK);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_INFERSHAPE(IndexerRegionScore).InferShape(InferShape).InferDataType(InferDtype);
} // namespace ops
