/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t PROCESSED_LOGITS_INDEX = 0;
constexpr size_t SAMPLED_TOKEN_IDS_INDEX = 0;
constexpr size_t LSE_INDEX = 1;
}  // namespace

static ge::graphStatus InferShapeCategoricalSample(gert::InferShapeContext* context)
{
    const gert::Shape* logitsShape = context->GetInputShape(PROCESSED_LOGITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, logitsShape);
    if (logitsShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numRows = logitsShape->GetDim(0);
    gert::Shape* sampledTokenIdsShape = context->GetOutputShape(SAMPLED_TOKEN_IDS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampledTokenIdsShape);
    sampledTokenIdsShape->SetDimNum(1);
    sampledTokenIdsShape->SetDim(0, numRows);

    gert::Shape* lseShape = context->GetOutputShape(LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseShape);
    lseShape->SetDimNum(1);
    // ACLNN requires every output to own non-null storage. Keep the custom-op
    // workspace row-shaped even when the public torch wrapper discards LSE.
    lseShape->SetDim(0, numRows);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CategoricalSample).InferShape(InferShapeCategoricalSample);
}  // namespace ops
