/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <register/op_impl_registry.h>

namespace ops {
namespace {
constexpr uint32_t INPUT_INDEX = 0;
constexpr uint32_t OUTPUT_INDEX = 0;
}

static ge::graphStatus InferShapeSfaRemapSparseIndices(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(INPUT_INDEX);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_INDEX);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(inputShape->GetDimNum());
    for (size_t i = 0; i < inputShape->GetDimNum(); ++i) {
        outputShape->SetDim(i, inputShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSfaRemapSparseIndices(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_INDEX, context->GetRequiredInputDataType(INPUT_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SfaRemapSparseIndices)
    .InferShape(InferShapeSfaRemapSparseIndices)
    .InferDataType(InferDataTypeSfaRemapSparseIndices);
}  // namespace ops
