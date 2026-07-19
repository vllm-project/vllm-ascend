/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope_infershape.cpp */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
using namespace gert;
namespace ops {
namespace {
constexpr size_t IN_Q = 0, IN_K = 1, IN_V = 2;
constexpr size_t OUT_Q = 0, OUT_K = 1, OUT_V = 2;
}
static ge::graphStatus InferShapeDgemmaFusedNormRope(InferShapeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    // outputs mirror q/k/v shapes exactly
    const size_t inIdx[3] = {IN_Q, IN_K, IN_V};
    const size_t outIdx[3] = {OUT_Q, OUT_K, OUT_V};
    for (int i = 0; i < 3; ++i) {
        auto in = context->GetInputShape(inIdx[i]);
        auto out = context->GetOutputShape(outIdx[i]);
        if (in == nullptr || out == nullptr) { return ge::GRAPH_FAILED; }
        out->SetDimNum(in->GetDimNum());
        for (size_t d = 0; d < in->GetDimNum(); ++d) { out->SetDim(d, in->GetDim(d)); }
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeDgemmaFusedNormRope(gert::InferDataTypeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    context->SetOutputDataType(OUT_Q, context->GetInputDataType(IN_Q));
    context->SetOutputDataType(OUT_K, context->GetInputDataType(IN_K));
    context->SetOutputDataType(OUT_V, context->GetInputDataType(IN_V));
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DgemmaFusedNormRope)
    .InferShape(InferShapeDgemmaFusedNormRope)
    .InferDataType(InferDataTypeDgemmaFusedNormRope);
} // namespace ops
