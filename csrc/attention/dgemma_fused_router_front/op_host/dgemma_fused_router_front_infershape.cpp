/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_infershape.cpp */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
using namespace gert;
namespace ops {
namespace {
constexpr size_t IN_X = 0;
constexpr size_t OUT_WEIGHTS = 0;
constexpr size_t OUT_IDS = 1;
}
static ge::graphStatus InferShapeDgemmaFusedRouterFront(InferShapeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto x = context->GetInputShape(IN_X);
    auto attrs = context->GetAttrs();
    if (x == nullptr || attrs == nullptr) { return ge::GRAPH_FAILED; }
    // attrs: 0=epsilon(float), 1=hidden_size(int), 2=num_experts(int), 3=top_k(int)
    const int64_t *topK = attrs->GetAttrPointer<int64_t>(3);
    int64_t m = x->GetDim(0);

    auto weights = context->GetOutputShape(OUT_WEIGHTS);
    auto ids = context->GetOutputShape(OUT_IDS);
    if (weights == nullptr || ids == nullptr || topK == nullptr) { return ge::GRAPH_FAILED; }
    weights->SetDimNum(2); weights->SetDim(0, m); weights->SetDim(1, *topK);
    ids->SetDimNum(2); ids->SetDim(0, m); ids->SetDim(1, *topK);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeDgemmaFusedRouterFront(gert::InferDataTypeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    (void)context->GetInputDataType(IN_X);
    context->SetOutputDataType(OUT_WEIGHTS, ge::DT_FLOAT);
    context->SetOutputDataType(OUT_IDS, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DgemmaFusedRouterFront)
    .InferShape(InferShapeDgemmaFusedRouterFront)
    .InferDataType(InferDataTypeDgemmaFusedRouterFront);
} // namespace ops
