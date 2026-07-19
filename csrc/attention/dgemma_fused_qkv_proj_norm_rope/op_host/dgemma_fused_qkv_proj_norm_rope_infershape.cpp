/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_infershape.cpp */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
using namespace gert;
namespace ops {
namespace {
constexpr size_t IN_HIDDEN = 0;
constexpr size_t IN_QKV_SCRATCH = 6;
constexpr size_t OUT_QKV_SCRATCH = 0, OUT_Q = 1, OUT_K = 2, OUT_V = 3;
}
static ge::graphStatus InferShapeDgemmaFusedQkvProjNormRope(InferShapeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto hidden = context->GetInputShape(IN_HIDDEN);
    auto qkvScratchIn = context->GetInputShape(IN_QKV_SCRATCH);
    auto attrs = context->GetAttrs();
    if (hidden == nullptr || qkvScratchIn == nullptr || attrs == nullptr) { return ge::GRAPH_FAILED; }
    const int64_t *numQ = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *numKv = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *headDim = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *syncBase = attrs->GetAttrPointer<int64_t>(5);
    int64_t m = hidden->GetDim(0);
    bool scratchOnly = syncBase != nullptr && ((static_cast<uint64_t>(*syncBase) & 0x100ULL) != 0ULL);

    auto qkvScratch = context->GetOutputShape(OUT_QKV_SCRATCH);
    auto q = context->GetOutputShape(OUT_Q);
    auto k = context->GetOutputShape(OUT_K);
    auto v = context->GetOutputShape(OUT_V);
    if (qkvScratch == nullptr || q == nullptr || k == nullptr || v == nullptr) { return ge::GRAPH_FAILED; }
    *qkvScratch = *qkvScratchIn;
    if (scratchOnly) {
        q->SetDimNum(3); q->SetDim(0, 1); q->SetDim(1, 1); q->SetDim(2, 1);
        k->SetDimNum(3); k->SetDim(0, 1); k->SetDim(1, 1); k->SetDim(2, 1);
        v->SetDimNum(3); v->SetDim(0, 1); v->SetDim(1, 1); v->SetDim(2, 1);
    } else {
        q->SetDimNum(3); q->SetDim(0, m); q->SetDim(1, *numQ);  q->SetDim(2, *headDim);
        k->SetDimNum(3); k->SetDim(0, m); k->SetDim(1, *numKv); k->SetDim(2, *headDim);
        v->SetDimNum(3); v->SetDim(0, m); v->SetDim(1, *numKv); v->SetDim(2, *headDim);
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeDgemmaFusedQkvProjNormRope(gert::InferDataTypeContext *context)
{
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    auto dt = context->GetInputDataType(IN_HIDDEN);
    context->SetOutputDataType(OUT_QKV_SCRATCH, dt);
    context->SetOutputDataType(OUT_Q, dt);
    context->SetOutputDataType(OUT_K, dt);
    context->SetOutputDataType(OUT_V, dt);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DgemmaFusedQkvProjNormRope)
    .InferShape(InferShapeDgemmaFusedQkvProjNormRope)
    .InferDataType(InferDataTypeDgemmaFusedQkvProjNormRope);
} // namespace ops
