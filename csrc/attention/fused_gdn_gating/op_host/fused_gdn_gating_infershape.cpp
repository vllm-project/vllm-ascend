/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating_infershape.cpp
 * \brief Shape and data-type inference for FusedGdnGating.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

using namespace gert;

namespace ops {

namespace {

constexpr size_t INPUT_A_INDEX     = 1;
constexpr size_t OUTPUT_G_INDEX    = 0;
constexpr size_t OUTPUT_BETA_INDEX = 1;
constexpr size_t OUTPUT_DIM_NUM    = 3;
constexpr int64_t OUTPUT_SEQ_LEN   = 1;

} // namespace

static ge::graphStatus InferShapeFusedGdnGating(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto shapeA = context->GetInputShape(INPUT_A_INDEX);
    auto shapeG = context->GetOutputShape(OUTPUT_G_INDEX);
    auto shapeBeta = context->GetOutputShape(OUTPUT_BETA_INDEX);
    if (shapeA == nullptr || shapeG == nullptr || shapeBeta == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (shapeA->GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch    = shapeA->GetDim(0);
    const int64_t numHeads = shapeA->GetDim(1);

    shapeG->SetDimNum(OUTPUT_DIM_NUM);
    shapeG->SetDim(0, OUTPUT_SEQ_LEN);
    shapeG->SetDim(1, batch);
    shapeG->SetDim(2, numHeads);

    shapeBeta->SetDimNum(OUTPUT_DIM_NUM);
    shapeBeta->SetDim(0, OUTPUT_SEQ_LEN);
    shapeBeta->SetDim(1, batch);
    shapeBeta->SetDim(2, numHeads);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedGdnGating(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputADtype = context->GetInputDataType(INPUT_A_INDEX);
    context->SetOutputDataType(OUTPUT_G_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_BETA_INDEX, inputADtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedGdnGating)
    .InferShape(InferShapeFusedGdnGating)
    .InferDataType(InferDataTypeFusedGdnGating);

} // namespace ops
