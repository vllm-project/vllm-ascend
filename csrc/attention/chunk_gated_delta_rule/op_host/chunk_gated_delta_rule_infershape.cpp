/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file chunk_gated_delta_rule_proto.cpp
 * \brief
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

using namespace gert;
namespace ops {

// 输入索引（与算子原型定义保持一致）
const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t G_INDEX = 6;

// 输出索引
const size_t OUTPUT_OUT_IDX = 0;
const size_t OUTPUT_FINAL_STATE_IDX = 1;
const size_t VALUE_DIM = 3;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeChunkGatedDeltaRule(InferShapeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("ChunkGatedDeltaRule", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeValue = context->GetInputShape(VALUE_INDEX);
    auto shapeInitialState = context->GetInputShape(STATE_INDEX);
    auto shapeOut = context->GetOutputShape(DIM_0);
    auto shapeFinalState = context->GetOutputShape(DIM_1);
    if (shapeValue == nullptr || shapeInitialState == nullptr || shapeOut == nullptr || shapeFinalState == nullptr) {
        OPS_LOG_E(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    // GetDim 之前先校验 value 和 initialState 的 DimNum 是否符合要求
    OPS_CHECK(shapeValue->GetDimNum() != VALUE_DIM,
                OPS_LOG_E(opName, "value dim num should be %zu, but got %zu", VALUE_DIM, shapeValue->GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_CHECK(
        shapeInitialState->GetDimNum() != STATE_DIM,
        OPS_LOG_E(opName, "initial_state dim num should be %zu, but got %zu", STATE_DIM, shapeInitialState->GetDimNum()),
        return ge::GRAPH_FAILED);

    // out 形状来自 value 的前三维 (T, Nv, Dv)
    shapeOut->SetDimNum(VALUE_DIM);
    int64_t outDim0 = shapeValue->GetDim(DIM_0);
    int64_t outDim1 = shapeValue->GetDim(DIM_1);
    int64_t outDim2 = shapeValue->GetDim(DIM_2);
    shapeOut->SetDim(DIM_0, outDim0);
    shapeOut->SetDim(DIM_1, outDim1);
    shapeOut->SetDim(DIM_2, outDim2);

    // final_state 形状与 initial_state 保持一致
    shapeFinalState->SetDimNum(STATE_DIM);
    int64_t stateDim0 = shapeInitialState->GetDim(DIM_0);
    int64_t stateDim1 = shapeInitialState->GetDim(DIM_1);
    int64_t stateDim2 = shapeInitialState->GetDim(DIM_2);
    int64_t stateDim3 = shapeInitialState->GetDim(DIM_3);
    shapeFinalState->SetDim(DIM_0, stateDim0);
    shapeFinalState->SetDim(DIM_1, stateDim1);
    shapeFinalState->SetDim(DIM_2, stateDim2);
    shapeFinalState->SetDim(DIM_3, stateDim3);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeChunkGatedDeltaRule(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("ChunkGatedDeltaRule", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    // out 跟随 query（恒 bf16）；final_state 跟随 initial_state 的 dtype（恒 float32）
    context->SetOutputDataType(OUTPUT_OUT_IDX, context->GetInputDataType(QUERY_INDEX));
    context->SetOutputDataType(OUTPUT_FINAL_STATE_IDX, context->GetInputDataType(STATE_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRule)
    .InferShape(InferShapeChunkGatedDeltaRule)
    .InferDataType(InferDataTypeChunkGatedDeltaRule);
} // namespace ops
