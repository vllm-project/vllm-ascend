/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file recurrent_gated_delta_rule_infershape.cpp
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {

const size_t VALUE_INDEX = 2;
const size_t STATE_INDEX = 4;
const size_t VALUE_DIM = 3;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeRecurrentGatedDeltaRule(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("RecurrentGatedDeltaRule", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto shapeValue = context->GetInputShape(VALUE_INDEX);
    auto shapeInitialState = context->GetInputShape(STATE_INDEX);
    auto shapeOut = context->GetOutputShape(DIM_0);
    auto shapeFinalState = context->GetOutputShape(DIM_1);
    if (shapeValue == nullptr || shapeInitialState == nullptr || shapeOut == nullptr || shapeFinalState == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    shapeOut->SetDimNum(VALUE_DIM);
    int64_t outDim0 = shapeValue->GetDim(DIM_0);
    int64_t outDim1 = shapeValue->GetDim(DIM_1);
    int64_t outDim2 = shapeValue->GetDim(DIM_2);
    shapeOut->SetDim(DIM_0, outDim0);
    shapeOut->SetDim(DIM_1, outDim1);
    shapeOut->SetDim(DIM_2, outDim2);

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

static ge::graphStatus InferDataTypeRecurrentGatedDeltaRule(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRule)
    .InferShape(InferShapeRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypeRecurrentGatedDeltaRule);
} // namespace ops
