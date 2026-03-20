/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sum_lstm_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;

namespace ops {

static ge::graphStatus InferShape4SumLstm(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* prevCellShape = context->GetInputShape(IDX_2);
    if (prevCellShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outStateShape = context->GetOutputShape(IDX_0);
    gert::Shape* outCellShape = context->GetOutputShape(IDX_1);

    if (outStateShape == nullptr || outCellShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    *outStateShape = *prevCellShape;
    *outCellShape = *prevCellShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SumLstm(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType inputDtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, inputDtype);  // out_state
    context->SetOutputDataType(IDX_1, inputDtype);  // out_cell

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SumLstm)
    .InferShape(InferShape4SumLstm)
    .InferDataType(InferDataType4SumLstm);

}  // namespace ops
