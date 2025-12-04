/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ge {
    constexpr uint32_t RESIDUAL_INDEX = 3;
    constexpr uint32_t OUTPUT_Y_INDEX = 0;
    constexpr uint32_t OUTPUT_ADD_OUT_INDEX = 1;
    constexpr int SHAPE_INDEX0 = 0;
    constexpr int SHAPE_INDEX1 = 1;
    constexpr int SHAPE_INDEX2 = 2;
    constexpr int DIM_NUM_2 = 2;
    constexpr int DIM_NUM_3 = 3;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* residualShape = context->GetInputShape(RESIDUAL_INDEX);
    gert::Shape* x1OutShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    gert::Shape* addOutShape = context->GetOutputShape(OUTPUT_ADD_OUT_INDEX);
    int residualDimNum = residualShape->GetDimNum();
    int bs = 1;
    int m = 0;
    int n = 0;

    if (residualDimNum == DIM_NUM_3) {
        bs = residualShape->GetDim(SHAPE_INDEX0);
        m = residualShape->GetDim(SHAPE_INDEX1);
        n = residualShape->GetDim(SHAPE_INDEX2);

        x1OutShape->SetDimNum(residualDimNum);
        x1OutShape->SetDim(SHAPE_INDEX0, bs);
        x1OutShape->SetDim(SHAPE_INDEX1, m);
        x1OutShape->SetDim(SHAPE_INDEX2, n);
        addOutShape->SetDimNum(residualDimNum);
        addOutShape->SetDim(SHAPE_INDEX0, bs);
        addOutShape->SetDim(SHAPE_INDEX1, m);
        addOutShape->SetDim(SHAPE_INDEX2, n);
    } else if (residualDimNum == DIM_NUM_2) {
        m = residualShape->GetDim(SHAPE_INDEX0);
        n = residualShape->GetDim(SHAPE_INDEX1);

        x1OutShape->SetDimNum(residualDimNum);
        x1OutShape->SetDim(SHAPE_INDEX0, m);
        x1OutShape->SetDim(SHAPE_INDEX1, n);
        addOutShape->SetDimNum(residualDimNum);
        addOutShape->SetDim(SHAPE_INDEX0, m);
        addOutShape->SetDim(SHAPE_INDEX1, n);
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto residualDataType = context->GetInputDataType(RESIDUAL_INDEX);
    context->SetOutputDataType(OUTPUT_Y_INDEX, residualDataType);
    context->SetOutputDataType(OUTPUT_ADD_OUT_INDEX, residualDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatmulAllreduceAddRmsnorm)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}