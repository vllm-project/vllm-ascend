/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

static void CloneShape(const gert::Shape* src, gert::Shape* dst)
{
    int ndim = src->GetDimNum();
    dst->SetDimNum(ndim);
    for (int i = 0; i < ndim; ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* residualShape = context->GetInputShape(RESIDUAL_INDEX);
    int residualDimNum = residualShape->GetDimNum();

    if (residualDimNum != DIM_NUM_2 && residualDimNum != DIM_NUM_3) {
        return GRAPH_FAILED;
    }

    gert::Shape* x1OutShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    gert::Shape* addOutShape = context->GetOutputShape(OUTPUT_ADD_OUT_INDEX);
    CloneShape(residualShape, x1OutShape);
    CloneShape(residualShape, addOutShape);

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