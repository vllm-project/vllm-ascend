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
namespace ops {
const int64_t X_INDEX = 0;
const int64_t BIAS_INDEX = 2;
const int64_t M_DIM_INDEX = 0;
const int64_t N_DIM_INDEX = 0;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    const gert::Shape* bias_shape = context->GetInputShape(BIAS_INDEX);
    gert::Shape* outShape = context->GetOutputShape(0);
    int64_t m = xShape->GetDim(M_DIM_INDEX);
    int64_t n = bias_shape->GetDim(N_DIM_INDEX);
    outShape->SetDimNum(2);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatmulGelu)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}

}
