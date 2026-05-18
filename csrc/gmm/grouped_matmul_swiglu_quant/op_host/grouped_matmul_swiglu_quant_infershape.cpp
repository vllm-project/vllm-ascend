/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_proto.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
const int64_t X_INDEX = 0;
const int64_t WEIGHTSCALE_INDEX = 2;
const int64_t M_DIM_INDEX = 0;
const int64_t DIM_LEN = 2;
const int64_t SPLIT_RATIO = 2;
static ge::graphStatus InferShape4GroupedMatmulSwigluQuant(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    const gert::Shape *weightScaleShape = context->GetInputShape(WEIGHTSCALE_INDEX);
    int64_t m = xShape->GetDim(M_DIM_INDEX);
    int64_t N_DIM_INDEX = weightScaleShape->GetDimNum() - 1;
    int64_t n = static_cast<int64_t>(weightScaleShape->GetDim(N_DIM_INDEX) / SPLIT_RATIO);
    auto outShape = context->GetOutputShape(0);
    outShape->SetDimNum(DIM_LEN);
    outShape->SetDim(0, m);
    outShape->SetDim(1, n);
    auto outScaleShape = context->GetOutputShape(1);
    outScaleShape->SetDimNum(1);
    outScaleShape->SetDim(0, m);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmulSwigluQuant(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DataType::DT_INT8);
    context->SetOutputDataType(1, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSwigluQuant)
    .InferShape(InferShape4GroupedMatmulSwigluQuant)
    .InferDataType(InferDataType4GroupedMatmulSwigluQuant);
} // namespace ops
