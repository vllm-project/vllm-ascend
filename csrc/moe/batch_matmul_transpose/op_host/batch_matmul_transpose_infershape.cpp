/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_transpose_infershape.cpp
 * \brief
 */
#include "tiling_base/error_log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4BatchMatmulTranspose(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4BatchMatmulTranspose");

    const gert::Shape* tensorAShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorAShape);
    const gert::Shape* tensorBShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorBShape);
    gert::Shape* tensorCShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorCShape);

    if (IsUnknownRank(*tensorAShape) || IsUnknownRank(*tensorBShape)) {
        SetUnknownRank(*tensorCShape);
        OP_LOGD(context, "End to do InferShape4BatchMatmulTranspose with unknown rank.");
        return GRAPH_SUCCESS;
    }

    size_t tensorADimNum = tensorAShape->GetDimNum();
    size_t tensorBDimNum = tensorBShape->GetDimNum();
    OP_CHECK_IF(
        tensorADimNum != 3 || (tensorBDimNum != 3 && tensorBDimNum != 4),
        OP_LOGE(context, "tensor_a should be 3d, tensor_b should be 3d in ND or 4d in NZ."),
        return GRAPH_FAILED);

    tensorCShape->SetDimNum(3);
    tensorCShape->SetDim(IDX_0, tensorAShape->GetDim(IDX_0));
    tensorCShape->SetDim(IDX_1, tensorAShape->GetDim(IDX_1));
    tensorCShape->SetDim(IDX_2,
        tensorBDimNum == 4 ? tensorBShape->GetDim(IDX_1) * tensorBShape->GetDim(3) : tensorBShape->GetDim(IDX_2));

    OP_LOGD(context, "End to do InferShape4BatchMatmulTranspose");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4BatchMatmulTranspose(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4BatchMatmulTranspose");
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    OP_LOGD(context, "End to do InferDataType4BatchMatmulTranspose");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchMatmulTranspose).InferShape(InferShape4BatchMatmulTranspose)
    .InferDataType(InferDataType4BatchMatmulTranspose);
} // namespace ops
