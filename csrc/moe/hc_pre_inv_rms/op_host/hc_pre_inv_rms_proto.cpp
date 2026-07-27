/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hc_pre_inv_rms_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INDEX_OUTPUT_Y = 0;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t BS_INPUT_DIM_NUM = 4;
const static int64_t TND_INPUT_DIM_NUM = 3;

static ge::graphStatus InferShape4HcPreInvRms(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4HcPreInvRms.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
    auto xDimNum = xShape->GetDimNum();

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    // The first one or two dimensions of y match those of x, and the last dimension of y is 1.
    // x: (b, s, hc, d) --> y: (b, s, 1)   or   x: (b * s, hc, d) --> y: (b * s, 1)
    yShape->SetDimNum(xDimNum);
    if (xDimNum == BS_INPUT_DIM_NUM) {
        yShape->SetDim(DIM_0, xShape->GetDim(DIM_0));
        yShape->SetDim(DIM_1, xShape->GetDim(DIM_1));
        yShape->SetDim(DIM_2, 1);
    } else if (xDimNum == TND_INPUT_DIM_NUM) {
        yShape->SetDim(DIM_0, xShape->GetDim(DIM_0));
        yShape->SetDim(DIM_1, 1);
    }

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4HcPreInvRms");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4HcPreInvRms(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcHost enter");
    context->SetOutputDataType(INDEX_OUTPUT_Y, ge::DT_FLOAT);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcPreInvRms end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcPreInvRms)
    .InferShape(InferShape4HcPreInvRms)
    .InferDataType(InferDtype4HcPreInvRms);
}  // namespace ops