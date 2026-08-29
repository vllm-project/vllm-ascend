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
 * \file scatter_nd_update_asc_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_VAR = 0;
const int32_t INDEX_OUTPUT_Y = 0;

static ge::graphStatus InferShape4ScatterNdUpdateAsc(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4ScatterNdUpdateAsc.");
    const gert::Shape* varShape = context->GetInputShape(INPUT_IDX_VAR);
    OPS_LOG_E_IF_NULL(context, varShape, return ge::GRAPH_FAILED);
    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    *yShape = *varShape;
    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4ScatterNdUpdateAsc");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4ScatterNdUpdateAsc(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4ScatterNdUpdateAsc enter");
    const auto varDtype = context->GetInputDataType(INPUT_IDX_VAR);
    context->SetOutputDataType(INDEX_OUTPUT_Y, varDtype);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4ScatterNdUpdateAsc end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterNdUpdateAsc)
    .InferShape(InferShape4ScatterNdUpdateAsc)
    .InferDataType(InferDtype4ScatterNdUpdateAsc);
}  // namespace ops