/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static graphStatus InferDataType4ScatterNdUpdateV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ScatterNdUpdateV2InferDtype.");
    auto var_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, var_dtype);
    OP_LOGD(context->GetNodeName(), "End to do ScatterNdUpdateV2InferDtype.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterNdUpdateV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ScatterNdUpdateV2InferShape.");
    const gert::Shape* var_in_shape = context->GetInputShape(0);
    gert::Shape* var_out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_in_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_out_shape);
    if (Ops::Base::IsUnknownRank(*var_in_shape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        Ops::Base::SetUnknownRank(*var_out_shape);
        return ge::GRAPH_SUCCESS;
    }
    *var_out_shape = *var_in_shape;
    OP_LOGD(context->GetNodeName(), "End to do ScatterNdUpdateV2InferShape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterNdUpdateV2)
    .InferShape(InferShape4ScatterNdUpdateV2)
    .InferDataType(InferDataType4ScatterNdUpdateV2);
} // namespace ops
