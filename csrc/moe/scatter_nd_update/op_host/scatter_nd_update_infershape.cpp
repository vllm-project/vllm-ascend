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
 * \file scatter_nd_update_infershape.cpp
 * \brief ScatterNdUpdate InferShape RT2.0 implementation
 */

#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t SC_IN_VAR_IDX = 0;
static constexpr size_t SC_OUT_VAR_IDX = 0;

graphStatus InferDataType4ScatterNdUpdate(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4ScatterNdUpdate enter");
    auto input_x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_x_dtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4ScatterNdUpdate end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ScatterNdUpdate(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ScatterNdUpdate Infershape.");
    const gert::Shape* var_shape = context->GetInputShape(SC_IN_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, var_shape);

    gert::Shape* output_shape = context->GetOutputShape(SC_OUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    if (Ops::Base::IsUnknownRank(*var_shape)) {
        OP_LOGD(context->GetNodeName(), "input shape is UnknownRank, set output shape to (-2, )");
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    *output_shape = *var_shape;

    OP_LOGD(context->GetNodeName(), "output_shape = %s.", Ops::Base::ToString(*output_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do ScatterNdUpdate Infershape.");

    return ge::GRAPH_SUCCESS;
}

// register infershape for ScatterNdUpdate op
IMPL_OP(ScatterNdUpdate).InferShape(InferShape4ScatterNdUpdate).InferDataType(InferDataType4ScatterNdUpdate);

} // namespace ops
