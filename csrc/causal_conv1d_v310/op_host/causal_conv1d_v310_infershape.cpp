/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_v310_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "../tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCausalConv1d");

    const gert::Shape *xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape *yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCausalConv1d");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1dV310).InferShape(InferShapeCausalConv1d);
}  // namespace ops