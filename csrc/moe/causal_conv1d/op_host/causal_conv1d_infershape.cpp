/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_Y = 0;
// Attrs are positional, in op-def declaration order:
// activationMode(0), padSlotId(1), runMode(2), split_qkv(3).
static constexpr size_t IDX_ATTR_SPLIT_QKV = 3;
static constexpr int64_t IDX_Y_Q = 1;
static constexpr int64_t IDX_Y_K = 2;
static constexpr int64_t IDX_Y_V = 3;

static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCausalConv1d");

    // get input shapes: x is [N, 3*C] (merged qkv)
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // P1: split_qkv attr decides output layout.
    bool splitQkv = false;
    auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *splitPtr = attrs->GetAttrPointer<bool>(IDX_ATTR_SPLIT_QKV);
        if (splitPtr != nullptr) {
            splitQkv = *splitPtr;
        }
    }

    if (splitQkv) {
        // y_q / y_k / y_v keep x's leading dims with the last (merged qkv
        // channel) dim divided by 3, e.g. x [N, 3*C] -> [N, C] each; the
        // caller can then view [N, C] as [1, N, H, D] without a copy. The
        // merged y stays unset in this mode.
        OP_CHECK_IF(xShape->GetDimNum() < 1,
                    OP_LOGE(context->GetNodeName(), "x rank must be >= 1."),
                    return GRAPH_FAILED);
        const int64_t lastDimIdx = static_cast<int64_t>(xShape->GetDimNum()) - 1;
        const int64_t lastDim = xShape->GetDim(lastDimIdx);
        OP_CHECK_IF(lastDim > 0 && lastDim % 3 != 0,
                    OP_LOGE(context->GetNodeName(),
                            "split_qkv requires the last dim of x to be divisible by 3, but got [%ld].",
                            lastDim),
                    return GRAPH_FAILED);
        gert::Shape singleShape = *xShape;
        // Propagate unknown (-1) dims instead of truncating them to 0.
        singleShape.SetDim(lastDimIdx, lastDim > 0 ? lastDim / 3 : lastDim);

        gert::Shape* yQShape = context->GetOutputShape(IDX_Y_Q);
        OP_CHECK_NULL_WITH_CONTEXT(context, yQShape);
        *yQShape = singleShape;
        gert::Shape* yKShape = context->GetOutputShape(IDX_Y_K);
        OP_CHECK_NULL_WITH_CONTEXT(context, yKShape);
        *yKShape = singleShape;
        gert::Shape* yVShape = context->GetOutputShape(IDX_Y_V);
        OP_CHECK_NULL_WITH_CONTEXT(context, yVShape);
        *yVShape = singleShape;
    } else {
        // default: merged y [N, 3*C] == x.
        gert::Shape* yShape = context->GetOutputShape(IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
        *yShape = *xShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCausalConv1d");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1d).InferShape(InferShapeCausalConv1d);
} // namespace ops
