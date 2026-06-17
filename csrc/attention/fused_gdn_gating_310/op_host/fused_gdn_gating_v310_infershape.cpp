/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int INPUT_A = 0;
static constexpr int INPUT_B = 1;
static constexpr int INPUT_A_LOG = 2;
static constexpr int INPUT_DT_BIAS = 3;

static constexpr int OUTPUT_G = 0;
static constexpr int OUTPUT_BETA_OUTPUT = 1;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus CheckLastDimEqualNumHeads(
    gert::InferShapeContext* context,
    const gert::Shape* input_shape,
    int64_t num_heads,
    const char* input_name)
{
    if (input_shape == nullptr) {
        return GRAPH_SUCCESS;
    }

    bool is_unknown_rank = IsUnknownRank(*input_shape);
    if (is_unknown_rank) {
        return GRAPH_SUCCESS;
    }

    size_t dim_num = input_shape->GetDimNum();
    bool is_dim_invalid = dim_num < 1;
    OP_CHECK_IF(is_dim_invalid,
        OP_LOGE(context, "%s must be at least 1-dimensional.", input_name),
        return GRAPH_FAILED);

    int64_t last_dim = input_shape->GetDim(dim_num - 1);
    bool need_check = num_heads > 0 && last_dim > 0;
    if (!need_check) {
        return GRAPH_SUCCESS;
    }

    bool is_mismatch = num_heads != last_dim;
    OP_CHECK_IF(is_mismatch,
        OP_LOGE(context,
            "Mismatch: num_heads of 'a' (%ld) must equal the shape of '%s' (%ld).",
            num_heads,
            input_name,
            last_dim),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

static void SetOutputShapeByInputA(const gert::Shape* a_shape, gert::Shape* output_shape)
{
    size_t a_dim_num = a_shape->GetDimNum();

    output_shape->SetDimNum(a_dim_num + 1);
    output_shape->SetDim(0, 1);

    for (size_t i = 0; i < a_dim_num; ++i) {
        int64_t dim_value = a_shape->GetDim(i);
        output_shape->SetDim(i + 1, dim_value);
    }
}

static ge::graphStatus InferShape4FusedGdnGatingV310(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape for FusedGdnGatingV310");

    const gert::Shape* a_shape = context->GetInputShape(INPUT_A);
    OP_CHECK_NULL_WITH_CONTEXT(context, a_shape);

    gert::Shape* g_shape = context->GetOutputShape(OUTPUT_G);
    OP_CHECK_NULL_WITH_CONTEXT(context, g_shape);

    gert::Shape* beta_output_shape = context->GetOutputShape(OUTPUT_BETA_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, beta_output_shape);

    bool is_a_unknown_rank = IsUnknownRank(*a_shape);
    if (is_a_unknown_rank) {
        SetUnknownRank(*g_shape);
        SetUnknownRank(*beta_output_shape);
        OP_LOGD(context, "End to do InferShape for FusedGdnGatingV310 with unknown rank.");
        return GRAPH_SUCCESS;
    }

    SetOutputShapeByInputA(a_shape, g_shape);
    SetOutputShapeByInputA(a_shape, beta_output_shape);

    size_t a_dim_num = a_shape->GetDimNum();
    if (a_dim_num == 0) {
        OP_LOGD(context, "End to do InferShape for FusedGdnGatingV310");
        return GRAPH_SUCCESS;
    }

    int64_t num_heads = a_shape->GetDim(a_dim_num - 1);

    const gert::Shape* a_log_shape = context->GetInputShape(INPUT_A_LOG);
    ge::graphStatus ret = CheckLastDimEqualNumHeads(
        context,
        a_log_shape,
        num_heads,
        "A_log");
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape* dt_bias_shape = context->GetInputShape(INPUT_DT_BIAS);
    ret = CheckLastDimEqualNumHeads(
        context,
        dt_bias_shape,
        num_heads,
        "dt_bias");
    if (ret != GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGD(context, "End to do InferShape for FusedGdnGatingV310");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FusedGdnGatingV310(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType for FusedGdnGatingV310");

    context->SetOutputDataType(OUTPUT_G, DT_FLOAT);

    ge::DataType beta_output_dtype = context->GetInputDataType(INPUT_B);
    context->SetOutputDataType(OUTPUT_BETA_OUTPUT, beta_output_dtype);

    OP_LOGD(context, "End to do InferDataType for FusedGdnGatingV310");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedGdnGatingV310)
    .InferShape(InferShape4FusedGdnGatingV310)
    .InferDataType(InferDataType4FusedGdnGatingV310);

} // namespace ops