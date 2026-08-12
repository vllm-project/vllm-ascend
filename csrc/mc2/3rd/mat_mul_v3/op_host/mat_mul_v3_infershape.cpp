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
 * \file mat_mul_v3_infer.cpp
 * \brief
 */

#include "common/op_host/matmul_common_infershape.h"
#include "mc2_log_compat.h"

using namespace gert;
namespace {
constexpr size_t MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t MATMUL_MAX_SHAPE_SIZE = 4;
constexpr size_t MATMUL_BIAS_IDX = 2; // input_bias index is different with op_type
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;

void InferComplementedInput(Shape &shape_x1_new, Shape &shape_x2_new, bool &shape_x1_reshape_flag,
                            bool &shape_x2_reshape_flag)
{
    if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > 0) {
        shape_x1_reshape_flag = true;
        int64_t ori_dim = shape_x1_new.GetDim(0);
        shape_x1_new.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape_x1_new.SetDim(0, 1);
        shape_x1_new.SetDim(1, ori_dim);
    }

    if (shape_x2_new.GetDimNum() == 1 && shape_x2_new.GetDim(0) > 0) {
        shape_x2_reshape_flag = true;
        int64_t ori_dim = shape_x2_new.GetDim(0);
        shape_x2_new.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape_x2_new.SetDim(0, ori_dim);
        shape_x2_new.SetDim(1, 1);
    }
}

static void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape &shape_out)
{
    size_t dim_num = shape_out.GetDimNum();
    if (dim_num >= MATMUL_MIN_SHAPE_SIZE) {
        if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
            shape_out.SetDim(dim_num - MATMUL_MIN_SHAPE_SIZE, shape_out.GetDim(dim_num - 1));
            shape_out.SetDimNum(dim_num - 1);
        }
        if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
            shape_out.SetDimNum(dim_num - 1);
        }
    }
}

bool CheckIsUnknownDimNum(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

void UpdateUnknowDimNumToUnkownRank(gert::Shape &shape)
{
    if (CheckIsUnknownDimNum(shape)) {
        shape.SetDimNum(MATMUL_MIN_SHAPE_SIZE);
        shape.SetDim(0, UNKNOWN_DIM);
        shape.SetDim(1, UNKNOWN_DIM);
    }
}

bool UpdateOutputShapeByBias(const std::string &op_name, gert::Shape *shape_out, const gert::Shape *shape_bias)
{
    if (shape_bias != nullptr && shape_bias->GetDimNum() > 0) {
        Shape shape_bias_new(*shape_bias);
        UpdateUnknowDimNumToUnkownRank(shape_bias_new);
        int64_t bias_dim = shape_bias_new.GetDimNum();
        if (shape_bias_new.GetDim(bias_dim - 1) != UNKNOWN_DIM && shape_out->GetDim(1) != UNKNOWN_DIM) {
            OP_CHECK_IF(shape_bias_new.GetDim(bias_dim - 1) != shape_out->GetDim(1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            op_name, "bias", (std::to_string(shape_bias_new.GetDim(bias_dim - 1))).c_str(),
                            "The shape dim of bias must be the same as the n dimension of output."),
                        return false);
        }
        if (shape_bias_new.GetDim(bias_dim - 1) != UNKNOWN_DIM && shape_out->GetDim(1) == UNKNOWN_DIM) {
            shape_out->SetDim(1, shape_bias_new.GetDim(bias_dim - 1));
        }
    }
    return true;
}

static ge::graphStatus InferShapeForMatMulV3(InferShapeContext *context)
{
    auto op_name = context->GetNodeName();
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    auto shape_out = context->GetOutputShape(0);
    auto tensor_x1 = context->GetInputDesc(0);
    auto attrs = context->GetAttrs();
    if (shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || tensor_x1 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(op_name, "x1 or x2 or output");
        return ge::GRAPH_FAILED;
    }
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(op_name, "attrs");
        return ge::GRAPH_FAILED;
    }
    auto shape_bias = context->GetOptionalInputShape(MATMUL_BIAS_IDX);
    if (CheckIsUnknownDimNum(*shape_x1) && CheckIsUnknownDimNum(*shape_x2) &&
        (shape_bias == nullptr || CheckIsUnknownDimNum(*shape_bias))) {
        shape_out->SetDimNum(1);
        shape_out->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    const bool *trans_a = attrs->GetAttrPointer<bool>(0);
    const bool *trans_b = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_IF(trans_a == nullptr || trans_b == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "attrs"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s, transpose_x1: %d, transpose_x2: %d",
            Ops::Base::ToString(*shape_x1).c_str(), Ops::Base::ToString(*shape_x2).c_str(), *trans_a, *trans_b);

    ge::DataType dtype = tensor_x1->GetDataType();
    if (dtype == ge::DT_FLOAT) {
        OP_LOGW(op_name, "%s fp32 op has poor performance!", context->GetNodeName());
    }

    Shape shape_x1_new(*shape_x1);
    Shape shape_x2_new(*shape_x2);
    UpdateUnknowDimNumToUnkownRank(shape_x1_new);
    UpdateUnknowDimNumToUnkownRank(shape_x2_new);
    bool shape_x1_reshape_flag = false;
    bool shape_x2_reshape_flag = false;

    InferComplementedInput(shape_x1_new, shape_x2_new, shape_x1_reshape_flag, shape_x2_reshape_flag);
    if (attrs->GetAttrNum() >= 5) { // 5 attrs: transpose_x1, transpose_x2, offset_x, input_size, hidden_size
        auto input_size = attrs->GetAttrPointer<int64_t>(3);  // 3: input_size
        auto hidden_size = attrs->GetAttrPointer<int64_t>(4); // 4: hidden_size
        if (input_size != nullptr && hidden_size != nullptr && *input_size > 0 && *hidden_size > 0) {
            OP_LOGD(op_name, "get private attr, input_size: %ld, hidden_size: %ld", *input_size, *hidden_size);
            shape_x2_new.SetDim(1, shape_x1_new.GetDim(1));
            int64_t align_dim = (*input_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE +
                                (*hidden_size + BLOCK_SIZE) / BLOCK_SIZE * BLOCK_SIZE;
            shape_x2_new.SetDim(0, align_dim);
        }
    }

    OP_LOGD(op_name, "check the input shape length.");
    if (shape_x1_new.GetDimNum() != MATMUL_MIN_SHAPE_SIZE && shape_x1_new.GetDimNum() != MATMUL_MAX_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "x1", (std::to_string(shape_x1_new.GetDimNum()) + "D").c_str(),
                                     "2D or 4D");
        return ge::GRAPH_FAILED;
    }

    size_t idx_m = 0;
    size_t idx_k_a = 1;
    size_t idx_k_b = 0;
    size_t idx_n = 1;
    if (*trans_a) {
        idx_m = 1;
        idx_k_a = 0;
    }

    if (*trans_b) {
        idx_k_b = 1;
        idx_n = 0;
    }

    if (shape_x1_new.GetDim(idx_k_a) != UNKNOWN_DIM && shape_x2_new.GetDim(idx_k_b) != UNKNOWN_DIM) {
        OP_CHECK_IF(
            shape_x1_new.GetDim(idx_k_a) != shape_x2_new.GetDim(idx_k_b),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                op_name, "x1 and x2",
                (std::to_string(shape_x1_new.GetDim(idx_k_a)) + " and " + std::to_string(shape_x2_new.GetDim(idx_k_b)))
                    .c_str(),
                "The k dimension of x1 must be equal to the k dimension of x2"),
            return ge::GRAPH_FAILED);
    }

    shape_out->SetDimNum(MATMUL_MIN_SHAPE_SIZE);
    shape_out->SetDim(0, shape_x1_new.GetDim(idx_m));
    shape_out->SetDim(1, shape_x2_new.GetDim(idx_n));
    if (!UpdateOutputShapeByBias(op_name, shape_out, shape_bias)) {
        return ge::GRAPH_FAILED;
    }
    InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);
    OP_LOGD(op_name, "end infershape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Mc2MatMulV3).InferShape(InferShapeForMatMulV3);
} // namespace
