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
 * \file matmul_common_infershape.cpp
 * \brief
 */
#include "matmul_common_infershape.h"
#include "mc2_log_compat.h"
using namespace gert;
namespace {
constexpr size_t BATCH_MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t BATCH_MATMUL_MAX_SHAPE_SIZE = 8;
constexpr size_t BATCH_MATMUL_BIAS_IDX = 2; // input_bias index is different with op_type
constexpr size_t BATCH_MATMUL_FIXPIPE_BIAS_IDX = 3;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
const int64_t B4_NUMS_IN_B32 = 8;

struct InferShapeBatchTensor {
    const Shape input_shape_a;
    const Shape input_shape_b;
    bool input_trans_a;
    bool input_trans_b;
};

class InferShapeBatchMatMul {
public:
    InferShapeBatchMatMul(gert::InferShapeContext *context, const InferShapeBatchTensor &inferShapeTensor,
                          size_t batch_matmul_bias_index = BATCH_MATMUL_BIAS_IDX)
        : op_name(context->GetNodeName()), shape_a(inferShapeTensor.input_shape_a),
          shape_b(inferShapeTensor.input_shape_b), trans_a(inferShapeTensor.input_trans_a),
          trans_b(inferShapeTensor.input_trans_b), shape_out(*(context->GetOutputShape(0))),
          shape_bias(context->GetOptionalInputShape(batch_matmul_bias_index))
    {
        num_dima = shape_a.GetDimNum();
        num_dimb = shape_b.GetDimNum();
        num_dim = std::max(num_dima, num_dimb);
        num_dim_bias = 0;
        if (shape_bias != nullptr) {
            num_dim_bias = shape_bias->GetDimNum();
            num_dim = std::max(num_dim, num_dim_bias);
        }
        shape_out.SetDimNum(num_dim);
    };

    InferShapeBatchMatMul(gert::InferShapeContext *context, const Shape &input_shape_a, const Shape &input_shape_b)
        : op_name(context->GetNodeName()), shape_a(input_shape_a), shape_b(input_shape_b),
          shape_out(*(context->GetOutputShape(0)))
    {
        shape_bias = context->GetOptionalInputShape(BATCH_MATMUL_FIXPIPE_BIAS_IDX);
        num_dima = shape_a.GetDimNum();
        num_dimb = shape_b.GetDimNum();
        num_dim = std::max(num_dima, num_dimb);
        auto attrs = context->GetAttrs();
        trans_a = *(attrs->GetAttrPointer<bool>(0));
        trans_b = *(attrs->GetAttrPointer<bool>(1));
        num_dim_bias = 0;
        if (shape_bias != nullptr) {
            num_dim_bias = shape_bias->GetDimNum();
            num_dim = std::max(num_dim, num_dim_bias);
        }
        shape_out.SetDimNum(num_dim);
    };

    ~InferShapeBatchMatMul() {};
    bool InferShape();

protected:
    bool InferBatch() const;
    bool InferBias();

    size_t num_dim = 0;
    size_t num_dima = 0;
    size_t num_dimb = 0;
    size_t num_dim_bias = 0;

    const char *op_name;
    const Shape &shape_a;
    const Shape &shape_b;
    bool trans_a = false;
    bool trans_b = false;
    Shape &shape_out;
    const Shape *shape_bias;
};

static void CopyOutShapeFromInputShape(const Shape &shape_in, Shape &shape_out, int64_t valid_offset)
{
    for (auto i = 0; i < valid_offset; ++i) {
        shape_out.SetDim(i, shape_in.GetDim(i));
    }
}

bool InferShapeBatchMatMul::InferBatch() const
{
    auto valid_offset = num_dim - std::min(num_dima, num_dimb);
    const Shape &shape_long = num_dima < num_dimb ? shape_b : shape_a;
    const Shape &shape_short = num_dima < num_dimb ? shape_a : shape_b;
    int64_t shape_value_long;
    int64_t shape_value_short;

    CopyOutShapeFromInputShape(shape_long, shape_out, valid_offset);
    // use index - 2 to get index of m
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
        shape_value_short = shape_short.GetDim(i - valid_offset);
        shape_value_long = shape_long.GetDim(i);
        if (shape_value_short > 1 && shape_value_long > 1 && shape_value_short != shape_value_long) {
            return false;
        }
        // 适配一根轴为1，一根轴为-1的情况
        if (shape_value_short == 1) {
            shape_out.SetDim(i, shape_value_long);
        } else if (shape_value_long == 1) {
            shape_out.SetDim(i, shape_value_short);
        } else {
            shape_out.SetDim(i, std::max(shape_value_short, shape_value_long));
        }
    }
    return true;
}

static bool BroadcastBatchDim(const char *op_name, const int64_t dim_a, const int64_t dim_b, int64_t &dim)
{
    if (dim_a > 1 && dim_b > 1) {
        OP_CHECK_IF(dim_a != dim_b,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        op_name, "a and b", (std::to_string(dim_a) + " and " + std::to_string(dim_b)).c_str(),
                        "The shapes of a and b must be the same at the corresponding dimensions"),
                    return false);

        dim = dim_a;
        return true;
    }
    // 适配一根轴为1，一根轴为-1的情况
    if (dim_a == 1) {
        dim = dim_b;
        return true;
    }
    if (dim_b == 1) {
        dim = dim_a;
        return true;
    }
    dim = std::max(dim_a, dim_b);
    return true;
}

static bool InferNWithBias(const char *op_name, const int64_t bias_n, const int64_t out_n, int64_t &n)
{
    if (bias_n == UNKNOWN_DIM || bias_n == UNKNOWN_DIM_NUM) {
        // 当bias中的n为动态shape，不改变最终输出的n
        return true;
    }
    if (bias_n > 0 && out_n > 0) {
        OP_CHECK_IF(bias_n != out_n,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        op_name, "bias_n and out_n", (std::to_string(bias_n) + " and " + std::to_string(out_n)).c_str(),
                        "The shapes of bias_n and out_n must be the same"),
                    return false);
        n = bias_n;
        return true;
    }
    if (out_n == UNKNOWN_DIM && bias_n > 0) {
        n = bias_n;
        return true;
    }

    return false;
}

bool InferShapeBatchMatMul::InferBias()
{
    int64_t shape_value_out = shape_out.GetDim(num_dim - 1);
    // 1) shape_bias = {}
    OP_CHECK_IF(
        num_dim_bias == 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op_name, "bias", "0D", "The shape dim of bias must be at least 1"),
        return true);
    OP_CHECK_IF(shape_bias->GetShapeSize() == 0, OP_LOGI(op_name, "[InferShape] bias shape size is zero"), return true);

    // 2) infer n with bias
    OP_CHECK_IF(
        !InferNWithBias(op_name, shape_bias->GetDim(num_dim_bias - 1), shape_out.GetDim(num_dim - 1), shape_value_out),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "bias and output",
            (std::to_string(shape_bias->GetDim(num_dim_bias - 1)) + " and " +
             std::to_string(shape_out.GetDim(num_dim - 1)))
                .c_str(),
            "The shape of bias and output must be compatible for inferring N dimension"),
        return false);

    shape_out.SetDim(num_dim - 1, shape_value_out);

    // 3) infer batch with bias
    auto valid_offset = num_dim - std::min(num_dim_bias, std::max(num_dima, num_dimb));
    if (num_dim_bias < num_dim) {
        // stop before num_dim - 2 so as to avoid traversing axis m, n
        for (auto i = valid_offset; i < num_dim - 2; ++i) {
            OP_CHECK_IF(
                !BroadcastBatchDim(op_name, shape_bias->GetDim(i - valid_offset), shape_out.GetDim(i), shape_value_out),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    op_name, "bias and output",
                    (std::to_string(shape_bias->GetDim(i - valid_offset)) + " and " +
                     std::to_string(shape_out.GetDim(i)))
                        .c_str(),
                    "The shapes of bias and output must be compatible for batch dimension broadcasting"),
                return false);

            shape_out.SetDim(i, shape_value_out);
        }
        return true;
    }
    CopyOutShapeFromInputShape(*shape_bias, shape_out, valid_offset);
    // stop before num_dim - 2 so as to avoid traversing axis m, n
    for (auto i = valid_offset; i < num_dim - 2; ++i) {
        OP_CHECK_IF(
            !BroadcastBatchDim(op_name, shape_bias->GetDim(i), shape_out.GetDim(i - valid_offset), shape_value_out),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                op_name, "bias and output",
                (std::to_string(shape_bias->GetDim(i)) + " and " + std::to_string(shape_out.GetDim(i - valid_offset)))
                    .c_str(),
                "The shapes of bias and output must be compatible for batch dimension broadcasting"),
            return false);

        shape_out.SetDim(i, shape_value_out);
    }
    return true;
}

bool InferShapeBatchMatMul::InferShape()
{
    if (shape_a.GetDimNum() < BATCH_MATMUL_MIN_SHAPE_SIZE || shape_b.GetDimNum() < BATCH_MATMUL_MIN_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(op_name, "x1 and x2", "less than 2D",
                                                  "The shape dims of x1 and x2 must be at least 2");
        return false;
    }
    for (size_t i = 0; i < shape_a.GetDimNum(); ++i) {
        OP_CHECK_IF(
            shape_a.GetDim(i) < UNKNOWN_DIM_NUM,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "x1",
                (std::to_string(static_cast<int64_t>(shape_a.GetDim(i))) + " at dim " + std::to_string(i)).c_str(),
                "The value of x1 dimension must be at least -2"),
            return false);
    }
    for (size_t i = 0; i < shape_b.GetDimNum(); ++i) {
        OP_CHECK_IF(
            shape_b.GetDim(i) < UNKNOWN_DIM_NUM,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "x2",
                (std::to_string(static_cast<int64_t>(shape_b.GetDim(i))) + " at dim " + std::to_string(i)).c_str(),
                "The value of x2 dimension must be at least -2"),
            return false);
    }
    // using index - 2 to get m_dim
    size_t idx_m = num_dima - 2;
    size_t idx_k_a = num_dima - 1;
    // using index - 2 to get k_dim
    size_t idx_k_b = num_dimb - 2;
    size_t idx_n = num_dimb - 1;
    if (trans_a) {
        idx_m = num_dima - 1;
        // using index - 2 to get k_dim
        idx_k_a = num_dima - 2;
    }
    if (trans_b) {
        idx_k_b = num_dimb - 1;
        // using index - 2 to get n_dim
        idx_n = num_dimb - 2;
    }

    // Check k dim in static shape
    if ((shape_a.GetDim(idx_k_a) != UNKNOWN_DIM && shape_b.GetDim(idx_k_b) != UNKNOWN_DIM) &&
        (shape_a.GetDim(idx_k_a) != shape_b.GetDim(idx_k_b))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            op_name, "x1(k) and x2(k)",
            (std::to_string(shape_a.GetDim(idx_k_a)) + " and " + std::to_string(shape_b.GetDim(idx_k_b))).c_str(),
            "The k dimension of x1 must be equal to the k dimension of x2");
        return false;
    }
    OP_CHECK_IF(
        !InferBatch(),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "x1 and x2",
            (std::to_string(shape_a.GetDimNum()) + "D and " + std::to_string(shape_b.GetDimNum()) + "D").c_str(),
            "The shapes of x1 and x2 must be compatible for inferring batch dimensions"),
        return false);

    // using index - 2 to get m_dim in shape_out
    shape_out.SetDim((num_dim - 2), shape_a.GetDim(idx_m));
    shape_out.SetDim((num_dim - 1), shape_b.GetDim(idx_n));
    if (shape_bias != nullptr) {
        OP_CHECK_IF(!InferBias(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        op_name, "bias", (std::to_string(shape_bias->GetDimNum()) + "D").c_str(),
                        "The shape of bias must be compatible with output for inferring bias"),
                    return false);
    }
    return true;
}

static void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape &shape_out)
{
    size_t dim_num = shape_out.GetDimNum();
    if (dim_num >= BATCH_MATMUL_MIN_SHAPE_SIZE) {
        if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
            shape_out.SetDim(dim_num - BATCH_MATMUL_MIN_SHAPE_SIZE, shape_out.GetDim(dim_num - 1));
            shape_out.SetDimNum(dim_num - 1);
        }

        if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
            shape_out.SetDimNum(dim_num - 1);
        }
    }
}
} // namespace

namespace Ops {
namespace Transformer {
bool CheckIsUnknownDimNum(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

bool CalculateTransX2Float(const gert::InferShapeContext *context, const Shape &shape_x2, bool trans_x1, bool trans_x2)
{
    auto shape_x1 = context->GetInputShape(0);
    auto x1_dim_num = shape_x1->GetDimNum();
    size_t x2_dim_num = shape_x2.GetDimNum();
    size_t k_x1_dim = trans_x1 ? (x1_dim_num - 2UL) : (x1_dim_num - 1UL);
    size_t k_x2_dim = trans_x2 ? (x2_dim_num - 1UL) : (x2_dim_num - 2UL);
    if (!trans_x1 && !trans_x2) {
        return shape_x1->GetDim(k_x1_dim) == shape_x2.GetDim(k_x2_dim) * B4_NUMS_IN_B32;
    }
    return false;
}

ge::graphStatus UpdateX2NewShape(const gert::InferShapeContext *context, Shape &new_shape, bool &reshape_flag,
                                 bool trans_x1, bool trans_x2, const bool is_packed)
{
    auto op_name = context->GetNodeName();
    if (new_shape.GetDimNum() == 1 && new_shape.GetDim(0) > UNKNOWN_DIM_NUM) {
        reshape_flag = true;
        int64_t ori_dim = new_shape.GetDim(0);
        new_shape.SetDimNum(BATCH_MATMUL_MIN_SHAPE_SIZE);
        new_shape.SetDim(0, ori_dim);
        new_shape.SetDim(1, 1);
    }

    if (is_packed) {
        auto *desc = context->GetInputDesc(1);
        OP_CHECK_IF(desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "x2 desc"), return ge::GRAPH_FAILED);
        size_t dim_num = new_shape.GetDimNum();
        size_t x2_dim_num = new_shape.GetDimNum();
        size_t k_x2_dim = trans_x2 ? (x2_dim_num - 1UL) : (x2_dim_num - 2UL);
        size_t n_x2_dim = trans_x2 ? (x2_dim_num - 2UL) : (x2_dim_num - 1UL);
        int64_t k_dim = new_shape.GetDim(k_x2_dim);
        int64_t n_dim = new_shape.GetDim(n_x2_dim);
        if (desc->GetDataType() == ge::DT_FLOAT && k_dim > 0 && n_dim > 0) {
            OP_CHECK_IF(
                dim_num < 1UL,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op_name, "x2", "0D", "The shape dim of x2 must be at least 1"),
                return ge::GRAPH_FAILED);
            bool trans_x2_float = CalculateTransX2Float(context, new_shape, trans_x1, trans_x2);
            if (trans_x2 || trans_x2_float) {
                new_shape.SetDim(k_x2_dim, new_shape.GetDim(k_x2_dim) * B4_NUMS_IN_B32);
            } else {
                new_shape.SetDim(n_x2_dim, new_shape.GetDim(n_x2_dim) * B4_NUMS_IN_B32);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForBatchMatMul(gert::InferShapeContext *context, const int32_t attr_adj_idx,
                                         const size_t input_bias_index, const bool is_x2_packed)
{
    auto shape_x1 = context->GetInputShape(0);
    auto shape_x2 = context->GetInputShape(1);
    auto shape_out = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    auto op_name = context->GetNodeName();
    if (CheckIsUnknownDimNum(*shape_x1) || CheckIsUnknownDimNum(*shape_x2)) {
        shape_out->SetDimNum(1);
        shape_out->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape_x1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "x1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape_x2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "x2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape_out == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "output shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "attrs"), return ge::GRAPH_FAILED);

    const bool *adj_x1 = attrs->GetAttrPointer<bool>(attr_adj_idx);
    const bool *adj_x2 = attrs->GetAttrPointer<bool>(attr_adj_idx + 1);

    OP_CHECK_IF(adj_x1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "adj_x1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(adj_x2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "adj_x2"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "x1_shape: %s, x2_shape: %s, adj_x1: %d, adj_x2: %d",
            Ops::Base::ToString(*shape_x1).c_str(), Ops::Base::ToString(*shape_x2).c_str(), *adj_x1, *adj_x2);

    auto dim_num = std::max(shape_x1->GetDimNum(), shape_x2->GetDimNum());
    if (dim_num < 1 || dim_num > BATCH_MATMUL_MAX_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "x1 and x2", (std::to_string(dim_num) + "D").c_str(),
            "The value of dimension count of x1 and x2 must be in the range of 1 to 8");
        return ge::GRAPH_FAILED;
    }

    // 补充bmmv3单轴逻辑，输入的单轴代表的都是k轴
    Shape shape_x2_new(*shape_x2);
    bool shape_x2_reshape_flag = false;
    if (UpdateX2NewShape(context, shape_x2_new, shape_x2_reshape_flag, *adj_x1, *adj_x2, is_x2_packed) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    Shape shape_x1_new(*shape_x1);
    bool shape_x1_reshape_flag = false;
    if (shape_x1_new.GetDimNum() == 1 && shape_x1_new.GetDim(0) > UNKNOWN_DIM_NUM) {
        shape_x1_reshape_flag = true;
        int64_t ori_dim = shape_x1_new.GetDim(0);
        shape_x1_new.SetDimNum(BATCH_MATMUL_MIN_SHAPE_SIZE);
        shape_x1_new.SetDim(0, 1);
        shape_x1_new.SetDim(1, ori_dim);
    }

    InferShapeBatchTensor InferShapeBatchTensor = {shape_x1_new, shape_x2_new, *adj_x1 && !shape_x1_reshape_flag,
                                                   *adj_x2 && !shape_x2_reshape_flag};
    InferShapeBatchMatMul batchMatMulInfer(context, InferShapeBatchTensor, input_bias_index);
    OP_CHECK_IF(
        !batchMatMulInfer.InferShape(),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "x1 and x2",
            (std::to_string(shape_x1_new.GetDimNum()) + "D and " + std::to_string(shape_x2_new.GetDimNum()) + "D")
                .c_str(),
            "The shapes of x1 and x2 must be compatible for inferring output shape"),
        return ge::GRAPH_FAILED);

    InferComplementedOutput(shape_x1_reshape_flag, shape_x2_reshape_flag, *shape_out);
    OP_LOGD(context->GetNodeName(), "output shape: %s", Ops::Base::ToString(*(context->GetOutputShape(0))).c_str());
    // no need to SetDataType in runtime
    return ge::GRAPH_SUCCESS;
}

// range相关逻辑
constexpr int64_t INFINITE_RANGE = -1;
constexpr int64_t NORMALIZE_INFINITE_RANGE = std::numeric_limits<int64_t>::max();
static const std::pair<int64_t, int64_t> NORMALIZE_FULL_RANGE = {0, NORMALIZE_INFINITE_RANGE};
bool InitializeRange(size_t num, const std::vector<std::pair<int64_t, int64_t>> &range,
                     std::vector<std::pair<int64_t, int64_t>> &new_range)
{
    // 扩充batch轴为1
    for (size_t i = 0; i < num - range.size(); ++i) {
        std::pair<int64_t, int64_t> new_dim = {1, 1};
        new_range.emplace_back(new_dim);
    }
    for (size_t i = 0; i < range.size(); ++i) {
        if (range.at(i).first < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "MatMulCommon", "range",
                (std::to_string(range.at(i).first) + ", " + std::to_string(range.at(i).second)).c_str(),
                "The value of range is incorrect.");
            return false;
        }
        if (range.at(i).second == INFINITE_RANGE) {
            // range {a, -1} -> new_range {a, MAX}
            new_range.emplace_back(std::make_pair(range.at(i).first, NORMALIZE_INFINITE_RANGE));
        } else {
            // range {a, b} -> new_range {a, b}
            new_range.emplace_back(range.at(i));
        }
    }

    return true;
}

bool GetBatchIntersection(const char *op_name, std::pair<int64_t, int64_t> &a, std::pair<int64_t, int64_t> &b,
                          std::pair<int64_t, int64_t> &out)
{
    // 存在1的情况，按范围比较大的情况broadcast
    if (a.first == 1 && b.first == 1) {
        out.first = 1;
        out.second = std::max(a.second, b.second);
        return true;
    }
    if (a.first == 1) {
        out.first = b.first;
        out.second = b.second;
        return true;
    }
    if (b.first == 1) {
        out.first = a.first;
        out.second = a.second;
        return true;
    }
    // 取交集
    out.first = std::max(a.first, b.first);
    out.second = std::min(a.second, b.second);
    // 交集不存在
    if (out.first > out.second || out.first == NORMALIZE_INFINITE_RANGE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "batch range",
                                              (std::to_string(a.first) + ", " + std::to_string(a.second) + " and " +
                                               std::to_string(b.first) + ", " + std::to_string(b.second))
                                                  .c_str(),
                                              "The value of batch range must be within the supported range.");
        return false;
    }
    return true;
}

bool GetKNIntersection(const char *op_name, const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b,
                       std::pair<int64_t, int64_t> &out)
{
    // 取交集
    out.first = std::max(a.first, b.first);
    out.second = std::min(a.second, b.second);
    // 交集不存在
    if (out.first > out.second || out.first == NORMALIZE_INFINITE_RANGE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "KN range",
                                              (std::to_string(a.first) + ", " + std::to_string(a.second) + " and " +
                                               std::to_string(b.first) + ", " + std::to_string(b.second))
                                                  .c_str(),
                                              "The value of KN range must be within the supported range.");
        return false;
    }
    return true;
}

void ExpendOneDimRange(size_t num_dim_x1, size_t num_dim_x2, std::vector<std::pair<int64_t, int64_t>> &shape_range_x1,
                       std::vector<std::pair<int64_t, int64_t>> &shape_range_x2)
{
    // bmmv3支持x1为1维
    if (num_dim_x1 == 1) {
        std::pair<int64_t, int64_t> new_dim = {1, 1};
        shape_range_x1.insert(shape_range_x1.begin(), new_dim);
    }
    // bmmv3支持x2为1维
    if (num_dim_x2 == 1) {
        std::pair<int64_t, int64_t> new_dim = {1, 1};
        shape_range_x2.emplace_back(new_dim);
    }
}

void ReduceOneDimRange(size_t num_dim_x1, size_t num_dim_x2, size_t &num_dim_out,
                       std::vector<std::pair<int64_t, int64_t>> &shape_range_out)
{
    // bmmv3支持x1为1维的还原
    if (num_dim_x1 == 1 && num_dim_x2 > 1) {
        num_dim_out -= 1;
        shape_range_out.erase(shape_range_out.end() - 2); // 去掉倒数第2维增加的m轴
    }
    // bmmv3支持x2为1维的还原
    if (num_dim_x2 == 1 && num_dim_x1 > 1) {
        num_dim_out -= 1;
        shape_range_out.pop_back(); // 去掉倒数第1维增加的n轴
    }
}

bool InferRangeBias(const char *op_name, std::vector<std::pair<int64_t, int64_t>> &new_shape_range_out, size_t idx_n,
                    const gert::Range<gert::Shape> *bias_shape_range,
                    const std::vector<std::pair<int64_t, int64_t>> &new_shape_range_x2)
{
    size_t num_dim_out = new_shape_range_out.size();
    if (bias_shape_range == nullptr) {
        new_shape_range_out[num_dim_out - 1] = new_shape_range_x2[idx_n];
    } else {
        auto bias_min_shape = bias_shape_range->GetMin();
        auto bias_max_shape = bias_shape_range->GetMax();
        OP_CHECK_IF(bias_min_shape == nullptr || bias_max_shape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(op_name, "bias min/max shape"), return false);
        size_t num_dim_bias = bias_min_shape->GetDimNum();
        // 数组长度为0，也是bias不存在，直接返回true不进行校验
        if (num_dim_bias == 0) {
            new_shape_range_out[num_dim_out - 1] = new_shape_range_x2[idx_n];
            return true;
        }
        // 初始化bias，转为vector形式
        std::vector<std::pair<int64_t, int64_t>> tmp_shape_range_bias;
        if (num_dim_out > num_dim_bias) {
            // 把前面差的轴补上
            for (size_t i = 0; i < num_dim_out - num_dim_bias; ++i) {
                tmp_shape_range_bias.emplace_back(std::make_pair(1, 1));
            }
        }
        for (size_t i = 0; i < num_dim_bias; ++i) {
            std::pair<int64_t, int64_t> new_dim = {bias_min_shape->GetDim(i), bias_max_shape->GetDim(i)};
            tmp_shape_range_bias.emplace_back(new_dim);
        }
        std::vector<std::pair<int64_t, int64_t>> new_shape_range_bias;
        OP_CHECK_IF(!InitializeRange(num_dim_out, tmp_shape_range_bias, new_shape_range_bias),
                    OP_LOGE(op_name, "[InferShapeRange] InitializeRange bias failed."), return false);
        // 按bias的batch轴校验并扩充，除去最后的2维，继承原有rt1.0逻辑
        for (size_t i = 0; i < num_dim_out - 2; ++i) {
            OP_CHECK_IF(
                !GetBatchIntersection(op_name, new_shape_range_out[i], new_shape_range_bias[i], new_shape_range_out[i]),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "bias", "range intersection is empty",
                    "The value of batch dimension of bias must be compatible with that of output"),
                return false);
        }
        if (!GetKNIntersection(op_name, new_shape_range_bias[num_dim_out - 1], new_shape_range_x2[idx_n],
                               new_shape_range_out[num_dim_out - 1])) {
            return false;
        }
    }
    return true;
}

class InferShapeRangeBatchMatMul {
public:
    InferShapeRangeBatchMatMul(gert::InferShapeRangeContext *in_context, int32_t in_attr_adj_idx,
                               size_t input_bias_index)
        : context(in_context), op_name(in_context->GetNodeName()), attr_adj_idx(in_attr_adj_idx),
          x1_shape_range(in_context->GetInputShapeRange(0)), x2_shape_range(in_context->GetInputShapeRange(1)),
          bias_shape_range(in_context->GetOptionalInputShapeRange(input_bias_index)),
          out_shape_range(in_context->GetOutputShapeRange(0)) {};
    bool Init();
    bool InferShapeRange();

protected:
    void SetOutput();
    gert::InferShapeRangeContext *context;
    const char *op_name;
    int32_t attr_adj_idx;

    const gert::Range<gert::Shape> *x1_shape_range;
    const gert::Range<gert::Shape> *x2_shape_range;
    const gert::Range<gert::Shape> *bias_shape_range;

    const bool *adj_x1;
    const bool *adj_x2;
    const gert::Shape *x1_min_shape;
    const gert::Shape *x1_max_shape;
    const gert::Shape *x2_min_shape;
    const gert::Shape *x2_max_shape;
    size_t num_dim_x1;
    size_t num_dim_x2;
    size_t num_dim_out;
    std::vector<std::pair<int64_t, int64_t>> src_shape_range_x1;
    std::vector<std::pair<int64_t, int64_t>> src_shape_range_x2;

    gert::Range<gert::Shape> *out_shape_range;
    std::vector<std::pair<int64_t, int64_t>> new_shape_range_out;
};
bool InferShapeRangeBatchMatMul::Init()
{
    OP_CHECK_IF(x1_shape_range == nullptr || x2_shape_range == nullptr || out_shape_range == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(op_name, "x1/x2/out shape range"), return false);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "attrs"), return false);

    adj_x1 = attrs->GetAttrPointer<bool>(attr_adj_idx);
    adj_x2 = attrs->GetAttrPointer<bool>(attr_adj_idx + 1);

    OP_CHECK_IF(adj_x1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "adj_x1"), return false);
    OP_CHECK_IF(adj_x2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(op_name, "adj_x2"), return false);

    x1_min_shape = x1_shape_range->GetMin();
    x1_max_shape = x1_shape_range->GetMax();
    x2_min_shape = x2_shape_range->GetMin();
    x2_max_shape = x2_shape_range->GetMax();
    OP_CHECK_IF(x1_min_shape == nullptr || x1_max_shape == nullptr || x2_min_shape == nullptr ||
                    x2_max_shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(op_name, "x1/x2 min/max shape"), return false);
    num_dim_x1 = x1_min_shape->GetDimNum();
    num_dim_x2 = x2_min_shape->GetDimNum();
    // 初始化x1和x2，转为vector的形式
    for (size_t i = 0; i < num_dim_x1; ++i) {
        std::pair<int64_t, int64_t> new_dim = {x1_min_shape->GetDim(i), x1_max_shape->GetDim(i)};
        src_shape_range_x1.emplace_back(new_dim);
    }
    for (size_t i = 0; i < num_dim_x2; ++i) {
        std::pair<int64_t, int64_t> new_dim = {x2_min_shape->GetDim(i), x2_max_shape->GetDim(i)};
        src_shape_range_x2.emplace_back(new_dim);
    }
    return true;
}

bool InferShapeRangeBatchMatMul::InferShapeRange()
{
    // bmmv3支持x1和x2为1维，输入的都为k轴
    ExpendOneDimRange(num_dim_x1, num_dim_x2, src_shape_range_x1, src_shape_range_x2);
    num_dim_out = std::max(src_shape_range_x1.size(), src_shape_range_x2.size());
    if (bias_shape_range != nullptr) {
        // bias也参与维度扩充
        size_t num_dim_bias = bias_shape_range->GetMin()->GetDimNum();
        num_dim_out = std::max(num_dim_out, num_dim_bias);
    }
    // 扩充x1和x2到一样的维度，用1在前面补充
    std::vector<std::pair<int64_t, int64_t>> new_shape_range_x1;
    OP_CHECK_IF(!InitializeRange(num_dim_out, src_shape_range_x1, new_shape_range_x1),
                OP_LOGE(op_name, "[InferShapeRange] InitializeRange x1 failed."), return false);
    std::vector<std::pair<int64_t, int64_t>> new_shape_range_x2;
    OP_CHECK_IF(!InitializeRange(num_dim_out, src_shape_range_x2, new_shape_range_x2),
                OP_LOGE(op_name, "[InferShapeRange] InitializeRange x2 failed."), return false);
    for (size_t i = 0; i < num_dim_out; ++i) {
        new_shape_range_out.emplace_back(NORMALIZE_FULL_RANGE);
    }
    // using index - 2 to get m_dim
    size_t idx_m = num_dim_out - 2;
    size_t idx_k_x1 = num_dim_out - 1;
    // using index - 2 to get k_dim
    size_t idx_k_x2 = num_dim_out - 2;
    size_t idx_n = num_dim_out - 1;
    // x1单轴时输入都为k轴，所以单轴时m和k的位置不因是否转置改变，x2同理
    if (*adj_x1 && num_dim_x1 != 1) {
        idx_m = num_dim_out - 1;
        // using index - 2 to get k_dim
        idx_k_x1 = num_dim_out - 2;
    }
    if (*adj_x2 && num_dim_x2 != 1) {
        idx_k_x2 = num_dim_out - 1;
        // using index - 2 to get n_dim
        idx_n = num_dim_out - 2;
    }
    // 推理所有维度的range，取交集
    // 推理batch（除去最后m、n这2维）
    for (size_t i = 0; i < num_dim_out - 2; ++i) {
        OP_CHECK_IF(
            !GetBatchIntersection(op_name, new_shape_range_x1[i], new_shape_range_x2[i], new_shape_range_out[i]),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "x1 and x2", "range intersection is empty",
                                                  "The value of batch dimension of x1 and x2 must be compatible"),
            return false);
    }
    // 推理m，输出的倒数第2维度
    new_shape_range_out[num_dim_out - 2] = new_shape_range_x1[idx_m];
    // 推理k，不输出，只校验
    std::pair<int64_t, int64_t> k_range;
    OP_CHECK_IF(!GetKNIntersection(op_name, new_shape_range_x1[idx_k_x1], new_shape_range_x2[idx_k_x2], k_range),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "x1(k) and x2(k)", "ranges do not intersect",
                                                      "The value range of k dimension of x1 and x2 must intersect"),
                return false);
    // 从bias推理N和推理bias的batch
    OP_CHECK_IF(
        !InferRangeBias(op_name, new_shape_range_out, idx_n, bias_shape_range, new_shape_range_x2),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(op_name, "bias and x2",
                                              (std::to_string(bias_shape_range->GetMin()->GetDimNum()) + "D and " +
                                               std::to_string(new_shape_range_x2.size()) + "D")
                                                  .c_str(),
                                              "The shapes of bias and x2 must be compatible for inferring n range"),
        return false);

    // 设置输出range
    SetOutput();
    return true;
}

void InferShapeRangeBatchMatMul::SetOutput()
{
    // bmmv3支持x1和x2为1维，返回前还原该维度
    ReduceOneDimRange(num_dim_x1, num_dim_x2, num_dim_out, new_shape_range_out);
    out_shape_range->GetMin()->SetDimNum(num_dim_out);
    out_shape_range->GetMax()->SetDimNum(num_dim_out);
    for (size_t i = 0; i < num_dim_out; ++i) {
        out_shape_range->GetMin()->SetDim(i, new_shape_range_out[i].first);
        // 无穷大还原回-1
        if (new_shape_range_out[i].second == NORMALIZE_INFINITE_RANGE) {
            new_shape_range_out[i].second = INFINITE_RANGE;
        }
        out_shape_range->GetMax()->SetDim(i, new_shape_range_out[i].second);
    }
}

ge::graphStatus InferShapeRangeForBatchMatMul(gert::InferShapeRangeContext *context, const int32_t attr_adj_idx,
                                              const size_t input_bias_index)
{
    InferShapeRangeBatchMatMul batchMatMulInferRange(context, attr_adj_idx, input_bias_index);
    auto op_name = context->GetNodeName();
    OP_CHECK_IF(!batchMatMulInferRange.Init(), OP_LOGE(op_name, "[InferShapeRange] Failed to init shape range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!batchMatMulInferRange.InferShapeRange(),
                OP_LOGE(op_name, "[InferShapeRange] Failed to infer output shape range"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace Transformer
} // namespace Ops
