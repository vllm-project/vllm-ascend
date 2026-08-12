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
 * \file runtime_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_
#define OPS_BUILT_IN_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_

#include "context_util.h"
#include "register/op_impl_registry.h"
#include "runtime/continuous_vector.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"

namespace ops {
using QuickVector = gert::Shape;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;

constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kInputIndex2 = 2U;
constexpr size_t kInputIndex3 = 3U;
constexpr size_t kInputIndex4 = 4U;
constexpr size_t kInputIndex5 = 5U;

constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kOutputIndex1 = 1U;
constexpr size_t kOutputIndex2 = 2U;
constexpr size_t kOutputIndex3 = 3U;

constexpr int64_t kRank0 = 0U;
constexpr int64_t kRank1 = 1U;
constexpr int64_t kRank2 = 2U;
constexpr int64_t kRank3 = 3U;

constexpr int64_t kNum0 = 0U;
constexpr int64_t kNum1 = 1U;
constexpr int64_t kNum2 = 2U;

constexpr size_t kAttrIndex0 = 0U;
constexpr size_t kAttrIndex2 = 2U;

constexpr size_t kMaxLoopCount = 20U;
/**
 * @brief interval [start, end) with a step size of `stride`
 */
struct StridedInterval {
    int64_t start;
    int64_t end;
    int64_t stride;
};

struct SubShapePara {
    int64_t start;
    int64_t end;
    int64_t stride;
    SubShapePara(int st, int e, int str) : start(st), end(e), stride(str)
    {
    }
};

// Do infershape for OP which is single-input single-output and in-shape equal out-shape.
ge::graphStatus InferShape4Elewise(gert::InferShapeContext *context);

// Do InferDataType for OP which is single-output and dtype is bool.
ge::graphStatus InferDataType4SingleOutBool(gert::InferDataTypeContext *context);

/*
 * @brief: get output shape
 * @param [in] context: gert::InferShapeContext
 * @param [in] input_idx: constvalue input index
 * @param [in] output_idx: constvalue output index
 * @return vector<int64_t>: success or failed
 */
ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext *context, int64_t input_idx, int64_t output_idx);

/*
 * @brief: get output shape
 * @param [in] context: gert::InferShapeContext
 * @param [in] input_idx: constvalue input index
 * @param [in] output_idxs: constvalue output indexes,vector<int64_t>
 * @return graphStatus: success or failed
 */
ge::graphStatus InferShape4InIdxAndOutVector(gert::InferShapeContext *context, int64_t input_idx,
                                             const std::vector<int64_t> &output_idxs);

std::string ShapeCannotBroadcastMsg(const gert::Shape &shape1, const gert::Shape &shape2);
/*
 * @brief: broadcast new shape to output shape
 * @param [in] shape: const gert::Shape*, new shape to broadcast
 * @param [in/out] shape_output: gert::Shape*, output shape
 * @return succeed or not
 */
bool BroadcastShape(const gert::Shape *in1_shape, const gert::Shape *in2_shape, gert::Shape *out_shape);
bool BroadcastShape(const std::vector<const gert::Shape *> &in_shapes, gert::Shape *out_shape);
bool BroadcastShape(const gert::Shape **in_shapes, size_t size, gert::Shape *out_shape);

/*
 * @brief: set all the output shape to [-1, -1, ....] with input rank
 * @param [in] rank: the output input rank
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape *output_shape)
{
    OP_CHECK_IF(output_shape == nullptr, OP_LOGD("SetAllUnknownDim", "the output_shape is nullptr, return unsuccess"),
                return ge::GRAPH_FAILED);

    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

/*
 * @brief: set output shape to [-2]
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline ge::graphStatus SetUnknownRank(gert::Shape *output_shape)
{
    OP_CHECK_IF(output_shape == nullptr, OP_LOGD("SetUnknownRank", "the output_shape is nullptr, return unsuccess"),
                return ge::GRAPH_FAILED);
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);

    OP_LOGD("SetUnknownRank", "set unknown rank = -2, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

/*
 * @brief: check whether the output shape is unknown rank
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline bool IsUnknownRank(const gert::Shape *check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}


/**
 * Check whether Shape's rank is at least rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
ge::graphStatus WithRankAtLeast(const gert::Shape *tensor, int64_t rank, gert::Shape *out_shape,
                                const std::string opName);

/**
 * Check whether Shape's rank is equal to rank
 * @param tensor Input tensor
 * @param rank expect val of Shape
 * @param out Output Shape
 * @return status whether Shape's condition Satisfied
 */
ge::graphStatus WithRank(const gert::Shape *tensor, int64_t rank, gert::Shape *out_shape, const std::string opName);

/**
 * Add two dims
 * @param dim0 first dim val
 * @param dim1 second dim val
 * @param out sum dim val
 * @return status whether this operation success
 */
ge::graphStatus Add(int64_t dim1, int64_t dim2, int64_t &out);


/**
 * Get SubShape according to start end index and step size stride
 * @param s input Shape
 * @param start sub start index
 * @param end sub end index
 * @param stride sub step size
 * @param out sub shape output
 * @return status whether this operation success
 */
ge::graphStatus SubShape(const gert::Shape *s, SubShapePara &para, gert::Shape *out, const std::string opName);

/*
 * @brief: check whether the output shape is unknown rank shape
 * @param [out] output_shape: the output shape ptr
 * @return bool
 */
inline bool IsUnknownShape(const gert::Shape *check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

/*
 * @brief: Check whether the input is scalar
 * @param context: gert::InferShapeContext
 * @param indices: index of scalar
 * @return ge::graphStatus: whether this operation success
 */
ge::graphStatus CheckInputsIsScalar(const gert::InferShapeContext *context, const std::vector<size_t> &indices);

// do inferDataType for output is same input.
ge::graphStatus InferDtype4SameInput(gert::InferDataTypeContext *context);

/*
 * @brief: Merge two dims of Shape
 * @param dim0 first dim val
 * @param dim1 second dim val
 * @param out merged dim val
 * @return ge::graphStatus: whether this operation success
 */
ge::graphStatus Merge(const int64_t dim1, const int64_t dim2, int64_t &out);

/**
 * @brief: Merge two shapes
 * @param s0 first shape val
 * @param s1 second shape val
 * @param out merged shape val
 * @param opName
 * @return ge::graphStatus: whether this operation success
 */
ge::graphStatus Merge(const gert::Shape *s0, const gert::Shape *s1, gert::Shape *out, const std::string opName);

/**
 * Get SubShape according to start end index and step size stride
 * @param s input Shape
 * @param interval: Struct StridedInterval
 * @param op_name
 * @param out sub shape output
 * @return status whether this operation success
 */
ge::graphStatus SubShape(const gert::Shape *s, StridedInterval &interval, const std::string op_name, gert::Shape *out);

/*
 * @brief: check whether the shape is unknown rank shape or unknown rank
 * @param [in] check_shape: the shape ptr
 * @return bool
 */
bool ShapeFullDefined(const gert::Shape *check_shape);

/*
 * @brief: concat two shape used as the output shape
 * @param [in] s1: the input1 shape ptr
 * @param [in] s2: the input2 shape ptr
 * @return ge::graphStatus: whether this operation success
 */
ge::graphStatus Concatenate(const gert::Shape *s1, const gert::Shape *s2, gert::Shape *out);
} // namespace ops

#endif // OPS_BUILT_IN_OP_PROTO_RUNTIME_RUNTIME_UTIL_H_
