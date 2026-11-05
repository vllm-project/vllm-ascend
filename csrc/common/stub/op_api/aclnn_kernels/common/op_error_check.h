/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_ERROR_CHECK_H__
#define OP_ERROR_CHECK_H__

#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"

const int32_t NCHW_N_DIM = 0;
const int32_t NCHW_C_DIM = 1;
const int32_t NHWC_N_DIM = 0;
const int32_t NHWC_C_DIM = 3;

static inline bool IsNullptr(const aclTensor *tensor, const char *name) {
  if (tensor == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a proper Tensor but got null for argument %s.", name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const aclTensorList *tensorList, const char *name) {
  if (tensorList == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a proper TensorList but got null for argument %s.", name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const aclScalar *scalar, const char *name) {
  if (scalar == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a value of type number for argument %s but instead found type null.",
            name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const aclIntArray *intArr, const char *name) {
  if (intArr == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a value of type List[int] for argument %s but instead found type null.",
            name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const aclBoolArray *boolArr, const char *name) {
  if (boolArr == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a value of type List[bool] for argument %s but instead found type null.",
            name);
    return true;
  }
  return false;
}

static inline bool IsNullptr(const aclFloatArray *floatArr, const char *name) {
  if (floatArr == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected a value of type List[float] for argument %s but instead found type \
            null.", name);
    return true;
  }
  return false;
}

static inline bool CheckDims(const aclTensor *tensor) {
  const auto& xShape = tensor->GetViewShape();
  for(size_t i = 0; i < xShape.GetDimNum(); i++) {
    if (xShape.GetDim(i) > INT32_MAX) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The tensor's shape cannot be larger than %d.", INT32_MAX);
      return false;
    }
  }
  return true;
}

static inline bool CheckReduceOutShape(const aclTensor *inferOut, const aclTensor *out)
{
  auto const &xShape = inferOut->GetViewShape();
  auto const &yShape = out->GetViewShape();
  if (xShape != yShape) {
    if (!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The out tensor's shape[%s] is not equal with inferOut shape[%s].",
              op::ToString(out->GetViewShape()).GetString(), op::ToString(inferOut->GetViewShape()).GetString());
      return false;
    }
  }
  return true;
}

static inline bool CheckNCDimValid(const aclTensor *self, const aclTensor *out) {
  auto format = self->GetStorageFormat();
  int64_t selfDimN = 0;
  int64_t selfDimC = 0;
  int64_t outDimN = 0;
  int64_t outDimC = 0;
  if (format == op::Format::FORMAT_NCHW) {
    selfDimN = self->GetViewShape().GetDim(NCHW_N_DIM);
    selfDimC = self->GetViewShape().GetDim(NCHW_C_DIM);
    outDimN = out->GetViewShape().GetDim(NCHW_N_DIM);
    outDimC = out->GetViewShape().GetDim(NCHW_C_DIM);
  } else if (format == op::Format::FORMAT_NHWC) {
    selfDimN = self->GetViewShape().GetDim(NHWC_N_DIM);
    selfDimC = self->GetViewShape().GetDim(NHWC_C_DIM);
    outDimN = out->GetViewShape().GetDim(NHWC_N_DIM);
    outDimC = out->GetViewShape().GetDim(NHWC_C_DIM);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Input and output format only support [NCHW, NHWC] format .");
    return false;
  }
  if ((selfDimN != outDimN) || (selfDimC != outDimC)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The selfDimN[%ld]/outDimN[%ld] or selfDimC[%ld]/outDimC[%ld] not equal .",
            selfDimN, outDimN, selfDimC, outDimC);
    return false;
  }
  return true;
}


#define OP_CHECK_NULL(param, retExpr) \
  if (IsNullptr(param, #param)) { \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SUPPORT(tensor, supportList, retExpr) \
  if (!CheckType(tensor->GetDataType(), supportList)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s not implemented for %s, should be in dtype support list %s.", \
            #tensor, op::ToString(tensor->GetDataType()).GetString(), op::ToString(supportList).GetString()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_MATCH(tensor, expectedDtype, retExpr) \
  if (tensor->GetDataType() != expectedDtype) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s expected dtype is %s but found %s.", \
            #tensor, op::ToString(expectedDtype).GetString(), op::ToString(tensor->GetDataType()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_DTYPE_NOT_SAME(tensor1, tensor2, retExpr) \
  if (tensor1->GetDataType() != tensor2->GetDataType()) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected both tensors to have same dtype, but found %s %s and %s %s.", \
            #tensor1, op::ToString(tensor1->GetDataType()).GetString(), \
            #tensor2, op::ToString(tensor2->GetDataType()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_RESULT_DTYPE_CAST_FAILED(dtype, desiredDtype, retExpr); \
  if (!CanCast(dtype, desiredDtype)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Result type %s can't be cast to the desired output type %s.", \
            op::ToString(dtype).GetString(), op::ToString(desiredDtype).GetString()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST(tensor1, tensor2, retExpr) \
  if (!CheckBroadcastShape(tensor1->GetViewShape(), tensor2->GetViewShape())) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of tensor %s %s must match the size of tensor %s %s.", \
            #tensor1, op::ToString(tensor1->GetViewShape()).GetString(), \
            #tensor2, op::ToString(tensor2->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST_WITH_SHAPE(tensor, shape, retExpr) \
  if (!CheckBroadcastShape(tensor->GetViewShape(), shape)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of tensor %s %s must match the size %s.", \
            #tensor, op::ToString(tensor->GetViewShape()).GetString(), op::ToString(shape).GetString()); \
    retExpr; \
  }

#define OP_CHECK_BROADCAST_AND_INFER_SHAPE(tensor1, tensor2, retShape, retExpr) \
  if (!BroadcastInferShape(tensor1->GetViewShape(), tensor2->GetViewShape(), retShape)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of tensor %s %s must match the size of tensor %s %s.", \
            #tensor1, op::ToString(tensor1->GetViewShape()).GetString(), \
            #tensor2, op::ToString(tensor2->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_SHAPE_NOT_EQUAL(tensor1, tensor2, retExpr) \
  if (tensor1->GetViewShape() != tensor2->GetViewShape()) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for %s to have same size as tensor for %s, but %s does not " \
            "equal %s.", #tensor1, #tensor2, op::ToString(tensor1->GetViewShape()).GetString(), \
            op::ToString(tensor2->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(tensor, shape, retExpr) \
  if (tensor->GetViewShape() != shape) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for %s to have same size as %s, but got %s.", \
            #tensor, op::ToString(shape).GetString(), op::ToString(tensor->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_WRONG_DIMENSION(tensor, expectedDimNum, retExpr) \
  if (tensor->GetViewShape().GetDimNum() != expectedDimNum) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected %zu dimension input, but got %s with sizes %s.", \
            static_cast<size_t>(expectedDimNum), #tensor, op::ToString(tensor->GetViewShape()).GetString()); \
    retExpr; \
  }

#define OP_CHECK_MAX_DIM(tensor, maxDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() > static_cast<size_t>(maxDim)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The %s tensor cannot be larger than %zu dimensions.", \
            #tensor, static_cast<size_t>(maxDim)); \
    retExpr; \
  }

#define OP_CHECK_MIN_DIM(tensor, minDim, retExpr) \
  if (tensor->GetViewShape().GetDimNum() < static_cast<size_t>(minDim)) { \
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The %s tensor must have at least %zu dimensions.", \
            #tensor, static_cast<size_t>(minDim)); \
    retExpr; \
  }

#define OP_CHECK_COMM_INPUT(workspaceSize, executor) \
  if (workspaceSize == nullptr || executor == nullptr) { \
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The workspaceSize or executor is nullptr."); \
    return ACLNN_ERR_PARAM_NULLPTR; \
  }

#define OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_CHECK_INFERSHAPE(cond, retExpr, errMsg, ...) \
  if (cond) { \
    OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, errMsg, ##__VA_ARGS__); \
    retExpr; \
  }

#define OP_CHECK_TENSORLIST_SIZE_EQUAL(tensorlist1, tensorlist2, retExpr)                                         \
    if ((tensorlist1)->Size() != (tensorlist2)->Size()) {                                                         \
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,                                                                          \
                "The %s tensorlist and %s tensorlist must have the same number of tensors, but got %ld and %ld.", \
                #tensorlist1, #tensorlist2, (tensorlist1)->Size(), (tensorlist2)->Size());                        \
        retExpr;                                                                                                  \
    }

#endif