/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_util.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "level0/fill.h"
#include "aclnn/aclnn_base.h"

namespace op {
const aclIntArray* getAllDims(const aclTensor* self, aclOpExecutor* executor) {
  auto input_shape = self->GetViewShape();
  const size_t input_dim_num = input_shape.GetDimNum();
  std::vector<int64_t> dims(input_dim_num);
  for (size_t idx = 0; idx < input_dim_num; idx++) {
    dims[idx] = idx;
  }
  return executor->AllocIntArray(dims.data(), input_dim_num);
}

constexpr size_t MAX_DIM_CNT = 5;
const aclTensor* ResizeFrom1D(const aclTensor* cdim, const aclTensor* input, bool isSupportNcdhw, aclOpExecutor* executor) {
  auto cdimContiguous = l0op::Contiguous(cdim, executor);
  if (cdimContiguous == nullptr) {
    return cdimContiguous;
  }

  auto cdimCast = l0op::Cast(cdimContiguous, DataType::DT_FLOAT, executor);
  if (cdimCast == nullptr) {
    return cdimCast;
  }

  size_t inputDim = input->GetViewShape().GetDimNum();

  const int64_t appendDim[] = {0, 2, 3};
  aclIntArray* newShape = executor->AllocIntArray(appendDim, sizeof(appendDim) / sizeof(int64_t));
  if (inputDim == MAX_DIM_CNT) {
    const int64_t value[] = {0, 2, 3, 4};
    newShape = executor->AllocIntArray(value, sizeof(value) / sizeof(int64_t));
  }
  auto cdimUnsqueeze = l0op::UnsqueezeNd(cdimCast, newShape, executor);
  if (cdimUnsqueeze == nullptr) {
    return cdimUnsqueeze;
  }

  op::Format format = inputDim == MAX_DIM_CNT ? Format::FORMAT_NCDHW : Format::FORMAT_NCHW;
  auto cdimFormat = l0op::ReFormat(cdimUnsqueeze, format);
  if (cdimFormat == nullptr) {
    return cdimFormat;
  }

  if ((inputDim == MAX_DIM_CNT) && !isSupportNcdhw) {
    return l0op::TransDataSpecial(cdimFormat, Format::FORMAT_NDC1HWC0, 0, executor);
  } 
  
  return cdimFormat;
}

const aclTensor* ResizeTo1D(const aclTensor* result, const aclTensor* output, bool isSupportNcdhw, aclOpExecutor* executor) {
  auto resultTransdata = result;
  size_t resultDim = result->GetViewShape().GetDimNum();
  if (resultDim >= MAX_DIM_CNT && !isSupportNcdhw) {
    resultTransdata = l0op::TransDataSpecial(result, Format::FORMAT_NCDHW, 0, executor);
    if (resultTransdata == nullptr) {
      return resultTransdata;
    }
  }

  const int64_t appendDim[] = {0, 2, 3};
  aclIntArray* newShape = executor->AllocIntArray(appendDim, sizeof(appendDim) / sizeof(int64_t));
  if (resultTransdata->GetViewShape().GetDimNum() == MAX_DIM_CNT) {
    const int64_t value[] = {0, 2, 3, 4};
    newShape = executor->AllocIntArray(value, sizeof(value) / sizeof(int64_t));
  }
  auto resultNchw = l0op::SqueezeNd(resultTransdata, newShape, executor);
  if (resultNchw == nullptr) {
    return resultNchw;
  }

  auto resultNd = l0op::ReFormat(resultNchw, Format::FORMAT_ND);
  if (resultNd == nullptr) {
    return resultNd;
  }

  auto resultCast = l0op::Cast(resultNd, output->GetDataType(), executor);
  if (resultCast == nullptr) {
    return resultCast;
  }

  return l0op::ViewCopy(resultCast, output, executor);
}

const aclTensor* ResizeFromND(const aclTensor* input, aclOpExecutor* executor) {
  const int nchw_dims = 4;
  auto inputShape = input->GetViewShape();
  int64_t nchwShape[nchw_dims];
  for (size_t i = 0; i < nchw_dims; i++) {
    nchwShape[i] = i < inputShape.GetDimNum() ? inputShape[i] : 1;
  }
  aclIntArray* nchwArray = executor->AllocIntArray(nchwShape, nchw_dims);

  auto inputReshape = l0op::Reshape(input, nchwArray, executor);
  if (inputReshape == nullptr) {
    return inputReshape;
  }

  return l0op::ReFormat(inputReshape, Format::FORMAT_NCHW);
}

const aclTensor* ResizeToND(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor) {
  auto inputShape = input->GetViewShape();
  size_t dimNum = inputShape.GetDimNum();

  int64_t ndShape[dimNum];
  for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
    ndShape[i] = inputShape[i];
  }
  aclIntArray* ndArray = executor->AllocIntArray(ndShape, dimNum);

  auto outputReshape = l0op::Reshape(output, ndArray, executor);
  if (outputReshape == nullptr) {
    return outputReshape;
  }

  return l0op::ReFormat(outputReshape, input->GetViewFormat());
}

const aclTensor* ResizeFrom5D(const aclTensor* input, aclOpExecutor* executor) {
  auto inputShape = input->GetViewShape();
  // NCDHW -> NDCHW
  const int64_t value[] = {0, 2, 1, 3, 4};
  aclIntArray* ndchwShape = executor->AllocIntArray(value, MAX_DIM_CNT);
  auto inputTranspose = l0op::Transpose(input, ndchwShape, executor);
  if (inputTranspose == nullptr) {
    return inputTranspose;
  }

  // NDCHW -> NCHW
  const int64_t nchwShape[] = {inputShape[0] * inputShape[2], inputShape[1], inputShape[3], inputShape[4]};
  aclIntArray* nchwArray = executor->AllocIntArray(nchwShape, sizeof(nchwShape) / sizeof(int64_t));
  auto inputReshape = l0op::Reshape(inputTranspose, nchwArray, executor);
  if (inputReshape == nullptr) {
    return inputReshape;
  }

  return l0op::ReFormat(inputReshape, Format::FORMAT_NCHW);
}

const aclTensor* ResizeTo5D(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor) {
  auto inputShape = input->GetViewShape();
  // nchw -> ndchw
  const int64_t ndchwShape[] = {inputShape[0], inputShape[2], inputShape[1], inputShape[3], inputShape[4]};
  aclIntArray* ndchwArray = executor->AllocIntArray(ndchwShape, MAX_DIM_CNT);
  auto outputReshape = l0op::Reshape(output, ndchwArray, executor);
  if (outputReshape == nullptr) {
    return outputReshape;
  }

  auto outputFormat = l0op::ReFormat(outputReshape, Format::FORMAT_NCDHW);
  if (outputFormat == nullptr) {
    return outputFormat;
  }
  // ndchw -> ncdhw
  const int64_t ncdhwShape[] = {0, 2, 1, 3, 4};
  aclIntArray* ncdhwArray = executor->AllocIntArray(ncdhwShape, MAX_DIM_CNT);
  return l0op::Transpose(outputFormat, ncdhwArray, executor);
}

aclTensor* FillScalar(int64_t dim, int value, aclOpExecutor* executor) {
  const aclScalar* dimScalar = executor->AllocScalar(dim);
  const aclTensor* dimTensor = executor->ConvertToTensor(dimScalar, op::DataType::DT_INT32);
  aclIntArray* outShape = executor->AllocIntArray(&dim, 1);

  const aclScalar* valueScalar = executor->AllocScalar(value);
  const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, op::DataType::DT_FLOAT);

  auto fillTensor = l0op::Fill(dimTensor, valueTensor, outShape, executor);
  if (fillTensor == nullptr) {
    return nullptr;
  }

  return const_cast<aclTensor*>(fillTensor);
}

aclTensor* FillVector(const op::Shape dstShape, const aclTensor* src, float value, aclOpExecutor* executor) {
  op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(dstShape);
  auto shapes = executor->AllocIntArray(fillDims.data(), src->GetViewShape().GetDimNum());
  const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT32);
  const aclScalar* valueScalar = executor->AllocScalar(value);
  const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, src->GetDataType());
  auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
  if (fillTensor == nullptr) {
    return nullptr;
  }
  fillTensor = l0op::ReFormat(fillTensor, op::Format::FORMAT_ND);
  return const_cast<aclTensor*>(fillTensor);
}

aclnnStatus ProcessEmptyTensorWithValue(aclTensor* src, float initValue, aclOpExecutor* executor) {
  auto srcShape = src->GetViewShape();
  auto dst = FillVector(srcShape, src, initValue, executor);
  auto dstCopyResult = l0op::ViewCopy(dst, src, executor);
  CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

op::DataType CombineCategories(op::DataType higher, op::DataType lower) {
  if (IsFloatingType(higher)) {
    return higher;
  }

  if (IsFloatingType(lower) || higher == op::DataType::DT_BOOL) {
    return op::PromoteType(higher, lower);
  }

  return (higher != op::DataType::DT_UNDEFINED) ? higher : lower;
}
}  // namespace op