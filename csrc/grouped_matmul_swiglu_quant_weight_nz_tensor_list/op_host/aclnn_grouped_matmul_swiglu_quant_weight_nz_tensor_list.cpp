/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include <new>
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t SPLIT = 2;
static constexpr int64_t K_LIMIT = 65536;
static constexpr int64_t N_LIMIT = 4096;
static constexpr int64_t NZ_DIM_3 = 32;
static constexpr int64_t NZ_DIM_2 = 16;
static constexpr int64_t OUTPUT_IDX_0 = 0;
static constexpr int64_t OUTPUT_IDX_1 = 1;
static constexpr size_t X_DIM_LIMIT = 2;
static constexpr size_t WEIGHT_ND_DIM_LIMIT = 2;
static constexpr size_t WEIGHT_NZ_DIM_LIMIT = 4;
static constexpr size_t WEIGHT_SCALE_DIM_LIMIT = 1;
static constexpr size_t TOKEN_SCALE_DIM_LIMIT = 1;
static constexpr size_t GROUP_LIST_DIM_LIMIT = 1;
static constexpr size_t QUANTOUT_DIM_LIMIT = 2;
static constexpr size_t QUANTSCALEOUT_DIM_LIMIT = 1;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
static const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* x, const aclTensorList* weight, const aclTensor* bias, const aclTensor* offset,
                         const aclTensorList* weightScale, const aclTensor* xScale, const aclTensor* groupList,
                         const aclTensor* output, const aclTensor* outputScale, const aclTensor* outputOffset)
{
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(weightScale, return false);
  OP_CHECK_NULL(xScale, return false);
  OP_CHECK_NULL(groupList, return false);
  OP_CHECK_NULL(output, return false);
  OP_CHECK_NULL(outputScale, return false);
  if (bias != nullptr) {
    OP_LOGW("aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario where bias is not 0. "
     "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
  }
  if (offset != nullptr) {
    OP_LOGW("aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario where offset is not 0. "
     "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
  }
  if (outputOffset != nullptr) {
    OP_LOGW("aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario where outputOffset is not 0. "
     "Features and accuracy are not guaranteed if inputting bias with values other than 0s.");
  }
  return true;
}

static bool CheckInputOutDims(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale, 
                              const aclTensor* xScale, const aclTensor* groupList,
                              const aclTensor* output, const aclTensor* outputScale)
{
  OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
  op::Format weightViewFormat = (*weight)[0]->GetViewFormat();
  if (IsPrivateFormat(weightViewFormat)){
    OP_CHECK_WRONG_DIMENSION((*weight)[0], WEIGHT_NZ_DIM_LIMIT, return false);
  } else {
    OP_CHECK_WRONG_DIMENSION((*weight)[0], WEIGHT_ND_DIM_LIMIT, return false);
  }
  OP_CHECK_WRONG_DIMENSION((*weightScale)[0], WEIGHT_SCALE_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(xScale, TOKEN_SCALE_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(output, QUANTOUT_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
  return true;
}

static bool CheckInputOutShape(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale, 
                              const aclTensor* xScale, const aclTensor* groupList,
                              const aclTensor* output, const aclTensor* outputScale)
{
  int64_t m = x->GetViewShape().GetDim(0);
  int64_t k = x->GetViewShape().GetDim(1);
  int64_t n = (*weightScale)[0]->GetViewShape().GetDim(0);
  int64_t e = weight->Size();
  if (n % SPLIT != 0){
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, N is %ld , not an even number.", n);
    return false;
  }
  int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
  // x shape is expected to be [M, K] 
  op::Shape xExpectShape = {m, k};
  // The ND shape of each weight in TensorList is expected to be [K, N] 
  op::Shape weightNDExpectShape = {k, n};
  // The NZ shape of each weight in TensorList is expected to be [N // 32, K // 16, 16, 32] 
  op::Shape weightNZExpectShape = {static_cast<int64_t>(n / NZ_DIM_3), 
                                   static_cast<int64_t>(k / NZ_DIM_2),
                                   NZ_DIM_2, NZ_DIM_3};
  // weightScale shape is expected to be [N] 
  op::Shape weightScaleExpectShape = {n};
  // xScale shape is expected to be [E, N] 
  op::Shape xScaleExpectShape = {m};
  // output shape is expected to be [M, N]
  op::Shape outputExpectShape = {m, nAfterHalve};
  // outputScale shape is expected to be [M]
  op::Shape outputScaleExpectShape = {m};
  for (size_t i = 0; i < weight->Size(); ++i) {
    op::Format weightViewFormat = (*weight)[i]->GetViewFormat();
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
    if (IsPrivateFormat(weightViewFormat)){
      OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*weight)[i], weightNZExpectShape, return false);
    } else {
      OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*weight)[i], weightNDExpectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE((*weightScale)[i], weightScaleExpectShape, return false);
  }
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScale, xScaleExpectShape, return false);

  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputExpectShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputScale, outputScaleExpectShape, return false);
  // The length of groupList should be less than or equal to the number of experts in weight
  int64_t groupListLen = groupList->GetViewShape().GetDim(0);
  if(groupListLen > e) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, Length of 'groupList' out of range (expected to be in range of [1, %ld], but got %ld)",
      e, groupListLen);
    return false;
  }
  if(nAfterHalve > N_LIMIT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario.\
      where N after halve is %ld greater than %ld.",
      nAfterHalve, N_LIMIT);
    return false;
  }
  if(k >= K_LIMIT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
      "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario.\
      The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
      k, K_LIMIT);
    return false;
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale, 
                            const aclTensor* xScale, const aclTensor* groupList,
                            const aclTensor* output, const aclTensor* outputScale)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
  for (size_t i = 0; i < weight->Size(); ++i) {
    OP_CHECK_DTYPE_NOT_SUPPORT((*weight)[i], WEIGHT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT((*weightScale)[i], WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensorList* weight, const aclTensor* output)
{
  bool isNZ = (*weight)[0]->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
  if (!isNZ) {
    // fp16 in fp32 out that is split k template, not precision-advanced now
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario.\
    weight Format expect is FRACTAL_NZ, but got [%s].", op::ToString((*weight)[0]->GetStorageFormat()).GetString());
    return false;
  }
  if (IsPrivateFormat(x->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario.\
    x Format Not support Private Format.");
    return false;
  }
  if (IsPrivateFormat(output->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnGroupedMatmulSwigluQuantWeightNzTensorList, The current version does not support the scenario.\
    output Format Not support Private Format.");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensorList* weight, const aclTensor* bias, const aclTensor* offset,
                               const aclTensorList* weightScale, const aclTensor* xScale, const aclTensor* groupList,
                               const aclTensor* output, const aclTensor* outputScale, const aclTensor* outputOffset) {
  // 1. Check if parameters are null pointers
  CHECK_RET(CheckNotNull(x, weight, bias, offset, weightScale, xScale, 
                         groupList, output, outputScale, outputOffset), ACLNN_ERR_PARAM_NULLPTR);

  // 2. Verify input and output parameter dimensions
  CHECK_RET(CheckInputOutDims(x, weight, weightScale, xScale, 
                              groupList, output, outputScale), ACLNN_ERR_PARAM_INVALID);
  
  // 3. Verify input and output shape parameters
  CHECK_RET(CheckInputOutShape(x, weight, weightScale, xScale, 
                               groupList, output, outputScale), ACLNN_ERR_PARAM_INVALID);

  // 4. Check if the input data types are within the supported data type range
  CHECK_RET(CheckDtypeValid(x, weight, weightScale, xScale, 
                            groupList, output, outputScale), ACLNN_ERR_PARAM_INVALID);

  // 5. Check if data format is supported
  CHECK_RET(CheckFormat(x, weight, output), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzTensorListGetWorkspaceSizeCommon(const aclTensor *x, const aclTensorList *weight,
                                                                       const aclTensor *bias, const aclTensor *offset,
                                                                       const aclTensorList *weightScale, const aclTensor *xScale, 
                                                                       const aclTensor *groupList,  
                                                                       aclTensor *output, aclTensor *outputScale,
                                                                       aclTensor *outputOffset, uint64_t *workspaceSize,
                                                                       aclOpExecutor **executor){
  // Fixed pattern, create OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // Fixed pattern, parameter check
  auto ret = CheckParams(x, weight, bias, offset, weightScale, xScale, 
                         groupList, output, outputScale, outputOffset);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // Empty tensor scenario
  if (output->IsEmpty() || groupList->IsEmpty() || outputScale->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  // Convert to contiguous
  x = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
  for (size_t i = 0; i < weight->Size(); ++i) {
    (*weight)[i]->SetOriginalShape((*weight)[i]->GetViewShape());
  }
  xScale = l0op::Contiguous(xScale, uniqueExecutor.get());
  CHECK_RET(xScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
  groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
  CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // Call L0 operator capability
  auto ret_0 = l0op::GroupedMatmulSwigluQuantWeightNzTensorList(x, weight, weightScale, xScale, groupList, uniqueExecutor.get());
  CHECK_RET(ret_0 != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
  auto out0 = std::get<OUTPUT_IDX_0>(ret_0);
  auto ret_1 = l0op::ViewCopy(out0, output, uniqueExecutor.get());
  CHECK_RET(ret_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto out1 = std::get<OUTPUT_IDX_1>(ret_0);
  auto ret_2 = l0op::ViewCopy(out1, outputScale, uniqueExecutor.get());
  CHECK_RET(ret_2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzTensorListGetWorkspaceSize(const aclTensor *x, const aclTensorList *weight,
                                                                  const aclTensor *bias, const aclTensor *offset,
                                                                  const aclTensorList *weightScale, const aclTensor *xScale, 
                                                                  const aclTensor *groupList,  
                                                                  aclTensor *output, aclTensor *outputScale,
                                                                  aclTensor *outputOffset, uint64_t *workspaceSize,
                                                                  aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNzTensorList,
                 DFX_IN(x, weight, bias, offset, weightScale, xScale, groupList),
                 DFX_OUT(output, outputScale, outputOffset));
  // weight is forcibly bound to StorageFormat and ViewFormat as NZ in this scenario
  CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  for (size_t i = 0; i < weight->Size(); ++i) {
    auto storgeShape = (*weight)[i]->GetStorageShape();
    auto viewShape = (*weight)[i]->GetViewShape();
    aclTensor* weightNZ = const_cast<aclTensor*>((*weight)[i]);
    CHECK_COND((storgeShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT), 
              ACLNN_ERR_PARAM_INVALID,
              "aclnnGroupedMatmulSwigluQuantWeightNZTensorList, The dimnum of storageShape for second input (weight) \
              must be 4. \n But StorageShape got %s , and dimNum is %lu.",
              op::ToString(storgeShape).GetString(), storgeShape.GetDimNum());
    // The StorageFormat of weight is unconditionally regarded as NZ
    weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    if (viewShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT){
      // If the viewShape of weight is 4-dimensional, it is regarded as NZ
      weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
    } else if (viewShape.GetDimNum() == WEIGHT_ND_DIM_LIMIT){
      // If the viewShape of weight is 2-dimensional, it is regarded as ND
      weightNZ->SetViewFormat(op::Format::FORMAT_ND);
    }
  }
  // Call the common interface
  return aclnnGroupedMatmulSwigluQuantWeightNzTensorListGetWorkspaceSizeCommon(x, weight, bias, offset, weightScale, xScale, groupList, 
    output, outputScale, outputOffset, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzTensorList(void *workspace, 
                                          uint64_t workspaceSize, 
                                          aclOpExecutor *executor,
                                          aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNzTensorList);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
             "This is an error in GroupedMatmulSwigluQuantWeightNzTensorList launch aicore");
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
