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
 * \file aclnn_apply_top_k_top_p_custom.cpp
 * \brief
 */
#include "aclnn_apply_top_k_top_p_custom.h"
#include "apply_top_k_top_p_custom.h"
#include "sort.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const int64_t EXPECTED_DIM_ONE = 1;
static const int64_t EXPECTED_DIM_TWO = 2;
static constexpr size_t DIM_ONE = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* logits, const aclTensor* p, const aclTensor *k, const aclTensor* out)
{
  OP_CHECK_NULL(logits, return false);
  if (p == nullptr && k == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The inputs, p and k, should not be nullptr at the same time.");
  }
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* logits, const aclTensor* p, const aclTensor *k, const aclTensor* out)
{
  // 检查数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(logits, DTYPE_SUPPORT_LIST, return false);
  if (p != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(p, DTYPE_SUPPORT_LIST, return false);
  }
  if (k != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(k, INT_DTYPE_SUPPORT_LIST, return false);
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

  // 检查数据类型是否相同
  if (p != nullptr) {
    OP_CHECK_DTYPE_NOT_MATCH(p, logits->GetDataType(), return false);
  }
  OP_CHECK_DTYPE_NOT_MATCH(out, logits->GetDataType(), return false);
  return true;
}

static bool CheckShapeValid(const aclTensor* logits, const aclTensor* p, const aclTensor *k, const aclTensor* out)
{
  OP_CHECK_WRONG_DIMENSION(logits, EXPECTED_DIM_TWO, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(out, logits, return false);
  if (p != nullptr) {
    OP_CHECK_WRONG_DIMENSION(p, EXPECTED_DIM_ONE, return false);
  }
  if (k != nullptr) {
    OP_CHECK_WRONG_DIMENSION(k, EXPECTED_DIM_ONE, return false);
  }
  if (p != nullptr && p->GetViewShape().GetDim(0) != logits->GetViewShape().GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected p.size(0) is equal to logits.size(0), but got %ld.",
            p->GetViewShape().GetDim(0));
    return false;
  }
  if (k != nullptr && k->GetViewShape().GetDim(0) != logits->GetViewShape().GetDim(0)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected k.size(0) is equal to logits.size(0), but got %ld.",
            k->GetViewShape().GetDim(0));
    return false;
  }
  return true;
}

static bool CheckFormatValid(const aclTensor* logits, const aclTensor* p, const aclTensor *k, const aclTensor* out)
{
  if (logits->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "logits format only support ND");
    return false;
  }
  if (p != nullptr && p->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "p format only support ND");
    return false;
  }
  if (k != nullptr && k->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k format only support ND");
    return false;
  }
  if (out->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out format only support ND");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* logits, const aclTensor* p, const aclTensor *k, const aclTensor* out)
{
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(logits, p, k, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(logits, p, k, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShapeValid(logits, p, k, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查format是否满足约束
  CHECK_RET(CheckFormatValid(logits, p, k, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnApplyTopKTopPCustomGetWorkspaceSize(
    const aclTensor* logits, const aclTensor* p, const aclTensor* k, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnApplyTopKTopPCustom, DFX_IN(logits, p, k), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(logits, p, k, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  bool pIsEmpty = false;
  bool kIsEmpty = false;
  if (p != nullptr) {
    pIsEmpty = p->IsEmpty();
  }
  if (k != nullptr) {
    kIsEmpty = k->IsEmpty();
  }
  if (logits->IsEmpty() || pIsEmpty || kIsEmpty) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  // 固定写法，将输入selfRef转换成连续的tensor
  auto logitsContiguous = l0op::Contiguous(logits, uniqueExecutor.get());
  CHECK_RET(logitsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* pContiguous = nullptr;
  const aclTensor* kContiguous = nullptr;
  if (p != nullptr) {
    pContiguous = l0op::Contiguous(p, uniqueExecutor.get());
    CHECK_RET(pContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (k != nullptr) {
    kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
    CHECK_RET(kContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  bool isLastDimSizeOne = logits->GetViewShape()[DIM_ONE] == 1;
  auto viewCopyResult = logitsContiguous;
  if (isLastDimSizeOne) {
    viewCopyResult = l0op::ViewCopy(logitsContiguous, out, uniqueExecutor.get());
  } else {
    auto sortResult = l0op::Sort(logitsContiguous, -1, false, true, op::DataType::DT_INT32, uniqueExecutor.get());
    const aclTensor* sortedValue = std::get<0>(sortResult);
    CHECK_RET(sortedValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* sortedIndices = std::get<1>(sortResult);
    CHECK_RET(sortedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto res = l0op::ApplyTopKTopPCustom(sortedValue, sortedIndices, pContiguous, kContiguous, uniqueExecutor.get());
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    viewCopyResult = l0op::ViewCopy(res, out, uniqueExecutor.get());
  }
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyTopKTopPCustom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
   L2_DFX_PHASE_2(aclnnApplyTopKTopPCustom);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

