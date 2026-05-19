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
 * \file aclnn_scatter_nd_update_v2.cpp
 * \brief
 */

#include "aclnn_scatter_nd_update_v2.h"
#include "scatter_nd_update_v2.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_INT16,
    op::DataType::DT_BF16, op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,
    op::DataType::DT_BF16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                             op::DataType::DT_INT32};

static bool CheckNotNull(aclTensor *varRef, const aclTensor *indices, const aclTensor *updates) {
  OP_CHECK_NULL(varRef, return false);
  OP_CHECK_NULL(indices, return false);
  OP_CHECK_NULL(updates, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  // if (Ops::NN::AclnnUtil::IsRegbase()) {
  //   return ASCEND950_DTYPE_DTYPE_SUPPORT_LIST;
  // }
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(aclTensor *varRef, const aclTensor *indices, const aclTensor *updates) {
  // 检查self的数据类型是否在算子的支持列表内
  auto supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(varRef, supportList, return false);
  // 检查index的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
  // varRef和updates的数据类型要一致
  if (varRef->GetDataType() != updates->GetDataType()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updates dtype %s should be in same with varRef dtype %s.",
            op::ToString(updates->GetDataType()).GetString(), op::ToString(varRef->GetDataType()).GetString());
    return false;
  }

  return true;
}

static aclnnStatus CheckParams(aclTensor *varRef, const aclTensor *indices, const aclTensor *updates) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(varRef, indices, updates), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(varRef, indices, updates), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnScatterNdUpdateV2GetWorkspaceSize(aclTensor *varRef, const aclTensor *indices, const aclTensor *updates,
                                                 const aclIntArray *strides, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnScatterNdUpdateV2, DFX_IN(varRef, indices, updates), DFX_OUT(varRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(varRef, indices, updates);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (varRef->IsEmpty() || indices->IsEmpty() || updates->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入varRef转换成连续的tensor
  // auto varRefContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
  // CHECK_RET(varRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  varRef->SetStorageShape(varRef->GetViewShape());
  // 将输入indices转换成连续的tensor
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入updates转换成连续的tensor
  auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
  CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 执行L0算子 (kernel直接原地更新varRef, 无需Contiguous和ViewCopy)
  auto scatterUpdateRes = l0op::ScatterNdUpdateV2(varRef, indicesContiguous, updatesContiguous, strides, false, uniqueExecutor.get());
  CHECK_RET(scatterUpdateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnScatterNdUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnScatterNdUpdateV2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif