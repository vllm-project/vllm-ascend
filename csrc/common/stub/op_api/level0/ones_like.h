/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_ONES_LIKE_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_ONES_LIKE_OP_H_
 
#include "opdev/op_executor.h"
 
namespace l0op {
static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_BOOL, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
  op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_UINT8, op::DataType::DT_INT32,
  op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128,
  op::DataType::DT_BF16};
const aclTensor *OnesLike(const aclTensor *self, aclOpExecutor *executor);
inline static bool IsAiCpuSupport(const aclTensor *self) {
  return op::CheckType(self->GetDataType(), AICPU_DTYPE_SUPPORT_LIST);
}
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_ONES_LIKE_OP_H_
