/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_INC_EXTERNAL_ACLNN_KERNELS_RESHAPE_H
#define COMMON_INC_EXTERNAL_ACLNN_KERNELS_RESHAPE_H

#include "opdev/shape_utils.h"
#include "opdev/op_def.h"

namespace l0op {
/**
 * @brief Modify input tensor's shape.
 * @param x Input Tensor. Should be contiguous.
 * @param shape Target Shape. Only one dimension can be -1.
 * @param executor aclOpExecutor.ldd
 * @return *aclTensor Output tensor.
 */
const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor);

/**
 * @brief Modify input tensor's shape.
 * @param x Input Tensor. Should be contiguous.
 * @param shape Target Shape. Only one dimension can be -1.
 * @param executor aclOpExecutor.
 * @return *aclTensor Output tensor.
 */
const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor);
} // namespace l0op

#endif // COMMON_INC_EXTERNAL_ACLNN_KERNELS_RESHAPE_H
