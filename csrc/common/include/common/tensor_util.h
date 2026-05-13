/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"

namespace op {
const aclIntArray* getAllDims(const aclTensor* self, aclOpExecutor* executor);

const aclTensor* ResizeFrom1D(const aclTensor* cdim, const aclTensor* input, bool isSupportNcdhw,
                              aclOpExecutor* executor);

const aclTensor* ResizeTo1D(const aclTensor* result, const aclTensor* output, bool isSupportNcdhw,
                            aclOpExecutor* executor);

const aclTensor* ResizeFromND(const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeToND(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeFrom5D(const aclTensor* input, aclOpExecutor* executor);

const aclTensor* ResizeTo5D(const aclTensor* output, const aclTensor* input, aclOpExecutor* executor);

aclTensor* FillScalar(int64_t dim, int value, aclOpExecutor* executor);

aclnnStatus ProcessEmptyTensorWithValue(aclTensor* src, float initValue, aclOpExecutor* executor);

op::DataType CombineCategories(op::DataType higher, op::DataType lower);
}  // namespace op

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus BatchNorm(const aclTensor* input, const aclTensor* weight, const aclTensor* bias, aclTensor* runningMean,
                      aclTensor* runningVar, bool training, float momentum, float eps, aclTensor** output,
                      aclTensor* saveMean, aclTensor* saveInvstd, aclOpExecutor* executor);

#ifdef __cplusplus
}
#endif