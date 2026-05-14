/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<aclTensor*, 2> BNTrainingUpdateGrad(const aclTensor* gradOut, const aclTensor* x,
                                                     const aclTensor* saveMean, const aclTensor* saveInvstd, float eps,
                                                     aclOpExecutor* executor);
const std::array<aclTensor*, 2> BN3DTrainingUpdateGrad(const aclTensor* gradOut, const aclTensor* x,
                                                       const aclTensor* saveMean, const aclTensor* saveInvstd,
                                                       float eps, aclOpExecutor* executor);

const aclTensor* BNTrainingReduceGrad(const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight,
                                      const aclTensor* gradBias, const aclTensor* weight, const aclTensor* saveMean,
                                      const aclTensor* saveInvstd, float eps, aclOpExecutor* executor);
const aclTensor* BN3DTrainingReduceGrad(const aclTensor* gradOut, const aclTensor* x, const aclTensor* gradWeight,
                                        const aclTensor* gradBias, const aclTensor* weight, const aclTensor* saveMean,
                                        const aclTensor* saveInvstd, float eps, aclOpExecutor* executor);

const aclTensor* BNInferGrad(const aclTensor* gradOut, const aclTensor* weight, const aclTensor* runningVar, float eps,
                             aclOpExecutor* executor);

constexpr size_t BN_GRAD_V3_OUTPUT_NUM = 3;
const std::array<aclTensor*, BN_GRAD_V3_OUTPUT_NUM> BatchNormGradV3(const aclTensor* gradOut,
                                                                    const aclTensor* input,
                                                                    const aclTensor* weight,
                                                                    const aclTensor* runningMean,
                                                                    const aclTensor* runningVar,
                                                                    const aclTensor* saveMean,
                                                                    const aclTensor* saveInvstd,
                                                                    bool training, float eps,
                                                                    aclOpExecutor* executor);
}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_NORM_GRAD_OP_H_
