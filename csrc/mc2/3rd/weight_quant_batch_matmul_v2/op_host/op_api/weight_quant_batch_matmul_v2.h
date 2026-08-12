/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *Mc2WeightQuantBatchMatmulV2(const aclTensor *x, const aclTensor *weight,
                                             const aclTensor *antiquantScale, const aclTensor *antiquantOffsetOptional,
                                             const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional,
                                             const aclTensor *biasOptional, bool transposeX, bool transposeWeight,
                                             int antiquantGroupSize, int64_t dtype, int innerPrecise,
                                             aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_WEIGHT_QUANT_BATCH_MATMUL_V2_H
