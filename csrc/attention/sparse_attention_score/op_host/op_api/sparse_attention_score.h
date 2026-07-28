/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_H_
#define SPARSE_ATTENTION_SCORE_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 2> SparseAttentionScore(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *selectIdx,
    const aclTensor *blockTable,
    const aclTensor *selectNumIdxOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclOpExecutor *executor);

} // namespace l0op

#endif  // SPARSE_ATTENTION_SCORE_H_
