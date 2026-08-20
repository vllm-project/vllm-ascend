/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dense_attention_score.h"
#include "sparse_attention_score.h"

namespace l0op {

const std::array<const aclTensor *, 2> DenseAttentionScore(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *actualSeqLengths,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *qDequantScale,
    const aclTensor *kDequantScale,
    const aclTensor *vDequantScale,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t innerPrecise,
    const char *inputLayout,
    const aclTensor *attentionOut,
    aclOpExecutor *executor)
{
    // Dense mode does not expose or consume sparse selection metadata. topK is
    // retained only as a compatibility placeholder in the shared GE op.
    constexpr int64_t DENSE_TOP_K_PLACEHOLDER = 0;
    // GE compacts an omitted optional input when it precedes a required input.
    // Keep the shared op's legacy input slots stable with an existing int32
    // tensor. Host and Kernel both ignore these two slots when isDense=true.
    const aclTensor *unusedSparseMetadata = blockTable;
    return SparseAttentionScore_950(
        query, key, value, unusedSparseMetadata, blockTable, unusedSparseMetadata,
        actualSeqLengths, actualSeqLengthsKv,
        qDequantScale, kDequantScale, vDequantScale,
        numKeyValueHeads, scaleValue, blockSize, DENSE_TOP_K_PLACEHOLDER,
        innerPrecise, inputLayout, true, attentionOut, executor);
}

} // namespace l0op
