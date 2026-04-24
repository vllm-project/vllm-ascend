/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H
#define L0_QUANT_LIGHTNING_INDEXER_METADATA_AICPU_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* QuantLightningIndexerMetadata(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    const char* socVersion,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKOptional,
    char* layoutQueryOptional,
    char* layoutKeyOptional,
    int64_t sparseCountOptional,
    int64_t sparseModeOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    int64_t cmpRatioOptional,
    const aclTensor* metaData,
    aclOpExecutor* executor);
} // namespace l0op

#endif
    