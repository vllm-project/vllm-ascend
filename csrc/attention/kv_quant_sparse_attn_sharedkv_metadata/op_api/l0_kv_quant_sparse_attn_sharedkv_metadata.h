/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H
#define L0_KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_AICPU_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* KvQuantSparseAttnSharedkvMetadata(
    const aclTensor* cuSeqLensQOptional,
    const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional,
    const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional,
    int64_t oriTopKOptional,
    int64_t cmpTopKOptional,
    int64_t kvQuantMode,
    int64_t tileSizeOptional,
    int64_t ropeHeadDimOptional,
    int64_t cmpRatioOptional,
    int64_t oriMaskModeOptional,
    int64_t cmpMaskModeOptional,
    int64_t oriWinLeftOptional,
    int64_t oriWinRightOptional,
    std::string layoutQOptional,
    std::string layoutKvOptional,
    bool hasOriKvOptional,
    bool hasCmpKvOptional,
    const char *socVersion,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    const aclTensor* metaData,
    aclOpExecutor* executor);
} // namespace l0op

#endif
