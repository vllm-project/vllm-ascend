/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_AICPU_H
#define L0_FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_AICPU_H

#include "opdev/op_executor.h"

namespace FusedInferAttentionScoreV2SinkMetadatal0op {
const aclTensor* FusedInferAttentionScoreV2SinkMetadata(
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDimQk,
    int64_t headDimV,
    int64_t batchSizeOptional,
    int64_t sparseModeOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    char *inputLayoutOptional,
    char *inputLayoutKvOptional,
    int64_t sinkNumOptional,
    int64_t kSinkNumOptional,
    bool batchInvariantOptional,
    int64_t ropeHeadDimOptional,
    int64_t blockSizeOptional,
    const char *socVersion,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    const aclTensor* metaData,
    aclOpExecutor* executor);
} // namespace l0op

#endif
