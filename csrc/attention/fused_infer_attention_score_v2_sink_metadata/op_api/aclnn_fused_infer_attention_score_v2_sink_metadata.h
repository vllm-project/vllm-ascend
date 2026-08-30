/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_AICPU_H
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV2SinkMetadataGetWorkspaceSize(
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
    int64_t ropeHeadDimOptional,
    int64_t blockSizeOptional,
    int64_t aicCoreNumOptional,
    int64_t aivCoreNumOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV2SinkMetadata(void* workspace,
                                uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA_AICPU_H
