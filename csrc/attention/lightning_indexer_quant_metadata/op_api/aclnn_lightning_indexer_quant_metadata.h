/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_QUANT_METADATA_AICPU_H
#define ACLNN_LIGHTNING_INDEXER_QUANT_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLightningIndexerQuantMetadataGetWorkspaceSize
 * parameters :
 * actualSeqLengthsQuery : required
 * actualSeqLengthsKey : required
 * aicCoreNum : required
 * aivCoreNum : required
 * int64_t batchSize : required
 * int64_t querySeqSize : required
 * int64_t queryHeadNum : required
 * int64_t kvSeqSize : required
 * int64_t kvHeadNum : required
 * layoutQuery : optional
 * layoutKey : optional
 * sparseMode : optional
 * socVersion : optional
 * isFd : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnLightningIndexerQuantMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional,
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
    bool isFdOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    int64_t cmpRatioOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/* funtion: aclnnLightningIndexerQuantMetadata
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnLightningIndexerQuantMetadata(void* workspace,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor,
                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_LIGHTNING_INDEXER_QUANT_METADATA_AICPU_H