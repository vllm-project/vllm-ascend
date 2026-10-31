/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_MLA_METADATA_H
#define ACLNN_SPARSE_FLASH_MLA_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, const aclTensor *metaData, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_MLA_METADATA_H
