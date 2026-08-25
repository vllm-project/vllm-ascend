/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
#define ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, int64_t isPackedGQA, const char *qInputLayout, const char *kvInputLayout,
    int64_t maskType, int64_t quantType, int64_t softmaxPrecision, int64_t windowSizeLeft, int64_t windowSizeRight,
    const aclTensor *metadata, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionMetadata(void *workspace,
                                                                                            uint64_t workspaceSize,
                                                                                            aclOpExecutor *executor,
                                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
