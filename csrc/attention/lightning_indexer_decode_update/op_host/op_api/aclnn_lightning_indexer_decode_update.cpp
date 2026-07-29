/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_lightning_indexer_decode_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerLightningIndexerDecodeUpdateGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *reqPoolEntries, const aclTensor *cacheSlots,
    const aclTensor *rowModes,
    const aclTensor *actualSeqLengthsKey, const aclTensor *blockTable,
    const aclTensor *topkIndexOut, const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut, const aclTensor *tailInfoOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningIndexerDecodeUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLightningIndexerDecodeUpdateGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlots,
    const aclTensor *rowModes,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *blockTable,
    const aclTensor *topkIndexOut,
    const aclTensor *topkSlotsOut,
    const aclTensor *missCountOut,
    const aclTensor *tailInfoOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerLightningIndexerDecodeUpdateGetWorkspaceSize(
        query, key, weights, reqPoolEntries, cacheSlots, rowModes, actualSeqLengthsKey, blockTable,
        topkIndexOut, topkSlotsOut, missCountOut, tailInfoOut, workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexerDecodeUpdate(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerLightningIndexerDecodeUpdate(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
