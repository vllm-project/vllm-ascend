/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_lightning_indexer_decode_update_H
#define ACLNN_lightning_indexer_decode_update_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexerDecodeUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_lightning_indexer_decode_update_H
