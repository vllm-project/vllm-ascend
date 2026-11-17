/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_attn_metadata_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerAttnMetadataUpdateGetWorkspaceSize(
    const aclTensor *reqIds,
    const aclTensor *discardLabel,
    const aclTensor *positions,
    const aclTensor *slotMapping,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    const aclTensor *discardPositions,
    const aclTensor *discardSlotMapping,
    const aclTensor *discardContextLens,
    const aclTensor *discardBlockTables,
    const aclTensor *discardAttnCnt,
    const aclTensor *discardReqIds,
    const aclTensor *originalBs,
    int64_t maxBs,

    int64_t layerIdx,
    const aclTensor *outPositions,
    const aclTensor *outSlots,
    const aclTensor *outSeqLens,
    const aclTensor *outBlockTable,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerAttnMetadataUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnAttnMetadataUpdateGetWorkspaceSize(
    const aclTensor *reqIds,
    const aclTensor *discardLabel,
    const aclTensor *positions,
    const aclTensor *slotMapping,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    const aclTensor *discardPositions,
    const aclTensor *discardSlotMapping,
    const aclTensor *discardContextLens,
    const aclTensor *discardBlockTables,
    const aclTensor *discardAttnCnt,
    const aclTensor *discardReqIds,
    const aclTensor *originalBs,
    int64_t maxBs,
    int64_t layerIdx,
    const aclTensor *outPositions,
    const aclTensor *outSlots,
    const aclTensor *outSeqLens,
    const aclTensor *outBlockTable,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAttnMetadataUpdateGetWorkspaceSize(
        reqIds, discardLabel, positions, slotMapping, blockTable, seqLens,
        discardPositions, discardSlotMapping, discardContextLens, discardBlockTables,
        discardAttnCnt, discardReqIds, originalBs, maxBs, layerIdx,
        outPositions, outSlots, outSeqLens, outBlockTable,
        workspaceSize, executor);
}

aclnnStatus aclnnAttnMetadataUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerAttnMetadataUpdate(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
