/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ATTN_METADATA_UPDATE_H
#define OP_API_INC_ATTN_METADATA_UPDATE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First-stage interface: calculates workspace size for attn_metadata_update.
 *
 * Merges current-batch attention metadata (positions, slot_mapping, seq_lens,
 * block_table) with per-layer discard metadata to produce unified tensors of
 * shape (max_bs,) covering both active and discarded requests.
 *
 * @param [in]  reqIds               (max_bs,) int32 – request IDs for each slot.
 * @param [in]  discardLabel         (max_bs,) int32 – 1 if the slot is discarded.
 * @param [in]  positions            (original_bs,) int64 – positions of scheduled batch.
 * @param [in]  slotMapping          (original_bs,) int32 – KV-cache slot mapping.
 * @param [in]  blockTable           (original_bs, table_size) int32.
 * @param [in]  seqLens              (original_bs,) int32 – sequence lengths.
 * @param [in]  discardPositions     (layer_num, max_discard_bs) int64.
 * @param [in]  discardSlotMapping   (layer_num, max_discard_bs) int32.
 * @param [in]  discardContextLens   (layer_num, max_discard_bs) int32.
 * @param [in]  discardBlockTables   (layer_num, max_discard_bs, table_size) int32.
 * @param [in]  discardAttnCnt       (layer_num,) int32 – current discard count per layer.
 * @param [in]  discardReqIds        (layer_num, max_discard_bs) int32 – req_id array of the discard bucket.
 * @param [in]  maxBs                Maximum batch capacity.
 * @param [in]  originalBs           Scheduled batch size.
 * @param [in]  layerIdx             Current layer index.
 * @param [out] outPositions         (max_bs,) int64.
 * @param [out] outSlots             (max_bs,) int32.
 * @param [out] outSeqLens           (max_bs,) int32.
 * @param [out] outBlockTable        (max_bs, table_size) int32.
 * @param [out] workspaceSize        Workspace size to allocate on NPU.
 * @param [out] executor             Op executor.
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAttnMetadataUpdateGetWorkspaceSize(
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

/**
 * @brief Second-stage interface: executes attn_metadata_update computation.
 * @param [in] workspace      Workspace memory on NPU.
 * @param [in] workspaceSize  Workspace size from GetWorkspaceSize.
 * @param [in] executor       Op executor.
 * @param [in] stream         ACL stream.
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAttnMetadataUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ATTN_METADATA_UPDATE_H
