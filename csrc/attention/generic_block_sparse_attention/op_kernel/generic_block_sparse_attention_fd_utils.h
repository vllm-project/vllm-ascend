/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_FD_UTILS_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_FD_UTILS_H

#include "generic_block_sparse_attention_metadata_kernel.h"
#include "kernel_common.hpp"

namespace GsaFd {

__aicore__ inline uint32_t HashConfigValue(uint32_t hash, uint32_t value)
{
    constexpr uint32_t FNV_PRIME = 16777619U;
    for (uint32_t shift = 0; shift < 32U; shift += 8U) {
        hash ^= (value >> shift) & 0xFFU;
        hash *= FNV_PRIME;
    }
    return hash;
}

__aicore__ inline uint32_t CalculateConfigSignature(
    __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData)
{
    uint32_t hash = 2166136261U;
    hash = HashConfigValue(hash, tilingData->batch);
    hash = HashConfigValue(hash, tilingData->numHeads);
    hash = HashConfigValue(hash, tilingData->kvHeads);
    hash = HashConfigValue(hash, tilingData->embeddingSize);
    hash = HashConfigValue(hash, tilingData->blockShapeX);
    hash = HashConfigValue(hash, tilingData->blockShapeY);
    hash = HashConfigValue(hash, tilingData->topK);
    hash = HashConfigValue(hash, tilingData->qBlockNum);
    hash = HashConfigValue(hash, 1U); // Generic kernel currently supports packed GQA only.
    return hash;
}

__aicore__ inline bool ValidateMetadata(
    __gm__ const GsaMetadata::Metadata *meta,
    __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData,
    uint32_t physicalAicNum)
{
    if (meta == nullptr || meta->magic != GsaMetadata::METADATA_MAGIC ||
        meta->version != GsaMetadata::METADATA_VERSION ||
        meta->metadataUsedSize != static_cast<int32_t>(GsaMetadata::METADATA_USED_SIZE) ||
        meta->configSignature != static_cast<int32_t>(CalculateConfigSignature(tilingData)) ||
        meta->saTotalTaskNum < 0 ||
        static_cast<uint64_t>(meta->saTotalTaskNum) >
            static_cast<uint64_t>(tilingData->totalQTokens) * tilingData->kvHeads) {
        return false;
    }

    const uint32_t flags = static_cast<uint32_t>(meta->fdScheduleFlags);
    constexpr uint32_t KNOWN_FLAGS = GsaMetadata::FD_SCHEDULE_ENABLED | GsaMetadata::FD_ACTUAL_BLOCK_PREFIX;
    if ((flags & ~KNOWN_FLAGS) != 0U || (flags & GsaMetadata::FD_ACTUAL_BLOCK_PREFIX) == 0U) {
        return false;
    }
    if ((flags & GsaMetadata::FD_SCHEDULE_ENABLED) == 0U) {
        return meta->fdActiveCoreNum == 0 && meta->combineTaskNum == 0 && meta->fdPartialTaskNum == 0;
    }
    if (meta->fdActiveCoreNum <= 0 ||
        meta->fdActiveCoreNum > static_cast<int32_t>(GsaMetadata::MAX_FD_ACTIVE_CORE_NUM) ||
        meta->fdActiveCoreNum > static_cast<int32_t>(physicalAicNum) || meta->decodePerCoreTaskNum <= 0 ||
        meta->fdTotalFlatTaskNum <= 0 || meta->combineTaskNum < 0 ||
        meta->combineTaskNum > static_cast<int32_t>(GsaMetadata::MAX_COMBINE_TASK_NUM) ||
        meta->fdPartialTaskNum < 0 ||
        (tilingData->fdStaticEnabled != 0U &&
         meta->fdPartialTaskNum > static_cast<int32_t>(tilingData->fdPartialCapacity))) {
        return false;
    }

    for (int32_t core = 0; core < meta->fdActiveCoreNum; ++core) {
        const __gm__ GsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[core];
        if (schedule.baseTaskStart < 0 || schedule.baseTaskEnd < schedule.baseTaskStart ||
            schedule.baseTaskEnd > meta->saTotalTaskNum || schedule.firstBlockStart < 0 ||
            schedule.firstBlockStart > static_cast<int32_t>(tilingData->topK) || schedule.lastBlockEnd < 0 ||
            schedule.lastBlockEnd > static_cast<int32_t>(tilingData->topK)) {
            return false;
        }
    }
    uint32_t scheduledPartialTaskNum = 0U;
    for (int32_t combineIdx = 0; combineIdx < meta->combineTaskNum; ++combineIdx) {
        const __gm__ GsaMetadata::CombineSchedule &combine = meta->combineSchedules[combineIdx];
        if (combine.baseTask < 0 || combine.baseTask >= meta->saTotalTaskNum || combine.firstCore < 0 ||
            combine.partialCount <= 1 ||
            combine.partialCount > static_cast<int32_t>(GsaMetadata::MAX_FD_PARTIAL_PER_BASE_TASK) ||
            combine.partialCount > meta->fdActiveCoreNum ||
            combine.firstCore + combine.partialCount > meta->fdActiveCoreNum || combine.partialStart < 0 ||
            combine.partialCount > meta->fdPartialTaskNum ||
            combine.partialStart > meta->fdPartialTaskNum - combine.partialCount ||
            combine.partialStart != static_cast<int32_t>(scheduledPartialTaskNum)) {
            return false;
        }
        scheduledPartialTaskNum += static_cast<uint32_t>(combine.partialCount);
    }
    return scheduledPartialTaskNum == static_cast<uint32_t>(meta->fdPartialTaskNum);
}

__aicore__ inline bool FindPartialTask(
    __gm__ const GsaMetadata::Metadata *meta, uint32_t baseTask, uint32_t coreIdx,
    uint32_t &partialTaskId, uint32_t &partialCount)
{
    partialTaskId = 0U;
    partialCount = 0U;
    for (int32_t combineIdx = 0; combineIdx < meta->combineTaskNum; ++combineIdx) {
        const __gm__ GsaMetadata::CombineSchedule &combine = meta->combineSchedules[combineIdx];
        if (combine.baseTask != static_cast<int32_t>(baseTask)) {
            continue;
        }
        partialCount = static_cast<uint32_t>(combine.partialCount);
        const uint32_t firstCore = static_cast<uint32_t>(combine.firstCore);
        if (coreIdx < firstCore || coreIdx >= firstCore + partialCount) {
            return false;
        }
        partialTaskId = static_cast<uint32_t>(combine.partialStart) + coreIdx - firstCore;
        return true;
    }
    return false;
}

template <class CuSeqTensor, class SeqUsedTensor>
__aicore__ inline bool DecodeTaskStorage(
    uint32_t taskIdx, uint32_t kvHeads, uint32_t batch, CuSeqTensor &gCuSeqLengths,
    SeqUsedTensor &gSeqUsedQ, bool hasSeqUsedQ, uint32_t &qStorageToken,
    uint32_t &qTokenInBatch, uint32_t &batchIdx, uint32_t &kvHeadIdx)
{
    if (kvHeads == 0U) {
        return false;
    }
    const uint32_t qToken = taskIdx / kvHeads;
    kvHeadIdx = taskIdx % kvHeads;
    uint32_t actualPrefix = 0U;
    for (uint32_t batchLoop = 0; batchLoop < batch; ++batchLoop) {
        const uint32_t storageStart = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchLoop));
        const uint32_t storageEnd = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchLoop + 1U));
        const uint32_t storageLen = storageEnd - storageStart;
        const uint32_t actualLen = hasSeqUsedQ ?
            static_cast<uint32_t>(gSeqUsedQ.GetValue(batchLoop)) : storageLen;
        if (qToken < actualPrefix + actualLen) {
            batchIdx = batchLoop;
            qTokenInBatch = qToken - actualPrefix;
            qStorageToken = storageStart + qTokenInBatch;
            return true;
        }
        actualPrefix += actualLen;
    }
    return false;
}

} // namespace GsaFd

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_FD_UTILS_H
