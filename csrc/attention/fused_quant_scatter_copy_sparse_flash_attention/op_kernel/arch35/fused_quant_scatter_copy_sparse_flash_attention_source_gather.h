/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_scatter_copy_sparse_flash_attention_source_gather.h
 * \brief
 */

// Included inside BaseApi, after QSFAVectorService is fully defined.
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::EnqueueSourceAwareKv(
    uint32_t indexBase, int64_t first, uint32_t count, int64_t *writebacks)
{
    auto packed = stage0InQue.template AllocTensor<uint8_t>();
    const DataCopyExtParams params{1, fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES, 0, 0, 0};
    const DataCopyPadExtParams<uint8_t> pad{
        true, 0, fused_quant_scatter_copy_sfa::KV_ROW_PADDING_BYTES, 0};
    for (uint32_t j = 0; j < count; ++j) {
        // Mapping is constant across this contiguous AIV slice, not per token.
        const uint32_t index = indexBase + first + j;
        writebacks[j] = -1;
        if (index < queryMisses && index < prefetchedUntil && OwnsMiss(index)) {
            // Prefetch already resolved ownership and persisted the owner's row.
            // Do not reload source IDs/block tables just to read our workspace.
            DataCopyPad(packed[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES],
                prefetchGm[prefetchRow + index * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], params, pad);
            continue;
        }
        const int32_t slot = SparseIndicesGm.GetValue(queryRow + index);
        if (index < queryMisses) {
            ASSERT(slot >= 0 && slot < fused_quant_scatter_copy_sfa::OWNER_SLOT_CAPACITY);
            const int32_t source = sourcesGm.GetValue(queryRow + index);
            const int64_t offset = fused_quant_scatter_copy_sfa::RowOffsetInRequest(dramTableGm, copyTiling->dramMaxBlocks,
                copyTiling->dramPhysicalBlocks, dramTableRow, source);
            ASSERT(offset >= 0);
            DataCopyPad(packed[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], dramGm[offset], params, pad);
            if ((earlierMissSlots[slot / fused_quant_scatter_copy_sfa::BITMAP_WORD_BITS] &
                (1U << (slot % fused_quant_scatter_copy_sfa::BITMAP_WORD_BITS))) == 0) {
                writebacks[j] = fused_quant_scatter_copy_sfa::RowOffsetInRequest(blockTableGm,
                    copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
                ASSERT(writebacks[j] >= 0);
            }
        } else {
            const int64_t offset = fused_quant_scatter_copy_sfa::RowOffsetInRequest(blockTableGm,
                copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
            ASSERT(offset >= 0);
            DataCopyPad(packed[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], keyGm[offset], params, pad);
        }
    }
    stage0InQue.EnQue(packed);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessSourceAwareKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, uint32_t indexBase)
{
    PrepareOwners(runInfo.boIdx, queryIndex);
    int64_t writebacks[fused_quant_scatter_copy_sfa::SOURCE_GATHER_BUFFER_COUNT]
        [fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS];
    uint32_t lane = 0;
    EnqueueSourceAwareKv(indexBase, sparseS2Start,
        Min(static_cast<int64_t>(fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS), sparseS2End - sparseS2Start),
        writebacks[lane]);
    for (int64_t first = sparseS2Start; first < sparseS2End;
        first += fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS) {
        const uint32_t count = Min(
            static_cast<int64_t>(fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS), sparseS2End - first);
        auto packed = stage0InQue.template DeQue<uint8_t>();
        const auto readEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        // Fence this batch BEFORE issuing the next MTE2 read: its writeback must
        // not wait for the following DRAM batch. Queue events also order MTE2->V.
        SetFlag<HardEvent::MTE2_MTE3>(readEvent);
        WaitFlag<HardEvent::MTE2_MTE3>(readEvent);
        if (first + fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS < sparseS2End) {
            EnqueueSourceAwareKv(indexBase, first + fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS,
                Min(static_cast<int64_t>(fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS),
                    sparseS2End - first - fused_quant_scatter_copy_sfa::SOURCE_GATHER_ROWS),
                writebacks[lane ^ 1U]);
        }
        bool wrote = false;
        // MTE3 persistence and V dequantization both consume the same DRAM read.
        // Their completion is joined before this UB buffer can be reused.
        for (uint32_t j = 0; j < count; ++j) {
            if (writebacks[lane][j] < 0) continue;
            // Only coalesce adjacent owner rows in this bank. Prefetched rows,
            // earlier-query owners and hits retain -1 and must not be persisted.
            const auto plan = fused_quant_scatter_copy_sfa::PlanMissWritebackPair(writebacks[lane][j],
                j + 1 < count ? writebacks[lane][j + 1] : -1);
            DataCopyPad(keyGm[writebacks[lane][j]],
                packed[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES],
                DataCopyExtParams{plan.blockCount, fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES,
                    0, plan.dstStride, 0});
            j += plan.blockCount - 1;
            wrote = true;
        }
        auto dequant = stage0OutQue.template AllocTensor<Q_T>();
        DequantKv(dequant, packed, count, constInfo);
        if (wrote) {
            const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(event);
            WaitFlag<HardEvent::MTE3_MTE2>(event);
        }
        stage0InQue.FreeTensor(packed);
        stage0OutQue.EnQue(dequant);
        dequant = stage0OutQue.template DeQue<Q_T>();
        if constexpr (IS_SPLIT_G) CopyOutKvUb2Gm(v0ResGm, dequant, count, first, runInfo, constInfo);
        else CopyOutKvUb2L1(outputL1, dequant, count, first, runInfo, constInfo);
        stage0OutQue.FreeTensor(dequant);
        lane ^= 1U;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::SubmitMissPrefetch(
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!sourceAware) return;
    const uint32_t query = queryIndex;
    const uint32_t miss = queryMisses;
    if (miss == 0 || miss > fused_quant_scatter_copy_sfa::FUSED_MISS_CAPACITY) return;
    // Only prefetch during the hit chunks, before this query reaches its miss chunk.
    const uint32_t logicalChunk = runInfo.s2LoopCount / fused_quant_scatter_copy_sfa::GATHER_TILES_PER_REGION;
    const uint32_t missChunk = fused_quant_scatter_copy_sfa::GATHER_REGION_COUNT - 1 -
        runInfo.boIdx % fused_quant_scatter_copy_sfa::GATHER_BATCH_PERIOD;
    if (logicalChunk >= missChunk) return;
    PrepareOwners(runInfo.boIdx, query);
    auto io = missPrefetchBuf.template Get<uint8_t>();
    uint32_t scan = prefetchedUntil;
    while (scan < miss && pendingRows < fused_quant_scatter_copy_sfa::MISS_PREFETCH_ROWS) {
        const uint32_t index = scan++;
        if (!OwnsMiss(index)) continue;
        const uint64_t p = queryRow + index;
        const int32_t source = sourcesGm.GetValue(p), slot = SparseIndicesGm.GetValue(p);
        const auto offset = fused_quant_scatter_copy_sfa::RowOffsetInRequest(dramTableGm, copyTiling->dramMaxBlocks,
            copyTiling->dramPhysicalBlocks, dramTableRow, source);
        ASSERT(offset >= 0 && slot >= 0 && slot < fused_quant_scatter_copy_sfa::OWNER_SLOT_CAPACITY);
        pendingIndices[pendingRows] = index;
        pendingDestinations[pendingRows] = -1;
        if ((earlierMissSlots[slot / fused_quant_scatter_copy_sfa::BITMAP_WORD_BITS] &
            (1U << (slot % fused_quant_scatter_copy_sfa::BITMAP_WORD_BITS))) == 0) {
            pendingDestinations[pendingRows] = fused_quant_scatter_copy_sfa::RowOffsetInRequest(blockTableGm,
                copyTiling->baseParams.maxBlockNumPerBatch, copyTiling->hbmPhysicalBlocks, hbmTableRow, slot);
            ASSERT(pendingDestinations[pendingRows] >= 0);
        }
        DataCopyPad(io[pendingRows * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], dramGm[offset],
            DataCopyExtParams{1, fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES, 0, 0, 0},
            DataCopyPadExtParams<uint8_t>{true, 0, fused_quant_scatter_copy_sfa::KV_ROW_PADDING_BYTES, 0});
        ++pendingRows;
    }
    pendingEnd = scan;
    if (pendingRows) {
        missReadEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(missReadEvent);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CompleteMissPrefetch()
{
    if (!pendingRows) return;
    WaitFlag<HardEvent::MTE2_MTE3>(missReadEvent);
    auto io = missPrefetchBuf.template Get<uint8_t>();
    for (uint32_t j = 0; j < pendingRows; ++j) {
        const DataCopyExtParams params{
            1, fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES, 0, 0, 0};
        DataCopyPad(prefetchGm[prefetchRow +
            pendingIndices[j] * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES],
            io[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], params);
        if (pendingDestinations[j] >= 0) {
            DataCopyPad(keyGm[pendingDestinations[j]],
                io[j * fused_quant_scatter_copy_sfa::ALIGNED_KV_ROW_BYTES], params);
        }
    }
    const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    prefetchedUntil = pendingEnd;
    pendingRows = 0;
}
