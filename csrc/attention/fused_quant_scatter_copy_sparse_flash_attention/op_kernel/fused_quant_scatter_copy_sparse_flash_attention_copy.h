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
 * \file fused_quant_scatter_copy_sparse_flash_attention_copy.h
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_COPY_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_COPY_H

#include "kernel_operator.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_schedule.h"

namespace fused_quant_scatter_copy_sfa {
using namespace AscendC;

__aicore__ inline int64_t RowOffsetInRequest(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint64_t tableRow, int32_t token)
{
    if (token < 0 || static_cast<uint32_t>(token) / CACHE_BLOCK_SIZE >= width) return -1;
    const int32_t block = table.GetValue(tableRow + token / CACHE_BLOCK_SIZE);
    if (block < 0 || static_cast<uint32_t>(block) >= blocks) return -1;
    return (static_cast<int64_t>(block) * CACHE_BLOCK_SIZE + token % CACHE_BLOCK_SIZE) *
        PACKED_KV_ROW_BYTES;
}

__aicore__ inline int64_t RowOffset(GlobalTensor<int32_t> &table, uint32_t width,
    uint32_t blocks, uint32_t batch, int32_t token)
{
    return RowOffsetInRequest(table, width, blocks, static_cast<uint64_t>(batch) * width, token);
}

// The gate is evaluated from device metadata so graph replay may select a
// different path. A heterogeneous batch may contain any mixture of MTP0..15.
__aicore__ inline bool RequiresUnionCopy(GM_ADDR missCounts, GM_ADDR topkMissCounts, GM_ADDR ends,
    GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens,
    const FusedQuantScatterCopySparseFlashAttentionTilingData &td)
{
    GlobalTensor<int32_t> c, m, q, kv, cache;
    c.SetGlobalBuffer((__gm__ int32_t *)missCounts);
    m.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
    q.SetGlobalBuffer((__gm__ int32_t *)ends);
    kv.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKv);
    cache.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
    bool fallback = false;
    int32_t previous = 0;
    for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
        const int32_t n = c.GetValue(b);
        const int32_t end = q.GetValue(b);
        const int32_t actualKvLength = kv.GetValue(b);
        const int32_t cacheTokenCount = cache.GetValue(b);
        ASSERT(n >= 0 && n <= COPY_CAPACITY);
        ASSERT(actualKvLength >= 0 && cacheTokenCount >= 0);
        ASSERT(cacheTokenCount != 0 || n == 0);
        ASSERT(IsValidQuerySegment(
            previous, end, static_cast<int32_t>(td.baseParams.qSeqSize)));
        fallback |= NeedsUnionCopy(n);
        for (int32_t t = previous; t < end; ++t) {
            const int32_t miss = m.GetValue(t);
            ASSERT(miss >= 0 && miss <= TOPK_CAPACITY && (n != 0 || miss == 0));
        }
        previous = end;
    }
    ASSERT(previous == td.baseParams.qSeqSize);
    return fallback;
}

__aicore__ inline void UnionCopy(TPipe &pipe, GM_ADDR hbm, GM_ADDR dram,
    GM_ADDR ht, GM_ADDR dt, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    const FusedQuantScatterCopySparseFlashAttentionTilingData &td)
{
    TBuf<TPosition::VECIN> storage;
    pipe.InitBuffer(storage, 2 * ALIGNED_KV_ROW_BYTES);
    if ASCEND_IS_AIV {
        GlobalTensor<int8_t> hg, dg;
        GlobalTensor<int32_t> htable, dtable, sources, dests, sizes;
        hg.SetGlobalBuffer((__gm__ int8_t *)hbm);
        dg.SetGlobalBuffer((__gm__ int8_t *)dram);
        htable.SetGlobalBuffer((__gm__ int32_t *)ht);
        dtable.SetGlobalBuffer((__gm__ int32_t *)dt);
        sources.SetGlobalBuffer((__gm__ int32_t *)missSrcIds);
        dests.SetGlobalBuffer((__gm__ int32_t *)missDstSlots);
        sizes.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        auto buffers = storage.Get<int8_t>();
        event_t readReady[2], writeDone[2];
        bool wrote[2] = {false, false};
        int64_t pendingDst[2];
        uint32_t issued = 0, finished = 0;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            readReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
            writeDone[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());
        }
        const uint32_t lanes = td.singleCoreParams.usedCoreNum * 2;
        for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
            const int32_t n = sizes.GetValue(b);
            for (uint32_t i = GetBlockIdx(); i < n; i += lanes) {
                const uint64_t p = static_cast<uint64_t>(b) * COPY_CAPACITY + i;
                const auto so = RowOffset(dtable, td.dramMaxBlocks,
                    td.dramPhysicalBlocks, b, sources.GetValue(p));
                const auto dest = RowOffset(htable, td.baseParams.maxBlockNumPerBatch,
                    td.hbmPhysicalBlocks, b, dests.GetValue(p));
                ASSERT(so >= 0 && dest >= 0);
                const uint32_t slot = issued % 2;
                if (wrote[slot]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
                    wrote[slot] = false;
                }
                auto buf = buffers[slot * ALIGNED_KV_ROW_BYTES];
                DataCopyPad(buf, dg[so], DataCopyExtParams{1, PACKED_KV_ROW_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
                pendingDst[slot] = dest;
                SetFlag<HardEvent::MTE2_MTE3>(readReady[slot]);
                ++issued;
                if (issued - finished == 2) {
                    const uint32_t out = finished % 2;
                    WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
                    DataCopyPad(hg[pendingDst[out]], buffers[out * ALIGNED_KV_ROW_BYTES],
                        DataCopyExtParams{1, PACKED_KV_ROW_BYTES, 0, 0, 0});
                    SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
                    wrote[out] = true;
                    ++finished;
                }
            }
        }
        while (finished < issued) {
            const uint32_t out = finished % 2;
            WaitFlag<HardEvent::MTE2_MTE3>(readReady[out]);
            DataCopyPad(hg[pendingDst[out]], buffers[out * ALIGNED_KV_ROW_BYTES],
                DataCopyExtParams{1, PACKED_KV_ROW_BYTES, 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(writeDone[out]);
            wrote[out] = true;
            ++finished;
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (wrote[slot]) WaitFlag<HardEvent::MTE3_MTE2>(writeDone[slot]);
            pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(readReady[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(writeDone[slot]);
        }
    }
    SyncAll<false>();
}

} // namespace fused_quant_scatter_copy_sfa

#endif
