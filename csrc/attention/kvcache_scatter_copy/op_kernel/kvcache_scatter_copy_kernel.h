/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef KVCACHE_SCATTER_COPY_KERNEL_H
#define KVCACHE_SCATTER_COPY_KERNEL_H

#include "kernel_operator.h"

namespace KvcacheScatterCopyNs {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BLOCK_SHIFT = 7;
constexpr int64_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;
constexpr int64_t K_ROPE_UB_BYTES = K_ROPE_DIM * sizeof(uint16_t);
constexpr int64_t KV_CACHE_UB_BYTES = KV_CACHE_DIM * sizeof(uint16_t);

template <typename T>
class KvcacheScatterCopyKernel {
public:
    __aicore__ inline KvcacheScatterCopyKernel(TPipe* pipe, const KvcacheScatterCopyTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKRoPE, GM_ADDR hbmKvCache, GM_ADDR dramKRoPE, GM_ADDR dramKvCache,
        GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR srcTokenIds, GM_ADDR dstSlots,
        GM_ADDR copyCounts)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        kRopeUbOffset_ = KV_CACHE_UB_BYTES / sizeof(T);
        pipe_->InitBuffer(copyQueue_, 1, KV_CACHE_UB_BYTES + K_ROPE_UB_BYTES);

        hbmKRoPEGm_.SetGlobalBuffer((__gm__ T*)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ T*)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ T*)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ T*)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)dramBlockTable);
        srcTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcTokenIds);
        dstSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)dstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t*)copyCounts);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        int64_t cachedBatchIdx = -1;
        int32_t cachedCopyCount = 0;
        int64_t flatPairIdx = blockIdx_;
        while (flatPairIdx < tiling_->totalPairSlots) {
            int64_t batchIdx = flatPairIdx / tiling_->copyCap;
            int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * tiling_->copyCap);
            if (batchIdx != cachedBatchIdx) {
                cachedCopyCount = copyCountsGm_.GetValue(batchIdx);
                ASSERT_MSG(cachedCopyCount >= 0 && cachedCopyCount <= tiling_->copyCap,
                    "copy_count must be in [0, 16384].");
                cachedBatchIdx = batchIdx;
            }

            if (copyIdx >= cachedCopyCount) {
                flatPairIdx = FirstFlatPairAtOrAfter((batchIdx + 1) * tiling_->copyCap);
                continue;
            }

            CopyOne(batchIdx, copyIdx);
            flatPairIdx += tiling_->usedCoreNum;
        }
    }

private:
    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        int64_t steps = CeilDiv(start - blockIdx_, static_cast<int64_t>(tiling_->usedCoreNum));
        return blockIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline void CopyOne(int64_t batchIdx, int32_t copyIdx)
    {
        int64_t pairOffset = batchIdx * tiling_->copyCap + copyIdx;
        int32_t srcTokenId = srcTokenIdsGm_.GetValue(pairOffset);
        int32_t dstSlot = dstSlotsGm_.GetValue(pairOffset);
        ASSERT_MSG(srcTokenId >= 0 && dstSlot >= 0, "active src_token_ids and dst_slots must be non-negative.");
        if (srcTokenId < 0 || dstSlot < 0) {
            return;
        }

        int64_t srcBlockCol = static_cast<int64_t>(srcTokenId) >> BLOCK_SHIFT;
        int64_t srcBlockOffset = static_cast<int64_t>(srcTokenId) & BLOCK_MASK;
        int64_t dstBlockCol = static_cast<int64_t>(dstSlot) >> BLOCK_SHIFT;
        int64_t dstBlockOffset = static_cast<int64_t>(dstSlot) & BLOCK_MASK;
        ASSERT_MSG(srcBlockCol < tiling_->dramMaxBlockNum && dstBlockCol < tiling_->hbmMaxBlockNum,
            "active source token or destination slot exceeds its block table.");
        if (srcBlockCol >= tiling_->dramMaxBlockNum || dstBlockCol >= tiling_->hbmMaxBlockNum) {
            return;
        }

        int32_t srcPhysicalBlock =
            dramBlockTableGm_.GetValue(batchIdx * tiling_->dramMaxBlockNum + srcBlockCol);
        int32_t dstPhysicalBlock =
            hbmBlockTableGm_.GetValue(batchIdx * tiling_->hbmMaxBlockNum + dstBlockCol);
        ASSERT_MSG(srcPhysicalBlock >= 0 && dstPhysicalBlock >= 0, "block table entries must be non-negative.");
        if (srcPhysicalBlock < 0 || dstPhysicalBlock < 0) {
            return;
        }

        int64_t srcKvAddr =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE + srcBlockOffset) * KV_CACHE_DIM;
        int64_t dstKvAddr =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE + dstBlockOffset) * KV_CACHE_DIM;
        int64_t srcRopeAddr =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE + srcBlockOffset) * K_ROPE_DIM;
        int64_t dstRopeAddr =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE + dstBlockOffset) * K_ROPE_DIM;

        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};

        DataCopyPad(local, dramKvCacheGm_[srcKvAddr], kvParams, padParams);
        DataCopyPad(local[kRopeUbOffset_], dramKRoPEGm_[srcRopeAddr], ropeParams, padParams);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<T>();
        DataCopyPad(hbmKvCacheGm_[dstKvAddr], local, kvParams);
        DataCopyPad(hbmKRoPEGm_[dstRopeAddr], local[kRopeUbOffset_], ropeParams);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

private:
    TPipe* pipe_;
    const KvcacheScatterCopyTilingData* tiling_;
    int32_t blockIdx_ = -1;
    int32_t kRopeUbOffset_ = 0;

    GlobalTensor<T> hbmKRoPEGm_;
    GlobalTensor<T> hbmKvCacheGm_;
    GlobalTensor<T> dramKRoPEGm_;
    GlobalTensor<T> dramKvCacheGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> srcTokenIdsGm_;
    GlobalTensor<int32_t> dstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> copyQueue_;
};

} // namespace KvcacheScatterCopyNs
#endif // KVCACHE_SCATTER_COPY_KERNEL_H
