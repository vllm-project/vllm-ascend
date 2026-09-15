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

template <typename T, bool CONDITIONAL_FIRST_FILL = false>
class KvcacheScatterCopyKernel {
public:
    __aicore__ inline KvcacheScatterCopyKernel(TPipe* pipe, const KvcacheScatterCopyTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKRoPE, GM_ADDR hbmKvCache, GM_ADDR dramKRoPE, GM_ADDR dramKvCache,
        GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR srcTokenIds, GM_ADDR dstSlots,
        GM_ADDR copyCounts, GM_ADDR cacheTokens)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        kRopeUbOffset_ = KV_CACHE_UB_BYTES / sizeof(T);
        // Keep one payload in VECOUT while MTE2 prefetches the next random
        // DRAM row into VECIN.  SCATTER has no vector compute with which to
        // hide either transfer, so a two-slot bound queue is the whole
        // software pipeline.
        pipe_->InitBuffer(copyQueue_, 2, KV_CACHE_UB_BYTES + K_ROPE_UB_BYTES);

        hbmKRoPEGm_.SetGlobalBuffer((__gm__ T*)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ T*)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ T*)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ T*)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)dramBlockTable);
        srcTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcTokenIds);
        dstSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)dstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t*)copyCounts);
        if constexpr (CONDITIONAL_FIRST_FILL) {
            cacheTokensGm_.SetGlobalBuffer(
                (__gm__ int32_t*)cacheTokens, tiling_->batchSize);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        if constexpr (CONDITIONAL_FIRST_FILL) {
            if (!IsBatchFirstFill()) {
                return;
            }
        }

        cachedBatchIdx_ = -1;
        cachedCopyCount_ = 0;

        int64_t currentFlatPair = FindNextValidPair(blockIdx_);
        CopyAddress currentAddress;
        while (currentFlatPair < tiling_->totalPairSlots &&
               !ResolveAddress(currentFlatPair, currentAddress)) {
            currentFlatPair = FindNextValidPair(currentFlatPair + tiling_->usedCoreNum);
        }
        if (currentFlatPair >= tiling_->totalPairSlots) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            int64_t nextFlatPair = FindNextValidPair(currentFlatPair + tiling_->usedCoreNum);
            CopyAddress nextAddress;
            while (nextFlatPair < tiling_->totalPairSlots &&
                   !ResolveAddress(nextFlatPair, nextAddress)) {
                nextFlatPair = FindNextValidPair(nextFlatPair + tiling_->usedCoreNum);
            }

            const bool hasNext = nextFlatPair < tiling_->totalPairSlots;
            if (hasNext) {
                // Issue the next DRAM->UB transfer before draining the current
                // UB->HBM payload.  The two queue slots keep both operations
                // in flight without changing per-core destination ownership.
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            currentFlatPair = nextFlatPair;
            currentAddress = nextAddress;
        }
    }

private:
    __aicore__ inline bool IsBatchFirstFill()
    {
        bool firstFill = false;
        for (int64_t batchIdx = 0; batchIdx < tiling_->batchSize; ++batchIdx) {
            const int32_t copyCount = copyCountsGm_.GetValue(batchIdx);
            const int32_t cacheTokenCount = cacheTokensGm_.GetValue(batchIdx);
            ASSERT_MSG(copyCount >= 0 && copyCount <= tiling_->copyCap,
                "first-fill copy_count exceeds the request-level input capacity.");
            ASSERT_MSG(cacheTokenCount >= 0,
                "num_cache_tokens must be non-negative.");
            if (copyCount >= cacheTokenCount) {
                firstFill = true;
            }
        }
        return firstFill;
    }

    struct CopyAddress {
        int64_t srcKv = 0;
        int64_t dstKv = 0;
        int64_t srcRope = 0;
        int64_t dstRope = 0;
    };

    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        int64_t steps = CeilDiv(start - blockIdx_, static_cast<int64_t>(tiling_->usedCoreNum));
        return blockIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline int64_t FindNextValidPair(int64_t flatPairIdx)
    {
        while (flatPairIdx < tiling_->totalPairSlots) {
            int64_t batchIdx = flatPairIdx / tiling_->copyCap;
            int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * tiling_->copyCap);
            if (batchIdx != cachedBatchIdx_) {
                cachedCopyCount_ = copyCountsGm_.GetValue(batchIdx);
                ASSERT_MSG(cachedCopyCount_ >= 0 && cachedCopyCount_ <= tiling_->copyCap,
                    "copy_count exceeds the SCATTER input capacity.");
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < cachedCopyCount_) {
                return flatPairIdx;
            }
            flatPairIdx = FirstFlatPairAtOrAfter((batchIdx + 1) * tiling_->copyCap);
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool ResolveAddress(int64_t flatPairIdx, CopyAddress& address)
    {
        int64_t batchIdx = flatPairIdx / tiling_->copyCap;
        int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * tiling_->copyCap);
        int64_t pairOffset = batchIdx * tiling_->copyCap + copyIdx;
        int32_t srcTokenId = srcTokenIdsGm_.GetValue(pairOffset);
        int32_t dstSlot = dstSlotsGm_.GetValue(pairOffset);
        ASSERT_MSG(srcTokenId >= 0 && dstSlot >= 0, "active src_token_ids and dst_slots must be non-negative.");
        if (srcTokenId < 0 || dstSlot < 0) {
            return false;
        }

        int64_t srcBlockCol = static_cast<int64_t>(srcTokenId) >> BLOCK_SHIFT;
        int64_t srcBlockOffset = static_cast<int64_t>(srcTokenId) & BLOCK_MASK;
        int64_t dstBlockCol = static_cast<int64_t>(dstSlot) >> BLOCK_SHIFT;
        int64_t dstBlockOffset = static_cast<int64_t>(dstSlot) & BLOCK_MASK;
        ASSERT_MSG(srcBlockCol < tiling_->dramMaxBlockNum && dstBlockCol < tiling_->hbmMaxBlockNum,
            "active source token or destination slot exceeds its block table.");
        if (srcBlockCol >= tiling_->dramMaxBlockNum || dstBlockCol >= tiling_->hbmMaxBlockNum) {
            return false;
        }

        int32_t srcPhysicalBlock =
            dramBlockTableGm_.GetValue(batchIdx * tiling_->dramMaxBlockNum + srcBlockCol);
        int32_t dstPhysicalBlock =
            hbmBlockTableGm_.GetValue(batchIdx * tiling_->hbmMaxBlockNum + dstBlockCol);
        ASSERT_MSG(srcPhysicalBlock >= 0 && dstPhysicalBlock >= 0, "block table entries must be non-negative.");
        if (srcPhysicalBlock < 0 || dstPhysicalBlock < 0) {
            return false;
        }

        address.srcKv =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE + srcBlockOffset) * KV_CACHE_DIM;
        address.dstKv =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE + dstBlockOffset) * KV_CACHE_DIM;
        address.srcRope =
            (static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE + srcBlockOffset) * K_ROPE_DIM;
        address.dstRope =
            (static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE + dstBlockOffset) * K_ROPE_DIM;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress& address)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};

        DataCopyPad(local, dramKvCacheGm_[address.srcKv], kvParams, padParams);
        DataCopyPad(local[kRopeUbOffset_], dramKRoPEGm_[address.srcRope], ropeParams, padParams);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress& address)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};

        DataCopyPad(hbmKvCacheGm_[address.dstKv], local, kvParams);
        DataCopyPad(hbmKRoPEGm_[address.dstRope], local[kRopeUbOffset_], ropeParams);
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
    int64_t cachedBatchIdx_ = -1;
    int32_t cachedCopyCount_ = 0;

    GlobalTensor<T> hbmKRoPEGm_;
    GlobalTensor<T> hbmKvCacheGm_;
    GlobalTensor<T> dramKRoPEGm_;
    GlobalTensor<T> dramKvCacheGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> srcTokenIdsGm_;
    GlobalTensor<int32_t> dstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};

} // namespace KvcacheScatterCopyNs
#endif // KVCACHE_SCATTER_COPY_KERNEL_H
