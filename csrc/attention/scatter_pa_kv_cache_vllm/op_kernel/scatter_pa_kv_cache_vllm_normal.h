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
 * \file scatter_pa_kv_cache_vllm_normal.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_NORMAL_H
#define SCATTER_PA_KV_CACHE_VLLM_NORMAL_H

#include "scatter_pa_kv_cache_vllm_normal_common.h"

namespace ScatterPaKvCacheVllm {
using namespace AscendC;

template <typename T1, typename T2, typename IndexDtype, bool NCT = false>
class ScatterPaKvCacheVllmNormal : public ScatterPaKvCacheVllmNormalCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmNormal(TPipe *pipe, const ScatterPaKvCacheVllmTilingData *__restrict__ tilingData)
    : ScatterPaKvCacheVllmNormalCommon(pipe, tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        int64_t coreNums = GetBlockNum(); // 用的核数
        int64_t blockId = GetBlockIdx(); //当前核
        curBlockFactor_ =
            (blockId == coreNums - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
        startTaskId_ = blockId * tilingData_->blockFactor; //当前核起始token

        keyInGm_.SetGlobalBuffer((__gm__ T1 *)keyIn);
        keyCacheOutGm_.SetGlobalBuffer((__gm__ T1 *)keyCacheOut);
        slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)slotMapping);
        valueInGm_.SetGlobalBuffer((__gm__ T2 *)valueIn);
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T2 *)valueCacheOut);
    }

    // 通用分支
    __aicore__ inline void Process()
    {
        usedUbSize_ = tilingData_->ubSize / BUFFER_NUM;
        pipe_->InitBuffer(queBind_, BUFFER_NUM, usedUbSize_);
        // key
        uint64_t kTokenSize = static_cast<uint64_t>(tilingData_->numHead) * tilingData_->kHeadSize; // Key一个token大小
        int64_t kUbLoop = kTokenSize * sizeof(T1) / usedUbSize_;
        int64_t kUbTail = kTokenSize * sizeof(T1) % usedUbSize_;
        // value
        uint64_t vTokenSize = static_cast<uint64_t>(tilingData_->numHead) * tilingData_->vHeadSize; // Value一个token大小
        int64_t vUbLoop = vTokenSize * sizeof(T2) / usedUbSize_;
        int64_t vUbTail = vTokenSize * sizeof(T2) % usedUbSize_;

        for (int64_t i = 0; i < curBlockFactor_; i++) {
            int64_t slotValue = slotMappingGm_.GetValue(startTaskId_ + i);
            if (slotValue < 0) {
                continue;
            }
            // key
            uint64_t kOffsetIn = (startTaskId_ + i) * kTokenSize;
            uint64_t kOffsetOut = GetCacheOffset(slotValue, kTokenSize, tilingData_->kCacheBlockStride);
            // value
            uint64_t vOffsetIn = (startTaskId_ + i) * vTokenSize;
            uint64_t vOffsetOut = GetCacheOffset(slotValue, vTokenSize, tilingData_->vCacheBlockStride);
            if (NCT) {
                kOffsetIn = (startTaskId_ + i) * tilingData_->kStride + tilingData_->kOffset;
                vOffsetIn = (startTaskId_ + i) * tilingData_->vStride + tilingData_->vOffset;
            }
            CopyToCache<T1>(kOffsetIn, kOffsetOut, kUbLoop, kUbTail, keyInGm_, keyCacheOutGm_);
            CopyToCache<T2>(vOffsetIn, vOffsetOut, vUbLoop, vUbTail, valueInGm_, valueCacheOutGm_);
        }
    }

    // 小shape下的性能分支 ———— 能够一次性完全搬入key和value的一个token，且每个核处理的token数量较少
    __aicore__ inline void ProcessFullyLoad()
    {
        int32_t kAlign = BLOCK_SIZE / sizeof(T1);
        int32_t vAlign = BLOCK_SIZE / sizeof(T2);
        TBuf<TPosition::VECCALC> tokenBuf;
        pipe_->InitBuffer(tokenBuf, tilingData_->ubSize);
        TEventID eventIdMTE2ToMTE3_ = pipe_->FetchEventID(HardEvent::MTE2_MTE3);
        TEventID eventIdMTE3ToMTE2_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
        // Key
        int64_t kTokenSize = tilingData_->numHead * tilingData_->kHeadSize; // Key一个token大小
        int64_t kTokenSizeAlign = (kTokenSize + kAlign - 1) / kAlign * kAlign; // Key一个token大小对齐
        DataCopyExtParams kCopyParam = {1, static_cast<uint32_t>(kTokenSize * sizeof(T1)), 0, 0, 0};
        DataCopyPadExtParams<T1> kPadParams = {false, 0, 0, 0};
        auto kLocal = tokenBuf.Get<T1>(kTokenSizeAlign);
        // Value
        int64_t vTokenSize = tilingData_->numHead * tilingData_->vHeadSize; // Value一个token大小
        int64_t vTokenSizeAlign = (vTokenSize + vAlign - 1) / vAlign * vAlign; // Value一个token大小对齐
        DataCopyExtParams vCopyParam = {1, static_cast<uint32_t>(vTokenSize * sizeof(T2)), 0, 0, 0};
        auto vLocal = tokenBuf.GetWithOffset<T2>(vTokenSizeAlign, kTokenSizeAlign * sizeof(T1));
        DataCopyPadExtParams<T2> vPadParams = {false, 0, 0, 0};

        for (int64_t i = 0; i < curBlockFactor_; i++) {
            int64_t slotValue = slotMappingGm_.GetValue(startTaskId_ + i);
            if (slotValue < 0) {
                continue;
            }
            // key
            int64_t kOffsetIn = (startTaskId_ + i) * kTokenSize;
            int64_t kOffsetOut = static_cast<int64_t>(
                GetCacheOffset(slotValue, static_cast<uint64_t>(kTokenSize), tilingData_->kCacheBlockStride));
            // value
            int64_t vOffsetIn = (startTaskId_ + i) * vTokenSize;
            int64_t vOffsetOut = static_cast<int64_t>(
                GetCacheOffset(slotValue, static_cast<uint64_t>(vTokenSize), tilingData_->vCacheBlockStride));
            if (NCT) {
                kOffsetIn = (startTaskId_ + i) * tilingData_->kStride + tilingData_->kOffset;
                vOffsetIn = (startTaskId_ + i) * tilingData_->vStride + tilingData_->vOffset;
            }

            DataCopyPad(kLocal, keyInGm_[kOffsetIn], kCopyParam, kPadParams);
            DataCopyPad(vLocal, valueInGm_[vOffsetIn], vCopyParam, vPadParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3_);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3_);
            DataCopyPad(keyCacheOutGm_[kOffsetOut], kLocal, kCopyParam);
            DataCopyPad(valueCacheOutGm_[vOffsetOut], vLocal, vCopyParam);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2_);
        }
    }

private:
    __aicore__ inline uint64_t GetCacheOffset(int64_t slotValue, uint64_t tokenSize, int64_t cacheBlockStride) const
    {
        if (cacheBlockStride > 0) {
            int64_t blockIndex = slotValue / tilingData_->blockSize;
            int64_t blockOffset = slotValue % tilingData_->blockSize;
            return static_cast<uint64_t>(blockIndex) * static_cast<uint64_t>(cacheBlockStride) +
                   static_cast<uint64_t>(blockOffset) * tokenSize;
        }
        return static_cast<uint64_t>(slotValue) * tokenSize;
    }

    GlobalTensor<T1> keyInGm_;
    GlobalTensor<T1> keyCacheOutGm_;
    GlobalTensor<T2> valueInGm_;
    GlobalTensor<T2> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
};
} // namespace ScatterPaKvCacheVllm
#endif