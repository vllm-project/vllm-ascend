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
 * \file scatter_pa_kv_cache_vllm_normal_siso.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_NORMAL_SISO_H
#define SCATTER_PA_KV_CACHE_VLLM_NORMAL_SISO_H

#include "scatter_pa_kv_cache_vllm_normal_common.h"

namespace ScatterPaKvCacheVllm {
using namespace AscendC;

template <typename T1, typename IndexDtype, bool NCT = false>
class ScatterPaKvCacheVllmNormalSiso : public ScatterPaKvCacheVllmNormalCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmNormalSiso(TPipe *pipe, const ScatterPaKvCacheVllmTilingData *__restrict__ tilingData)
    : ScatterPaKvCacheVllmNormalCommon(pipe, tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        int64_t coreNums = GetBlockNum(); // 用的核数
        int64_t blockId = GetBlockIdx(); //当前核
        curBlockFactor_ =
            (blockId == coreNums - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;
        startTaskId_ = blockId * tilingData_->blockFactor; //当前核起始token

        keyInGm_.SetGlobalBuffer((__gm__ T1 *)keyIn);
        keyCacheOutGm_.SetGlobalBuffer((__gm__ T1 *)keyCacheOut);
        slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)slotMapping);
    }

    // 通用分支
    __aicore__ inline void Process()
    {
        usedUbSize_ = tilingData_->ubSize / BUFFER_NUM;
        pipe_->InitBuffer(queBind_, BUFFER_NUM, usedUbSize_);
        // key
        uint64_t kTokenSize = static_cast<uint64_t>(tilingData_->numHead) * tilingData_->kHeadSize; // Key一个token大小
        int64_t kUbLoop = kTokenSize * sizeof(T1) / (usedUbSize_);
        int64_t kUbTail = kTokenSize * sizeof(T1) % (usedUbSize_);

        for (int64_t i = 0; i < curBlockFactor_; i++) {
            int64_t slotValue = slotMappingGm_.GetValue(startTaskId_ + i);
            if (slotValue < 0) {
                continue;
            }
            // key
            uint64_t kOffsetIn = (startTaskId_ + i) * kTokenSize;
            uint64_t kOffsetOut = slotValue * kTokenSize;
            if (NCT) {
                kOffsetIn = (startTaskId_ + i) * tilingData_->kStride + tilingData_->kOffset;
            }
            CopyToCache<T1>(kOffsetIn, kOffsetOut, kUbLoop, kUbTail, keyInGm_, keyCacheOutGm_);
        }
    }

    // 小shape下的性能分支 ———— 能够一次性完全搬入key和value的一个token，且每个核处理的token数量较少
    __aicore__ inline void ProcessFullyLoad()
    {
        int32_t kAlign = BLOCK_SIZE / sizeof(T1);
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

        for (int64_t i = 0; i < curBlockFactor_; i++) {
            int64_t slotValue = slotMappingGm_.GetValue(startTaskId_ + i);
            if (slotValue < 0) {
                continue;
            }
            int64_t kOffsetIn = (startTaskId_ + i) * kTokenSize;
            int64_t kOffsetOut = slotValue * kTokenSize;
            if (NCT) {
                kOffsetIn = (startTaskId_ + i) * tilingData_->kStride + tilingData_->kOffset;
            }

            DataCopyPad(kLocal, keyInGm_[kOffsetIn], kCopyParam, kPadParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3_);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3_);
            DataCopyPad(keyCacheOutGm_[kOffsetOut], kLocal, kCopyParam);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2_);
        }
    }

private:
    GlobalTensor<T1> keyInGm_;
    GlobalTensor<T1> keyCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;
};
} // namespace ScatterPaKvCacheVllm
#endif