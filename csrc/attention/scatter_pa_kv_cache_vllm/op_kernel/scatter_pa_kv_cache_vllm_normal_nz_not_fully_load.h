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
 * \file scatter_pa_kv_cache_vllm_normal_nz_not_fully_load.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_NORMAL_NZ_NOT_FULLY_LOAD_H_
#define SCATTER_PA_KV_CACHE_VLLM_NORMAL_NZ_NOT_FULLY_LOAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_pa_kv_cache_vllm_common.h"

namespace ScatterPaKvCacheVllm {
using namespace AscendC;

template <typename T1, typename T2, typename IndexDtype>
class ScatterPaKvCacheVllmNormalNzNotFullyLoad {
public:
    __aicore__ inline ScatterPaKvCacheVllmNormalNzNotFullyLoad(TPipe *pipe,
                                                           const ScatterPaKvCacheVllmTilingData *__restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyToCacheNzK(uint32_t start, uint32_t cacheStart);
    __aicore__ inline void CopyToCacheNzV(uint32_t start, uint32_t cacheStart);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inputQueue_;
    const ScatterPaKvCacheVllmTilingData *tilingData_;

    GlobalTensor<T1> keyGm_;
    GlobalTensor<T2> valueGm_;
    GlobalTensor<T1> keyCacheOutGm_;
    GlobalTensor<T2> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    uint32_t blockSize_{0};
    uint64_t tokenSizeK_{0};
    uint64_t tokenSizeV_{0};
    int32_t lastDimK_{0};
    int32_t lastDimV_{0};
    uint32_t maxUbUsed_{0};
    uint32_t loopHK_{0};
    uint32_t tailHK_{0};
    uint32_t loopHV_{0};
    uint32_t tailHV_{0};
    uint64_t ubSizeK_{0};
    uint64_t ubSizeV_{0};
};

template <typename T1, typename T2, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheVllmNormalNzNotFullyLoad<T1, T2, IndexDtype>::Init(
    GM_ADDR key, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockSize_ = tilingData_->blockSize;
    tokenSizeK_ = tilingData_->numHead * tilingData_->kHeadSize; // H
    tokenSizeV_ = tilingData_->numHead * tilingData_->vHeadSize;

    lastDimK_ = BLOCK_SIZE / sizeof(T1);
    lastDimV_ = BLOCK_SIZE / sizeof(T2);

    keyGm_.SetGlobalBuffer((__gm__ T1 *)(key));
    valueGm_.SetGlobalBuffer((__gm__ T2 *)(value));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping));
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T1 *)(key_cache_out));
    valueCacheOutGm_.SetGlobalBuffer((__gm__ T2 *)(value_cache_out));

    maxUbUsed_ = tilingData_->ubSize / BUFFER_NUM;
    loopHK_ = (tokenSizeK_ * sizeof(T1)) / maxUbUsed_;
    tailHK_ = (tokenSizeK_ * sizeof(T1)) % maxUbUsed_;
    loopHV_ = (tokenSizeV_ * sizeof(T2)) / maxUbUsed_;
    tailHV_ = (tokenSizeV_ * sizeof(T2)) % maxUbUsed_;
    ubSizeK_ = loopHK_ == 0 ? tokenSizeK_ * sizeof(T1) : maxUbUsed_;
    ubSizeV_ = loopHV_ == 0 ? tokenSizeV_ * sizeof(T2) : maxUbUsed_;
    pipe_->InitBuffer(inputQueue_, BUFFER_NUM, maxUbUsed_);
}

template <typename T1, typename T2, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheVllmNormalNzNotFullyLoad<T1, T2, IndexDtype>::CopyToCacheNzK(uint32_t start,
                                                                                                uint32_t cacheStart)
{
    AscendC::DataCopyParams copyParamsIn{1, static_cast<uint16_t>(ubSizeK_ / BLOCK_SIZE), 0, 0};
    AscendC::DataCopyParams copyParamsOut{static_cast<uint16_t>(ubSizeK_ / BLOCK_SIZE), 1, 0,
                                          static_cast<uint16_t>(blockSize_ - 1)};
    for (uint32_t i = 0; i < loopHK_; i++) {
        // CopyIn
        LocalTensor<T1> input = inputQueue_.AllocTensor<T1>();
        DataCopy(input, keyGm_[start], copyParamsIn);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T1>();
        DataCopy(keyCacheOutGm_[cacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
        start += (ubSizeK_ / sizeof(T1));
        cacheStart += static_cast<uint64_t>((ubSizeK_ / sizeof(T1)) * blockSize_);
    }
    if (tailHK_ > 0) {
        // CopyIn
        copyParamsIn.blockLen = tailHK_ / BLOCK_SIZE;
        copyParamsOut.blockCount = tailHK_ / BLOCK_SIZE;
        LocalTensor<T1> input = inputQueue_.AllocTensor<T1>();
        DataCopy(input, keyGm_[start], copyParamsIn);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T1>();
        DataCopy(keyCacheOutGm_[cacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
    }
}

template <typename T1, typename T2, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheVllmNormalNzNotFullyLoad<T1, T2, IndexDtype>::CopyToCacheNzV(uint32_t start,
                                                                                                uint32_t cacheStart)
{
    AscendC::DataCopyParams copyParamsIn{1, static_cast<uint16_t>(ubSizeV_ / BLOCK_SIZE), 0, 0};
    AscendC::DataCopyParams copyParamsOut{static_cast<uint16_t>(ubSizeV_ / BLOCK_SIZE), 1, 0,
                                          static_cast<uint16_t>(blockSize_ - 1)};
    for (uint32_t i = 0; i < loopHV_; i++) {
        // CopyIn
        LocalTensor<T2> input = inputQueue_.AllocTensor<T2>();
        DataCopy(input, valueGm_[start], copyParamsIn);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T2>();
        DataCopy(valueCacheOutGm_[cacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
        start += (ubSizeV_ / sizeof(T2));
        cacheStart += static_cast<uint64_t>((ubSizeV_ / sizeof(T2)) * blockSize_);
    }
    if (tailHV_ > 0) {
        // CopyIn
        copyParamsIn.blockLen = tailHV_ / BLOCK_SIZE;
        copyParamsOut.blockCount = tailHV_ / BLOCK_SIZE;
        LocalTensor<T2> input = inputQueue_.AllocTensor<T2>();
        DataCopy(input, valueGm_[start], copyParamsIn);
        inputQueue_.EnQue(input);
        // CopyOut
        input = inputQueue_.DeQue<T2>();
        DataCopy(valueCacheOutGm_[cacheStart], input, copyParamsOut);
        inputQueue_.FreeTensor(input);
    }
}

template <typename T1, typename T2, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheVllmNormalNzNotFullyLoad<T1, T2, IndexDtype>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    uint32_t startTaskId = blockIdx_ * tilingData_->blockFactor;
    int32_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    for (int32_t loopToken = 0; loopToken < curBlockFactor; loopToken++) {
        // The offset of the token's initial address, with a unit of 32B
        int64_t slotValue = slotMappingGm_.GetValue(loopToken + startTaskId);
        if (slotValue < 0) {
            continue;
        }
        uint64_t startK = (loopToken + startTaskId) * tokenSizeK_;
        uint64_t startV = (loopToken + startTaskId) * tokenSizeV_;
        uint64_t blocksIdx = static_cast<uint64_t>(slotValue) / blockSize_;
        uint64_t blocksOffset = static_cast<uint64_t>(slotValue) - blockSize_ * blocksIdx;
        uint64_t cacheStartK = blocksIdx * blockSize_ * tokenSizeK_ + blocksOffset * lastDimK_;
        uint64_t cacheStartV = blocksIdx * blockSize_ * tokenSizeV_ + blocksOffset * lastDimV_;

        CopyToCacheNzK(startK, cacheStartK);
        CopyToCacheNzV(startV, cacheStartV);
    }
}
} // namespace ScatterPaKvCacheVllm

#endif // SCATTER_PA_KV_CACHE_NORMAL_NZ_NOT_FULLY_LOAD_H_