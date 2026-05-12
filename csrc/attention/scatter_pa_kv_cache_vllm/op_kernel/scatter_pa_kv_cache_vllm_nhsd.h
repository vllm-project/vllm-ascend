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
 * \file scatter_pa_kv_cache_vllm_nhsd.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_NHSD_H
#define SCATTER_PA_KV_CACHE_VLLM_NHSD_H

#include "kernel_operator.h"

namespace ScatterPaKvCacheVllm {
using namespace AscendC;

template <typename T1, typename T2>
class ScatterPaKvCacheVllmNHSD {
public:
    __aicore__ inline ScatterPaKvCacheVllmNHSD(TPipe *pipe): pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR slotIndices, GM_ADDR keyCacheOut,
                                GM_ADDR valueCacheOut, const ScatterPaKvCacheVllmTilingData* tilingData) {
        this->blockIdx = AscendC::GetBlockIdx();
        ParseTilingData(tilingData);

        keyInOffset = this->blockIdx * (blockFactor * numHead * kHeadSize);
        valueInOffset = this->blockIdx * (blockFactor * numHead * vHeadSize);
        slotmappingOffset = this->blockIdx * blockFactor;

        this->perCoreKeySize = numHead * kHeadSize;
        this->perCoreValueSize = numHead * vHeadSize;
        keyInGm.SetGlobalBuffer((__gm__ T1*)key + keyInOffset);
        valueInGm.SetGlobalBuffer((__gm__ T1*)value + valueInOffset);
        keycacheOutGm.SetGlobalBuffer((__gm__ T1*)keyCacheOut);
        valuecacheOutGm.SetGlobalBuffer((__gm__ T1*)valueCacheOut);
        slotmappingGm.SetGlobalBuffer((__gm__ T2*)slotIndices + slotmappingOffset);
        pipe_->InitBuffer(keyInBuf, this->perCoreKeySize * sizeof(T1));
        pipe_->InitBuffer(valueInBuf, this->perCoreValueSize * sizeof(T1));

        this->keyIn = keyInBuf.Get<T1>();
        this->valueIn = valueInBuf.Get<T1>();
    }

    __aicore__ inline void Process()
    {
        int64_t perCoreDoKeyCount = numHead * kHeadSize;
        int64_t perCoreDoValueCount = numHead * vHeadSize;
        int64_t perCoreDoRowWork = 0;
        int64_t keyInPerCoreOffset = 0;
        int64_t valueInPerCoreOffset = 0;
        int64_t keycacheOutGmOffset = 0;
        int64_t valuecacheOutGmOffset = 0;
        int64_t blockIndex = 0;
        int64_t blockOffset = 0;
        uint32_t keyCacheStride = static_cast<uint32_t>((blockSize - 1) * kHeadSize);
        uint32_t valueCacheStride = static_cast<uint32_t>((blockSize - 1) * vHeadSize);
        if (this->blockIdx < useCoreNum - 1) {
            perCoreDoRowWork = blockFactor;
        } else {
            perCoreDoRowWork = tailBlockFactor;
        }
        for (int64_t n = 0; n < perCoreDoRowWork; n++) {
            keyInPerCoreOffset = n * perCoreDoKeyCount;
            valueInPerCoreOffset = n * perCoreDoValueCount;
            DataCopyIn<T1>(keyIn, keyInGm[keyInPerCoreOffset], perCoreDoKeyCount);
            DataCopyIn<T1>(valueIn, valueInGm[valueInPerCoreOffset], perCoreDoValueCount);
            PipeSync<AscendC::HardEvent::MTE2_MTE3>();
            int64_t slotmappingValue = static_cast<int64_t>(slotmappingGm.GetValue(n));
            blockIndex = slotmappingValue / blockSize;
            blockOffset = slotmappingValue % blockSize;
            keycacheOutGmOffset = blockIndex * numHead * blockSize * kHeadSize + blockOffset * kHeadSize;
            valuecacheOutGmOffset = blockIndex * numHead * blockSize * vHeadSize + blockOffset * vHeadSize;
            DataCopyOut<T1>(keycacheOutGm[keycacheOutGmOffset], keyIn,
                        static_cast<uint32_t>(kHeadSize), keyCacheStride, static_cast<uint16_t>(numHead));
            DataCopyOut<T1>(valuecacheOutGm[valuecacheOutGmOffset], valueIn,
                        static_cast<uint32_t>(vHeadSize), valueCacheStride, static_cast<uint16_t>(numHead));
            PipeSync<AscendC::HardEvent::MTE3_MTE2>();
        }
    }

private:
    __aicore__ inline void ParseTilingData(const ScatterPaKvCacheVllmTilingData* tilingData) {
        blockFactor = tilingData->blockFactor;
        tailBlockFactor = tilingData->tailBlockFactor;
        useCoreNum = tilingData->usedCoreNum;
        numTokens = tilingData->numTokens;
        numHead = tilingData->numHead;
        kHeadSize = tilingData->kHeadSize;
        vHeadSize = tilingData->vHeadSize;
        blockSize = tilingData->blockSize;
    }

    template <typename U>
    __aicore__ inline void DataCopyIn(
        const AscendC::LocalTensor<U>& dst, const AscendC::GlobalTensor<U>& src, uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<U> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename U>
    __aicore__ inline void DataCopyOut(
        const AscendC::GlobalTensor<U>& dst, const AscendC::LocalTensor<U>& src, uint32_t perHeadCount,
        uint32_t dstStride, uint16_t blockCount)
    {
        AscendC::DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(perHeadCount * sizeof(U)), 0, static_cast<uint32_t>(dstStride * sizeof(U)), 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    template <AscendC::HardEvent hardEvent>
    __aicore__ inline void PipeSync()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
        AscendC::SetFlag<hardEvent>(eventID);
        AscendC::WaitFlag<hardEvent>(eventID);
    }

private:
    TPipe *pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> keyInBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> valueInBuf;

    AscendC::GlobalTensor<T1> keyInGm;
    AscendC::GlobalTensor<T1> valueInGm;
    AscendC::GlobalTensor<T1> keycacheOutGm;
    AscendC::GlobalTensor<T1> valuecacheOutGm;
    AscendC::GlobalTensor<T2> slotmappingGm;

    AscendC::LocalTensor<T1> keyIn;
    AscendC::LocalTensor<T1> valueIn;

    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t numTokens = 0;
    int64_t numHead = 0;
    int64_t kHeadSize = 0;
    int64_t vHeadSize = 0;
    int64_t blockSize = 0;
    int32_t blockIdx = 0;
    int64_t useCoreNum = 0;
    int64_t keyInOffset = 0;
    int64_t valueInOffset = 0;
    int64_t slotmappingOffset = 0;
    int64_t perCoreKeySize = 0;
    int64_t perCoreValueSize = 0;

};
} // namespace ScatterPaKvCacheVllm
#endif