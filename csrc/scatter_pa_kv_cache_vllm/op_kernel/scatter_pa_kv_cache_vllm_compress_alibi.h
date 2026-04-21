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
 * \file scatter_pa_kv_cache_vllm_compress_alibi.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_COMPRESS_ALIBI_H
#define SCATTER_PA_KV_CACHE_VLLM_COMPRESS_ALIBI_H

#include "scatter_pa_kv_cache_vllm_compress_common.h"
namespace ScatterPaKvCacheVllm {
template <typename T>
class ScatterPaKvCacheVllmCompressAlibi : public ScatterPaKvCacheVllmCompressCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmCompressAlibi()
    {
    }

    __aicore__ inline void Process(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
                                   GM_ADDR slotMapping, GM_ADDR winsIn, GM_ADDR seqLenIn, GM_ADDR keyCacheOut,
                                   GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<int32_t>(winsInputGt_, winsIn);
        InitGlobalTensor<int32_t>(seqLenInputGt_, seqLenIn);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        InitTBuf<int32_t>(winsBuf_, numHeads_ * numBatchs_);
        winsLocal_ = winsBuf_.Get<int32_t>();

        InitTBuf<int32_t>(seqLenBuf_, numBatchs_);
        seqLenLocal_ = seqLenBuf_.Get<int32_t>();

        tokenSize_ = 1 * headSizeK_;
        numBlocks_ = static_cast<uint32_t>(tokenSize_) * sizeof(T) / BLOCK_SIZE;

        uint32_t winSize = RoundUp(numHeads_ * numBatchs_ * sizeof(int32_t), BLOCK_SIZE);
        uint32_t seqSize = RoundUp(numHeads_ * sizeof(int32_t), BLOCK_SIZE);
        uint32_t ubSize = MAX_UB_SIZE - winSize - seqSize;
        InitTBuf<T>(tokenBuf_, ubSize / sizeof(T));
        tokenLocal_ = tokenBuf_.Get<T>();

        DataCopy(winsLocal_, winsInputGt_, RoundUp(numHeads_ * numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(seqLenLocal_, seqLenInputGt_, RoundUp(numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int i = 1; i < numBatchs_; i++) { // 获取累加的seqlen
            seqLenLocal_.SetValue(i, seqLenLocal_.GetValue(i) + seqLenLocal_.GetValue(i - 1));
        }

        AllocateTask();
        AscendC::DataCopyParams copyParamsIn = {0, 0, 0, 0};
        AscendC::DataCopyParams copyParamsOut = {0, 0, 0, 0};
        uint64_t start;
        uint64_t cacheStart;
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            auto batchId = (i + startTaskId_) / numHeads_;
            auto headId = (i + startTaskId_) % numHeads_;
            uint32_t headWin = static_cast<uint32_t>(winsLocal_.GetValue(i + startTaskId_));
            int32_t consumSeqLen = seqLenLocal_.GetValue(batchId);
            auto offsetPerLine = (numHeads_ - 1) * numBlocks_;

            // 至多需要这么多次 去分块搬运一个head
            uint32_t totalCopyLoop = ((headWin * tokenSize_ * sizeof(T)) / ubSize) + 1;
            uint32_t tokensPerLoop = headWin / totalCopyLoop;
            uint32_t tailTokens = headWin - totalCopyLoop * tokensPerLoop;
            for (int j = 0; j < totalCopyLoop; j++) {
                copyParamsIn = {static_cast<uint16_t>(tokensPerLoop), static_cast<uint16_t>(numBlocks_),
                                static_cast<uint16_t>(offsetPerLine), 0};
                copyParamsOut = {static_cast<uint16_t>(tokensPerLoop), static_cast<uint16_t>(numBlocks_), 0, 0};
                start = tokenSize_ * (numHeads_ * (consumSeqLen - headWin + tokensPerLoop * j) + headId);
                cacheStart = tokenSize_ * (slotInputGt_.GetValue(i + startTaskId_) + tokensPerLoop * j);
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
            }
            if (tailTokens != 0) {
                copyParamsIn = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_),
                                static_cast<uint16_t>(offsetPerLine), 0};
                copyParamsOut = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_), 0, 0};
                start = tokenSize_ * (numHeads_ * (consumSeqLen - tailTokens) + headId);
                cacheStart = tokenSize_ * (slotInputGt_.GetValue(i + startTaskId_) + tokensPerLoop * totalCopyLoop);
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParamsIn, copyParamsOut);
            }
        }
    }

private:
    uint64_t tokenSize_ = 0;
    uint32_t numBlocks_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;

    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::GlobalTensor<int32_t> winsInputGt_;
    AscendC::GlobalTensor<int32_t> seqLenInputGt_;

    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> winsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> seqLenBuf_;
};
} // namespace ScatterPaKvCacheVllm
#endif