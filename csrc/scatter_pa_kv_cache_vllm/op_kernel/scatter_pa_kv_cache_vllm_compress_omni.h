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
 * \file scatter_pa_kv_cache_vllm_compress_omni.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_COMPRESS_OMNI_H
#define SCATTER_PA_KV_CACHE_VLLM_COMPRESS_OMNI_H

#include "scatter_pa_kv_cache_vllm_compress_common.h"

namespace ScatterPaKvCacheVllm {
template <typename T>
class ScatterPaKvCacheVllmCompressOmni : public ScatterPaKvCacheVllmCompressCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmCompressOmni()
    {
    }

    template <bool PHASE>
    __aicore__ inline void Copy2Cache(AscendC::GlobalTensor<T> &inputCache, AscendC::GlobalTensor<T> &outputCache,
                                      uint32_t srcStart, uint32_t dstStart)
    {
        event_t eventID = EVENT_ID0;
        uint32_t totalCopyLoop = 0; // 需要搬多少次
        uint32_t tokensPerLoop = 0; // 每次搬几个token
        uint32_t tailTokens = 0;    // 最后一轮需要搬多少个token
        if constexpr (PHASE) {
            totalCopyLoop = ((offsetIdx * tokenSize_) / copyFp16Num);
            tokensPerLoop = totalCopyLoop == 0 ? 0 : copyFp16Num / tokenSize_;
            tailTokens = offsetIdx - totalCopyLoop * tokensPerLoop;
        } else {
            uint32_t remainingLength = seqLen - offsetIdx - headWin;
            totalCopyLoop = ((remainingLength * tokenSize_) / copyFp16Num);
            tokensPerLoop = totalCopyLoop == 0 ? 0 : copyFp16Num / tokenSize_;
            tailTokens = remainingLength - totalCopyLoop * tokensPerLoop;
        }
        AscendC::DataCopyParams copyParamsIn = {static_cast<uint16_t>(tokensPerLoop), static_cast<uint16_t>(numBlocks_),
                                                static_cast<uint16_t>(offsetPerLine), 0};
        AscendC::DataCopyParams copyParamsOut = {static_cast<uint16_t>(tokensPerLoop),
                                                 static_cast<uint16_t>(numBlocks_), 0, 0};
        if (totalCopyLoop != 0) {
            CopyKvCacheDoubleBuf(inputCache, copyLocalPing_, outputCache, eventID, srcStart, dstStart, copyParamsIn,
                                 copyParamsOut);
        }
        if constexpr (PHASE) { // partA
            srcStart = tokenSize_ * (numHeads_ * consumSeqLen + headId);
            dstStart = tokenSize_ * slotValue;
        } else { // partC
            srcStart = tokenSize_ * (numHeads_ * (consumSeqLen + offsetIdx + headWin) + headId);
            dstStart = tokenSize_ * (slotValue + offsetIdx); // note: potential modification, remove + 1
        }
        for (uint32_t j = 1; j < totalCopyLoop; j++) {
            srcStart += tokenSize_ * numHeads_ * tokensPerLoop;
            dstStart += tokenSize_ * tokensPerLoop;
            if (j % BUFFER_NUM == 0) {
                copyLocal_ = copyLocalPing_;
                eventID = EVENT_ID0;
            } else {
                copyLocal_ = copyLocalPong_;
                eventID = EVENT_ID1;
            }
            CopyKvCacheDoubleBuf(inputCache, copyLocal_, outputCache, eventID, srcStart, dstStart, copyParamsIn,
                                 copyParamsOut);
        }
        if (tailTokens != 0) {
            copyParamsIn = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_),
                            static_cast<uint16_t>(offsetPerLine), 0};
            copyParamsOut = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(numBlocks_), 0, 0};
            if constexpr (PHASE) {
                srcStart = tokenSize_ * (numHeads_ * (consumSeqLen + offsetIdx - tailTokens) + headId);
                dstStart = tokenSize_ * (slotValue + tokensPerLoop * totalCopyLoop);
            } else {
                srcStart = tokenSize_ * (numHeads_ * (consumSeqLen + seqLen - tailTokens) + headId);
                dstStart = tokenSize_ * (slotValue + offsetIdx + tokensPerLoop * totalCopyLoop);
            }
            if (totalCopyLoop % BUFFER_NUM == 0) {
                copyLocal_ = copyLocalPing_;
                eventID = EVENT_ID0;
            } else {
                copyLocal_ = copyLocalPong_;
                eventID = EVENT_ID1;
            }
            CopyKvCacheDoubleBuf(inputCache, copyLocal_, outputCache, eventID, srcStart, dstStart, copyParamsIn,
                                 copyParamsOut);
        }
    }

    __aicore__ inline void Compress4Rope(AscendC::GlobalTensor<T> &inputCache, AscendC::GlobalTensor<T> &outputCache,
                                         uint32_t startTaskOffset)
    {
        ParamSet(slotInputGt_, startTaskOffset);
        // ------------ PART B ------------ 舍弃wins个token
        headWin = offsetIdx == seqLen ? 0 : headWin;
        // note: comment ComputeAvg for duo-attention

        // ------------ PART A ------------ 将舍弃前的tokens 直接搬出
        uint64_t srcStart = tokenSize_ * (numHeads_ * consumSeqLen + headId);
        uint64_t dstStart = tokenSize_ * slotValue;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        Copy2Cache<true>(inputCache, outputCache, srcStart, dstStart);

        // // ------------ PART C ------------ 将舍弃后剩余的tokens 直接搬出
        if (headWin != 0 && seqLen - offsetIdx - headWin != 0) {
            // 若不舍弃 partA将全部搬出 或A、B已处理完，则不需要partC
            srcStart = tokenSize_ * (numHeads_ * (consumSeqLen + offsetIdx + headWin) + headId);
            dstStart = tokenSize_ * (slotValue + offsetIdx);
            Copy2Cache<false>(inputCache, outputCache, srcStart, dstStart);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void InitTensor(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR slotMapping, GM_ADDR winsIn,
                                      GM_ADDR seqLenIn, GM_ADDR offsetIdx, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<int32_t>(winsInputGt_, winsIn);
        InitGlobalTensor<int32_t>(seqLenInputGt_, seqLenIn);
        InitGlobalTensor<int32_t>(offsetInputGt_, offsetIdx);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        RopeInitscalarBuf<T>();
        OmniInitBuf<T>(copyLocalPing_, copyLocalPong_);
    }

    __aicore__ inline void Method(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
                                  GM_ADDR slotMapping, GM_ADDR winsIn, GM_ADDR seqLenIn, GM_ADDR offsetIdx,
                                  GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitTensor(keyIn, valueIn, slotMapping, winsIn, seqLenIn, offsetIdx, keyCacheOut, valueCacheOut);
        AllocateTaskRope();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(tokenSize_);
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            PreProcess(winsInputGt_, seqLenInputGt_, offsetInputGt_);
            if (i + startTaskId_ < numTokens_) {
                uint32_t startTaskOffset = i + startTaskId_;
                Compress4Rope(keyInputGt_, keyOutputGt_, startTaskOffset);
            } else { // note: tail
                uint32_t startTaskOffset = i + startTaskId_ - numTokens_;
                Compress4Rope(valueInputGt_, valueOutputGt_, startTaskOffset);
            }
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

private:
    uint16_t repeatDivisor = 0;
    uint16_t addNumPerLoop = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::LocalTensor<T> copyLocal_;
    AscendC::LocalTensor<T> copyLocalPing_;
    AscendC::LocalTensor<T> copyLocalPong_;

    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::GlobalTensor<int32_t> winsInputGt_;
    AscendC::GlobalTensor<int32_t> seqLenInputGt_;
    AscendC::GlobalTensor<int32_t> offsetInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;
};
} // namespace ScatterPaKvCacheVllm
#endif