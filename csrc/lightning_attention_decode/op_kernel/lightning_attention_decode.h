/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef LIGHTNING_ATTENTION_DECODE_H
#define LIGHTNING_ATTENTION_DECODE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

namespace LightningAttention {

template <typename T>
class LightningAttentionDecode {
public:
    __aicore__ inline LightningAttentionDecode() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR slope_rate,
                                GM_ADDR kv_history, GM_ADDR slot_ids, GM_ADDR attention_out, GM_ADDR kv_cache_out,
                                GM_ADDR workspace, const LightningAttentionDecodeTilingData *__restrict tiling,
                                AscendC::TPipe *pipe);
    __aicore__ inline void Process();

public:
    // define matmul object for matmul(Q, KV)
    using a1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using b1Type = MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, T, false>;
    using c1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using bias1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    Matmul<a1Type, b1Type, c1Type, bias1Type> mm1;

private:
    __aicore__ inline void GenerateDecay();
    __aicore__ inline void ComputeAttention(uint32_t offset);
    __aicore__ inline void UpdateKVCache(uint32_t kvCacheOffset, uint32_t offset, uint32_t headIdx);
    __aicore__ inline void SaveKVCache(uint32_t kvCacheOffset);
    __aicore__ inline void WaitKVCacheSaved();

private:
    AscendC::GlobalTensor<T> queryGM_;
    AscendC::GlobalTensor<T> keyGM_;
    AscendC::GlobalTensor<T> valueGM_;
    AscendC::GlobalTensor<T> slopeRateGM_;
    AscendC::GlobalTensor<int32_t> slotIdsGM_;
    AscendC::GlobalTensor<T> attentionOutGM_;
    AscendC::GlobalTensor<T> kvCacheHistoryGM_;
    AscendC::GlobalTensor<T> kvCacheOutGM_;

    uint32_t currentCoreId_;
    const LightningAttentionDecodeTilingData *__restrict tiling_;
    uint32_t batchSize_;
    uint32_t kvCacheBatchSize_;
    uint32_t headNum_;
    uint32_t headNumPad_;
    uint32_t headDim_;
    uint32_t actualUsedAivNum_;
    uint32_t taskNum_;
    uint32_t eleCountPerKVCache_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> kvCacheBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qInQue_;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> attentionOutQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kvFp32Buf_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> kvCacheFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> attentionFp32Buf_;
};

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR slope_rate, GM_ADDR kv_history, GM_ADDR slot_ids,
        GM_ADDR attention_out, GM_ADDR kv_cache_out, GM_ADDR workspace,
        const LightningAttentionDecodeTilingData *__restrict tiling, AscendC::TPipe *pipe)
{
    currentCoreId_ = GetBlockIdx();
    tiling_ = tiling;
    batchSize_ = tiling->laBaseParams.batchSize;
    kvCacheBatchSize_ = tiling->laBaseParams.kvCacheBatchSize;
    headNum_ = tiling->laBaseParams.headNum;
    headNumPad_ = headNum_ < 16 ? 16 : headNum_;
    headDim_ = tiling->laBaseParams.headDim;
    actualUsedAivNum_ = tiling->laBaseParams.actualUsedAivNum;
    taskNum_ = tiling->laBaseParams.taskNum;
    eleCountPerKVCache_ = headDim_ * headDim_;

    queryGM_.SetGlobalBuffer((__gm__ T*)query);
    keyGM_.SetGlobalBuffer((__gm__ T*)key);
    valueGM_.SetGlobalBuffer((__gm__ T*)value);
    slopeRateGM_.SetGlobalBuffer((__gm__ T*)slope_rate);
    slotIdsGM_.SetGlobalBuffer((__gm__ int32_t*)slot_ids);
    attentionOutGM_.SetGlobalBuffer((__gm__ T*)attention_out);

    kvCacheHistoryGM_.SetGlobalBuffer((__gm__ T*)kv_history);
    kvCacheOutGM_.SetGlobalBuffer((__gm__ T*)kv_cache_out);

    auto maxBufSize = 128 * 128;

    pipe->InitBuffer(kvCacheBuf_, sizeof(T) * maxBufSize);           // 32k for half, 64k for fp32
    pipe->InitBuffer(decayBuf_, sizeof(T) * headNumPad_);        // maximum headNum is 64, 0.125k
    pipe->InitBuffer(kBuf_, sizeof(T) * headDim_);                    // 0.25k
    pipe->InitBuffer(vBuf_, sizeof(T) * headDim_);                    // 0.25k
    pipe->InitBuffer(qInQue_, 1, sizeof(T) * headDim_);               // 0.25k
    pipe->InitBuffer(kvFp32Buf_, sizeof(float) * maxBufSize);         // 64k
    pipe->InitBuffer(broadCastBuf_, 4096);                            // reserved for broadcast, 4k
    pipe->InitBuffer(attentionOutQue_, 1, sizeof(T) * headDim_);      // 0.25k

    if constexpr (!IsSameType<T, float>::value) {
        pipe->InitBuffer(kvCacheFp32Buf_, sizeof(float) * maxBufSize);    // 64k
        pipe->InitBuffer(decayFp32Buf_, sizeof(float) * headNumPad_);     // 0.25k
        pipe->InitBuffer(qFp32Buf_, sizeof(float) * headDim_);            // 0.5k
        pipe->InitBuffer(kFp32Buf_, sizeof(float) * headDim_);            // 0.5k
        pipe->InitBuffer(vFp32Buf_, sizeof(float) * headDim_);            // 0.5k
        pipe->InitBuffer(attentionFp32Buf_, sizeof(float) * headDim_);    // 0.5k
    }
}

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::Process()
{
    uint16_t absoluteHeadIdx = currentCoreId_;
    uint32_t offset = absoluteHeadIdx * headDim_;
    uint32_t offsetStep = actualUsedAivNum_ * headDim_;
    uint32_t kvCacheOffsetPerBatch = eleCountPerKVCache_ * headNum_;

    GenerateDecay();

    bool isFirstLoop = true;
    for (uint32_t relativeHeadIdx, batchId, kvCacheSlotId, kvCacheOffset; absoluteHeadIdx < taskNum_;
         absoluteHeadIdx += actualUsedAivNum_, offset += offsetStep) {
        batchId = absoluteHeadIdx / headNum_;
        kvCacheSlotId = slotIdsGM_.GetValue(batchId);
        if (kvCacheSlotId < 0 || kvCacheSlotId >= kvCacheBatchSize_) {
            continue;
        }
        relativeHeadIdx = absoluteHeadIdx % headNum_;
        kvCacheOffset = kvCacheSlotId * kvCacheOffsetPerBatch + relativeHeadIdx * eleCountPerKVCache_;
        if (isFirstLoop) {
            isFirstLoop = false;
        } else {
            WaitKVCacheSaved();
        }
        UpdateKVCache(kvCacheOffset, offset, relativeHeadIdx);
        ComputeAttention(offset);
        SaveKVCache(kvCacheOffset);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::GenerateDecay()
{
    auto decayTTensor = decayBuf_.Get<T>();
    // Copy in
    AscendC::DataCopy(decayTTensor, slopeRateGM_, headNumPad_);
    int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    // Compute
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Muls(decayTTensor, decayTTensor, (float)-1.0, headNumPad_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(decayTTensor, decayTTensor, headNumPad_);
    } else {
        auto decayFp32Tensor = decayFp32Buf_.Get<float>();
        AscendC::Cast(decayFp32Tensor, decayTTensor, RoundMode::CAST_NONE, headNumPad_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(decayFp32Tensor, decayFp32Tensor, (float)-1.0, headNumPad_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(decayFp32Tensor, decayFp32Tensor, headNumPad_);
    }
}


template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::ComputeAttention(uint32_t offset)
{
    // calculate O = matmul(Q, KV)
    if constexpr (IsSameType<T, float>::value) {
        auto kvCacheTensor = kvCacheBuf_.Get<T>();
        mm1.SetTensorA(queryGM_[offset]);
        mm1.SetTensorB(kvCacheTensor);
        mm1.template IterateAll<true>(attentionOutGM_[offset]);
        mm1.End();
    } else {
        auto qTensor = qInQue_.AllocTensor<T>();
        auto qBroadCastTensor = kvFp32Buf_.Get<float>();
        auto broadCastTensor = broadCastBuf_.Get<uint8_t>();
        const uint32_t dstShape[2] = {headDim_, headDim_};
        const uint32_t srcShape[2] = {headDim_, 1};
        uint32_t eleCount = 64;

        AscendC::DataCopy(qTensor, queryGM_[offset], headDim_);
        qInQue_.EnQue<T>(qTensor);

        qTensor = qInQue_.DeQue<T>();
        auto kvCacheFp32Tensor = kvCacheFp32Buf_.Get<float>();
        auto qFp32Tensor = qFp32Buf_.Get<float>();
        auto attentionFp32Tensor = attentionFp32Buf_.Get<float>();
        AscendC::Cast(qFp32Tensor, qTensor, RoundMode::CAST_NONE, headDim_);
        qInQue_.FreeTensor(qTensor);
        AscendC::BroadCast<float, 2, 1>(qBroadCastTensor, qFp32Tensor, dstShape, srcShape, broadCastTensor);
        AscendC::Duplicate(attentionFp32Tensor, 0.0f, headDim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::MulAddDst(attentionFp32Tensor, qBroadCastTensor, kvCacheFp32Tensor, eleCount, headDim_,
                           {1, 1, 1, 0, 16, 16});
        AscendC::MulAddDst(attentionFp32Tensor[eleCount], qBroadCastTensor[eleCount], kvCacheFp32Tensor[eleCount],
                           eleCount, headDim_, {1, 1, 1, 0, 16, 16});
        AscendC::PipeBarrier<PIPE_V>();

        auto attentionTensor = attentionOutQue_.AllocTensor<T>();
        AscendC::Cast(attentionTensor, attentionFp32Tensor, RoundMode::CAST_ROUND, headDim_);
        attentionOutQue_.EnQue<T>(attentionTensor);

        attentionTensor = attentionOutQue_.DeQue<T>();
        AscendC::DataCopy(attentionOutGM_[offset], attentionTensor, headDim_);
        attentionOutQue_.FreeTensor<T>(attentionTensor);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::UpdateKVCache(uint32_t kvCacheOffset, uint32_t offset,
                                                                  uint32_t headIndex)
{
    uint64_t mask = 64;
    auto kvCacheTensor = kvCacheBuf_.Get<T>();
    auto kTensor = kBuf_.Get<T>();
    auto vTensor = vBuf_.Get<T>();
    auto kvFp32Tensor = kvFp32Buf_.Get<float>();
    auto broadCastTensor = broadCastBuf_.Get<uint8_t>();
    const uint32_t dstShape[2] = {headDim_, headDim_};
    const uint32_t srcShape[2] = {headDim_, 1};
    AscendC::DataCopy(kvCacheTensor, kvCacheHistoryGM_[kvCacheOffset], eleCountPerKVCache_);
    AscendC::DataCopy(kTensor, keyGM_[offset], headDim_);
    AscendC::DataCopy(vTensor, valueGM_[offset], headDim_);

    int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);

    if constexpr (IsSameType<T, float>::value) {
        float decayLambda = decayBuf_.Get<T>().GetValue(headIndex);
        // multiply with decay
        AscendC::Muls(kvCacheTensor, kvCacheTensor, decayLambda, eleCountPerKVCache_);
        // KV_cur = Ki * Vi
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BroadCast<float, 2, 1>(kvFp32Tensor, kTensor, dstShape, srcShape, broadCastTensor);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(kvFp32Tensor, kvFp32Tensor, vTensor, mask, 128, {1, 1, 1, 16, 16, 0});
        AscendC::Mul(kvFp32Tensor[64], kvFp32Tensor[64], vTensor[64], mask, 128, {1, 1, 1, 16, 16, 0});
        AscendC::PipeBarrier<PIPE_V>();
        // KV_cache = KV_cur + KV_cache * kv_decay
        AscendC::Add(kvCacheTensor, kvCacheTensor, kvFp32Tensor, eleCountPerKVCache_);
    } else {
        float decayLambda = decayFp32Buf_.Get<float>().GetValue(headIndex);
        auto kvCacheFp32Tensor = kvCacheFp32Buf_.Get<float>();
        auto kFp32Tensor = kFp32Buf_.Get<float>();
        auto vFp32Tensor = vFp32Buf_.Get<float>();
        // cast kvCache to fp32 and multiply with decay
        AscendC::Cast(kvCacheFp32Tensor, kvCacheTensor, RoundMode::CAST_NONE, eleCountPerKVCache_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(kvCacheFp32Tensor, kvCacheFp32Tensor, decayLambda, eleCountPerKVCache_);

        // KV_cur = Ki * Vi
        AscendC::Cast(kFp32Tensor, kTensor, RoundMode::CAST_NONE, headDim_);
        AscendC::Cast(vFp32Tensor, vTensor, RoundMode::CAST_NONE, headDim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BroadCast<float, 2, 1>(kvFp32Tensor, kFp32Tensor, dstShape, srcShape, broadCastTensor);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(kvFp32Tensor, kvFp32Tensor, vFp32Tensor, mask, 128, {1, 1, 1, 16, 16, 0});
        AscendC::Mul(kvFp32Tensor[64], kvFp32Tensor[64], vFp32Tensor[64], mask, 128, {1, 1, 1, 16, 16, 0});
        AscendC::PipeBarrier<PIPE_V>();
        // KV_cache = KV_cur + KV_cache * kv_decay
        AscendC::Add(kvCacheFp32Tensor, kvCacheFp32Tensor, kvFp32Tensor, eleCountPerKVCache_);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::SaveKVCache(uint32_t kvCacheOffset)
{
    auto kvCacheTensor = kvCacheBuf_.Get<T>();
    if constexpr (!IsSameType<T, float>::value) {
        auto kvCacheFp32Tensor = kvCacheFp32Buf_.Get<float>();
        AscendC::Cast(kvCacheTensor, kvCacheFp32Tensor, RoundMode::CAST_ROUND, eleCountPerKVCache_);
        int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
    }
    AscendC::DataCopy(kvCacheOutGM_[kvCacheOffset], kvCacheTensor, eleCountPerKVCache_);
}

template <typename T>
__aicore__ inline void LightningAttentionDecode<T>::WaitKVCacheSaved()
{
    int32_t eventIdMte3ToMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

} // namespace LightningAttention

#endif // LIGHTNING_ATTENTION_DECODE_H
