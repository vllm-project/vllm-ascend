/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_ATTENTION_PREFILL_H
#define LIGHTNING_ATTENTION_PREFILL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

namespace LightningAttention {

template <typename T>
class LightningAttentionPrefill {
public:
    __aicore__ inline LightningAttentionPrefill()
    {
    }
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR slope_rate, GM_ADDR kv_history,
                                GM_ADDR attention_out, GM_ADDR kv_caches, GM_ADDR workspace,
                                const LightningAttentionPrefillTilingData *__restrict tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

public:
    // define matmul object for matmul(Q, Kt)
    using a1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using b1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
    using c1Type = MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float, false>;
    using bias1Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    Matmul<a1Type, b1Type, c1Type, bias1Type> mm1;

    // define matmul object for matmul(P, V)
    using a2Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using b2Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using c2Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false>;
    using bias2Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    Matmul<a2Type, b2Type, c2Type, bias2Type> mm2;

    // define matmul object for matmul(Q, KV)
    using a3Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using b3Type = MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, T, false>;
    using c3Type = MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float, false>;
    using bias3Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    Matmul<a3Type, b3Type, c3Type, bias3Type> mm3;

    // define matmul object for matmul(Kt, V)
    using a4Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
    using b4Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    using c4Type = MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float, false>;
    using bias4Type = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
    Matmul<a4Type, b4Type, c4Type, bias4Type> mm4;

private:
    __aicore__ inline void InitWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitMask();
    __aicore__ inline void GenerateMask(uint32_t headIdx, float s);
    __aicore__ inline void GenerateDecay(uint32_t headIdx, float s);
    __aicore__ inline void GenerateTailBlockDecay(const AscendC::LocalTensor<float> &decayTensor, uint32_t batchId,
                                                  float s);
    __aicore__ inline float GetSlope(uint32_t headIdx);
    __aicore__ inline void ComputeEachBlock(uint32_t offset, uint32_t headIdx,
                                            const AscendC::LocalTensor<float> &decayTensor);
    __aicore__ inline void ComputeOIntra(uint32_t offset, uint32_t headIdx);
    __aicore__ inline void CopyMIn(uint32_t headIdx, uint32_t maskOffset, uint32_t copyRows);
    __aicore__ inline void CopyDecayIn(uint32_t headIdx);
    __aicore__ inline void ComputePSplit(uint32_t headIdx, uint32_t computeRound,
                                         const AscendC::LocalTensor<float> &pOutTensor);
    __aicore__ inline void CopyPOut(uint32_t computeRound);
    __aicore__ inline void ComputeOInter(uint32_t offset, const AscendC::LocalTensor<float> &qDecayTensor);
    __aicore__ inline void InitKVCache(uint32_t kvCacheOffset);
    __aicore__ inline void UpdateKVCache(uint32_t offset, const AscendC::LocalTensor<float> &kDecayTensor,
                                         const AscendC::LocalTensor<float> &blockDecayTensor);
    __aicore__ inline void SaveKVCache(uint32_t kvCacheOffset);
    __aicore__ inline void WaitKVCacheSaved();
    __aicore__ inline void CopyOIntraIn(uint32_t attentionOffset);
    __aicore__ inline void CalculateOFinal(const AscendC::LocalTensor<float> &oInterTensor, uint32_t attentionOffset);
    __aicore__ inline void CopyAttentionOut(uint32_t attentionOffset);

private:
    AscendC::GlobalTensor<T> queryGM_;
    AscendC::GlobalTensor<T> keyGM_;
    AscendC::GlobalTensor<T> valueGM_;
    AscendC::GlobalTensor<T> slopeRateGM_;
    AscendC::GlobalTensor<T> attentionOutGM_;
    AscendC::GlobalTensor<T> outPWorkspaceGM_;
    AscendC::GlobalTensor<float> outIntraWorkspaceGM_;
    AscendC::GlobalTensor<float> maskGM_;
    AscendC::GlobalTensor<float> decayGM_;
    AscendC::GlobalTensor<T> updatedKeyGM_;
    AscendC::GlobalTensor<float> oInterWorkspaceGM_;
    AscendC::GlobalTensor<T> kvCacheHistoryGM_;
    AscendC::GlobalTensor<T> kvCacheOutGM_;

    const LightningAttentionPrefillTilingData *__restrict tiling_;
    uint32_t headNum_;
    uint32_t headDim_;
    uint32_t blockSize_;
    uint32_t eleCountPerS_;
    uint32_t eleCountPerSSplit_;
    uint32_t mm1RoundM_;
    uint32_t eleCountPerHead_;
    uint32_t eleCountPerBlock_;
    uint32_t eleCountPerOinterSplit_;
    uint32_t eleCountPerKVCache_;
    uint32_t currentCoreId_;
    uint32_t maskMaxSize_;
    uint32_t actualUsedAivNum_;
    const uint16_t *blockCountPerBatch_;
    const uint16_t *tailBlockSize_;
    const uint16_t *headStart_;
    const uint16_t *headEnd_;
    uint32_t eleCountOFinal_;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> maskQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> decayQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> attentionOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> pOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kvCacheBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kPrimeBuf_;
    AscendC::TBuf<> castDataBuf_;
    AscendC::TBuf<> decayHelp_;
};

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR slope_rate, GM_ADDR kv_history, GM_ADDR attention_out,
    GM_ADDR kv_caches, GM_ADDR workspace, const LightningAttentionPrefillTilingData *__restrict tiling,
    AscendC::TPipe *pipe)
{
    currentCoreId_ = GetBlockIdx();
    tiling_ = tiling;
    headNum_ = tiling->laBaseParams.headNum;
    headDim_ = tiling->laBaseParams.headDim;
    blockSize_ = tiling->laBaseParams.blockSize;
    eleCountPerS_ = blockSize_ * blockSize_;
    eleCountPerSSplit_ = tiling->mm1TilingData.baseM * tiling->mm1TilingData.baseN;
    mm1RoundM_ = tiling->mm1TilingData.singleCoreM / tiling->mm1TilingData.baseM;
    eleCountPerHead_ = tiling->laBaseParams.eleCountPerHead;
    eleCountPerBlock_ = tiling->laBaseParams.eleCountPerBlock;
    eleCountPerOinterSplit_ = tiling->mm3TilingData.baseM * tiling->mm3TilingData.baseN;
    eleCountPerKVCache_ = tiling->laBaseParams.headDim * tiling->laBaseParams.headDim;
    actualUsedAivNum_ = tiling->laBaseParams.actualUsedAivNum;
    blockCountPerBatch_ = tiling->laBaseParams.blockCountPerBatch;
    tailBlockSize_ = tiling->laBaseParams.tailBlockSize;
    headStart_ = tiling->laBaseParams.headStart;
    headEnd_ = tiling->laBaseParams.headEnd;

    queryGM_.SetGlobalBuffer((__gm__ T *)query);
    keyGM_.SetGlobalBuffer((__gm__ T *)key);
    valueGM_.SetGlobalBuffer((__gm__ T *)value);
    slopeRateGM_.SetGlobalBuffer((__gm__ T *)slope_rate);
    attentionOutGM_.SetGlobalBuffer((__gm__ T *)attention_out);
    kvCacheHistoryGM_.SetGlobalBuffer((__gm__ T *)kv_history);
    kvCacheOutGM_.SetGlobalBuffer((__gm__ T *)kv_caches);
    InitWorkspace(workspace);

    auto maxBufSize = 128 * 128;
    maskMaxSize_ = 64 * 64;
    eleCountOFinal_ = eleCountPerOinterSplit_ < maskMaxSize_ ? eleCountPerOinterSplit_ : maskMaxSize_;
    pipe->InitBuffer(pOutQueue_, 1, sizeof(float) * maxBufSize);                     // 64k
    if constexpr (!IsSameType<T, float>::value) {
        pipe->InitBuffer(castDataBuf_, sizeof(T) * maxBufSize);                      // 32k
    }
    pipe->InitBuffer(maskQueue_, 1, sizeof(float) * maskMaxSize_);                   // 16k
    pipe->InitBuffer(decayQueue_, 1, sizeof(float) * (blockSize_ + blockSize_ + 8)); // 3k
    pipe->InitBuffer(attentionOutQueue_, 1, sizeof(T) * maskMaxSize_);               // 8k
    pipe->InitBuffer(decayHelp_, sizeof(float) * 3 * blockSize_);                    // 3k
    pipe->InitBuffer(kvCacheBuf_, sizeof(float) * maxBufSize);                       // 64k
    pipe->InitBuffer(kPrimeBuf_, sizeof(T) * headDim_);                              // 0.25k

    InitMask();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::InitWorkspace(GM_ADDR workspace)
{
    // workspace reserved for each core
    uint32_t pWorkSpaceSize = eleCountPerS_ * sizeof(T), oIntraWorkSpaceSize = eleCountPerBlock_ * sizeof(float),
             updatedKeyWorkSpaceSize = eleCountPerBlock_ * sizeof(float),
             baseWorkspaceOffset = (pWorkSpaceSize + oIntraWorkSpaceSize + updatedKeyWorkSpaceSize) * currentCoreId_;
    outPWorkspaceGM_.SetGlobalBuffer((__gm__ T *)(workspace + baseWorkspaceOffset), eleCountPerS_);
    outIntraWorkspaceGM_.SetGlobalBuffer((__gm__ float *)(workspace + baseWorkspaceOffset + pWorkSpaceSize),
                                         eleCountPerBlock_);
    auto updatedKeyOffset = baseWorkspaceOffset + pWorkSpaceSize + oIntraWorkSpaceSize;
    updatedKeyGM_.SetGlobalBuffer((__gm__ T *)(workspace + updatedKeyOffset), eleCountPerBlock_);
    // reuse same workspace
    oInterWorkspaceGM_.SetGlobalBuffer((__gm__ float *)(workspace + updatedKeyOffset), eleCountPerBlock_);

    // workspace shared by each core
    uint32_t sharedWorkspaceOffset =
        (pWorkSpaceSize + oIntraWorkSpaceSize + updatedKeyWorkSpaceSize) * actualUsedAivNum_;
    maskGM_.SetGlobalBuffer((__gm__ float *)(workspace + sharedWorkspaceOffset));
    decayGM_.SetGlobalBuffer(
        (__gm__ float *)(workspace + sharedWorkspaceOffset + headNum_ * blockSize_ * blockSize_ * sizeof(float)));
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::InitMask()
{
    uint32_t curHeadIdx = currentCoreId_;
    float s;

    while (curHeadIdx < headNum_) {
        s = GetSlope(curHeadIdx);
        GenerateMask(curHeadIdx, s);
        GenerateDecay(curHeadIdx, s);
        curHeadIdx += actualUsedAivNum_;
    }
    AscendC::SyncAll();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::GenerateMask(uint32_t headIdx, float s)
{
    // use kvCacheBuf (128, 128) as help tensor to generate mask then copy to GM
    // blockSize greater than 128 shall iterate multiple times to get all mask values
    auto helpTensor = kvCacheBuf_.Get<float>();

    uint32_t mOffset;
    uint32_t tmp = 0xFF800000; // -inf
    int32_t eleCountPerMask = blockSize_ * blockSize_;
    int32_t eleCountPerHelp = 128 * 128;
    int32_t iterTimes = (eleCountPerMask + eleCountPerHelp - 1) / (eleCountPerHelp);
    int32_t blocksPerIter = blockSize_ / iterTimes;
    int32_t eleCountPerIter = blocksPerIter * blockSize_;

    int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t eventIdMte3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    int32_t eventIdMte3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));

    for (int32_t iter = 0; iter < iterTimes; ++iter) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(helpTensor, *((float *)&tmp), eleCountPerIter);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t b = 0; b < blocksPerIter; ++b) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CreateVecIndex(helpTensor[b * blockSize_], (float)-(iter * blocksPerIter + b),
                                    (iter * blocksPerIter + b) + 1);
            AscendC::PipeBarrier<PIPE_V>();

            // CreateVecIndex() will pad to 8 float elems, set padded values back to -inf
            for (int32_t i = b + 1; i < 8; ++i) {
                helpTensor.SetValue(iter * blocksPerIter + b * blockSize_ + i, *((float *)&tmp));
            }
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(helpTensor, helpTensor, s, eleCountPerIter);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(helpTensor, helpTensor, eleCountPerIter);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        mOffset = headIdx * blockSize_ * blockSize_ + iter * eleCountPerIter;
        AscendC::DataCopy(maskGM_[mOffset], helpTensor, eleCountPerIter);
        SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
        SetFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventIdMte3ToS);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::Process()
{
    float s;
    uint16_t absoluteHeadIdx = headStart_[currentCoreId_];
    uint32_t offset = absoluteHeadIdx * eleCountPerHead_, blockOffset;
    uint32_t kvCacheOffset = absoluteHeadIdx * eleCountPerKVCache_;
    bool isFirstLoop = true;
    for (uint16_t relativeHeadIdx, batchId; absoluteHeadIdx <= headEnd_[currentCoreId_];
         ++absoluteHeadIdx, offset += eleCountPerHead_, kvCacheOffset += eleCountPerKVCache_) {
        if (isFirstLoop) {
            isFirstLoop = false;
        } else {
            WaitKVCacheSaved();
        }
        InitKVCache(kvCacheOffset);
        relativeHeadIdx = absoluteHeadIdx % headNum_;
        batchId = absoluteHeadIdx / headNum_;
        CopyDecayIn(relativeHeadIdx);
        auto decayTensor = decayQueue_.DeQue<float>();

        blockOffset = offset;
        for (uint32_t blockIdx = 0; blockIdx + 1 < blockCountPerBatch_[batchId]; ++blockIdx) {
            ComputeEachBlock(blockOffset, relativeHeadIdx, decayTensor);
            blockOffset += eleCountPerBlock_;
        }
        if (tailBlockSize_[batchId] != 0) {
            GenerateTailBlockDecay(decayTensor, batchId, GetSlope(relativeHeadIdx));
        }
        ComputeEachBlock(blockOffset, relativeHeadIdx, decayTensor);
        decayQueue_.FreeTensor(decayTensor);
        SaveKVCache(kvCacheOffset);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CopyDecayIn(uint32_t headIdx)
{
    auto decayLocal = decayQueue_.AllocTensor<float>();
    uint32_t decayOffset = headIdx * (blockSize_ + blockSize_ + 8);
    AscendC::DataCopy(decayLocal, decayGM_[decayOffset], blockSize_ + blockSize_ + 8);
    decayQueue_.EnQue<float>(decayLocal);
}

template <typename T>
__aicore__ inline float LightningAttentionPrefill<T>::GetSlope(uint32_t headIdx)
{
    float s;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        s = AscendC::ToFloat(slopeRateGM_.GetValue(headIdx));
    } else {
        s = (float)slopeRateGM_.GetValue(headIdx);
    }
    return s;
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::GenerateDecay(uint32_t headIdx, float s)
{
    auto helpTensor = decayHelp_.Get<float>();

    // q_decay
    auto qDecayTensor = helpTensor;
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CreateVecIndex(qDecayTensor, (float)1, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(qDecayTensor, qDecayTensor, -s, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(qDecayTensor, qDecayTensor, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();

    // k_decay
    auto kDecayTensor = helpTensor[blockSize_];
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CreateVecIndex(kDecayTensor, (float)1, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(kDecayTensor, kDecayTensor, (float)-1, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(kDecayTensor, kDecayTensor, (float)(int32_t)blockSize_, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(kDecayTensor, kDecayTensor, -s, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(kDecayTensor, kDecayTensor, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();

    // block_decay
    auto blockDecayTensor = helpTensor[blockSize_ + blockSize_];
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(blockDecayTensor, -s * blockSize_, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(blockDecayTensor, blockDecayTensor, 8);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t decayOffset = headIdx * (blockSize_ + blockSize_ + 8);
    int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t eventIdMte3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
    AscendC::DataCopy(decayGM_[decayOffset], helpTensor, blockSize_ + blockSize_ + 8);
    SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::GenerateTailBlockDecay(
    const AscendC::LocalTensor<float> &decayTensor, uint32_t batchId, float s)
{
    auto currentTailBlockSize = (int32_t)tailBlockSize_[batchId];
    int32_t tailSizePad = (currentTailBlockSize + 8 - 1) / 8 * 8;
    auto kDecayTensor = decayTensor[blockSize_];
    AscendC::Duplicate(kDecayTensor, (float)0, blockSize_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CreateVecIndex(kDecayTensor, (float)1, tailSizePad);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(kDecayTensor, kDecayTensor, (float)-1, tailSizePad);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(kDecayTensor, kDecayTensor, (float)currentTailBlockSize, tailSizePad);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(kDecayTensor, kDecayTensor, -s, tailSizePad);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(kDecayTensor, kDecayTensor, tailSizePad);
    AscendC::PipeBarrier<PIPE_V>();

    auto blockDecayTensor = decayTensor[blockSize_ + blockSize_];
    AscendC::Duplicate(blockDecayTensor, -s * currentTailBlockSize, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(blockDecayTensor, blockDecayTensor, 8);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::ComputeEachBlock(uint32_t offset, uint32_t headIdx,
                                                                      const AscendC::LocalTensor<float> &decayTensor)
{
    auto qDecayTensor = decayTensor;
    auto kDecayTensor = decayTensor[blockSize_];
    auto blockDecayTensor = decayTensor[blockSize_ + blockSize_];

    ComputeOIntra(offset, headIdx);
    ComputeOInter(offset, qDecayTensor);
    UpdateKVCache(offset, kDecayTensor, blockDecayTensor);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::ComputeOIntra(uint32_t offset, uint32_t headIdx)
{
    // Step 1: calculate S = matmul(Q, K)
    mm1.SetTensorA(queryGM_[offset]);
    mm1.SetTensorB(keyGM_[offset], true);
    for (uint32_t computeRound = 0; mm1.template Iterate<true>(); ++computeRound) {
        auto pOutTensor = pOutQueue_.AllocTensor<float>();
        mm1.template GetTensorC<true>(pOutTensor, false, true);
        // Step 2: calculate P = mul(S, M)
        ComputePSplit(headIdx, computeRound, pOutTensor);
        CopyPOut(computeRound);
    }
    mm1.End();

    // Step 3: calculate Ointra = matmul(P, V)
    mm2.SetTensorA(outPWorkspaceGM_);
    mm2.SetTensorB(valueGM_[offset]);
    mm2.template IterateAll<true>(outIntraWorkspaceGM_);
    mm2.End();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::ComputeOInter(uint32_t offset,
                                                                   const AscendC::LocalTensor<float> &qDecayTensor)
{
    float qDecay;
    uint32_t mm3BaseM = tiling_->mm3TilingData.baseM;
    // Step 1: calculate O_inter = matmul(Q, KV)
    auto kvCacheTensor = kvCacheBuf_.Get<float>();
    mm3.SetWorkspace(oInterWorkspaceGM_);
    mm3.SetTensorA(queryGM_[offset]);
    if constexpr (IsSameType<T, float>::value) {
        mm3.SetTensorB(kvCacheTensor);
    } else {
        auto kvCastBuf = castDataBuf_.Get<T>();
        AscendC::Cast(kvCastBuf, kvCacheTensor, RoundMode::CAST_ROUND, eleCountPerKVCache_);
        mm3.SetTensorB(kvCastBuf);
    }
    mm3.template Iterate<false>();
    for (uint32_t computeRound = 0, attentionBaseOffset = 0, totalRound = blockSize_ / mm3BaseM;
         computeRound < totalRound; ++computeRound, attentionBaseOffset += eleCountPerOinterSplit_) {
        auto oInterTensor = pOutQueue_.AllocTensor<float>();
        mm3.template GetTensorC<false>(oInterTensor, false, true);
        // headDim <= 128, which means only M will split, N will not split
        // Step 2: update O_inter with decay
        for (uint32_t b = 0; b < mm3BaseM; b++) {
            qDecay = qDecayTensor.GetValue(computeRound * mm3BaseM + b);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(oInterTensor[b * headDim_], oInterTensor[b * headDim_], qDecay, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t attentionRelativeOffset = 0; attentionRelativeOffset < eleCountPerOinterSplit_;
             attentionRelativeOffset += eleCountOFinal_) {
            CopyOIntraIn(attentionBaseOffset + attentionRelativeOffset);
            // Step 3: Add O_inter and Cast
            CalculateOFinal(oInterTensor, attentionRelativeOffset);
            // Step 4: Save to O
            CopyAttentionOut(offset + attentionBaseOffset + attentionRelativeOffset);
        }
        pOutQueue_.FreeTensor(oInterTensor);
    }
    mm3.End();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::InitKVCache(uint32_t kvCacheOffset)
{
    auto kvCacheTensor = kvCacheBuf_.Get<float>();
    if (kvCacheHistoryGM_.GetPhyAddr() == nullptr) {
        AscendC::Duplicate(kvCacheTensor, 0.0f, eleCountPerKVCache_);
    } else {
        if constexpr (IsSameType<T, float>::value) {
            AscendC::DataCopy(kvCacheTensor, kvCacheHistoryGM_[kvCacheOffset], eleCountPerKVCache_);
        } else {
            auto tmpBuf = kvCacheTensor[eleCountPerKVCache_ / 2].ReinterpretCast<T>();
            AscendC::DataCopy(tmpBuf, kvCacheHistoryGM_[kvCacheOffset], eleCountPerKVCache_);
            int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
            AscendC::Cast(kvCacheTensor, tmpBuf, RoundMode::CAST_NONE, eleCountPerKVCache_);
        }
    }
}


template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::UpdateKVCache(uint32_t offset,
                                                                   const AscendC::LocalTensor<float> &kDecayTensor,
                                                                   const AscendC::LocalTensor<float> &blockDecayTensor)
{
    // Step 1: update & save K
    auto kTensor = kPrimeBuf_.Get<T>();
    int32_t eventIdMte3ToMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    int32_t eventIdMte2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    for (uint32_t b = 0; b < blockSize_; b++) {
        float kDecay = kDecayTensor.GetValue(b);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(kTensor, keyGM_[offset + b * headDim_], headDim_);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<AscendC::HardEvent::MTE2_V>(eventIdMte2ToV);
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Muls(kTensor, kTensor, kDecay, headDim_);
        } else {
            auto tmpBuf = castDataBuf_.Get<float>();
            AscendC::Cast(tmpBuf, kTensor, RoundMode::CAST_NONE, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmpBuf, tmpBuf, kDecay, headDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kTensor, tmpBuf, RoundMode::CAST_ROUND, headDim_);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(updatedKeyGM_[b * headDim_], kTensor, headDim_);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    // Step 2: update KV
    auto kvCacheTensor = kvCacheBuf_.Get<float>();
    float blockDecay = blockDecayTensor.GetValue(0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(kvCacheTensor, kvCacheTensor, blockDecay, eleCountPerKVCache_);

    // Step 3: calculate KV_cur = matmul(Kt, V)
    mm4.SetTensorA(updatedKeyGM_, true);
    mm4.SetTensorB(valueGM_[offset]);
    // KV_cur shape is (headDim, headDim), which means matmul will finish in one round
    if (mm4.template Iterate<true>()) {
        // Step 4: Add KV_cur
        auto kvCurrentTensor = pOutQueue_.AllocTensor<float>();
        mm4.template GetTensorC<true>(kvCurrentTensor, false, true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(kvCacheTensor, kvCacheTensor, kvCurrentTensor, eleCountPerKVCache_);
        pOutQueue_.FreeTensor<float>(kvCurrentTensor);
    }
    mm4.End();
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::SaveKVCache(uint32_t kvCacheOffset)
{
    auto kvCacheTensor = kvCacheBuf_.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(kvCacheOutGM_[kvCacheOffset], kvCacheTensor, eleCountPerKVCache_);
    } else {
        auto tmpBuf = castDataBuf_.Get<T>();
        int32_t eventIdMte3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        SetFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<AscendC::HardEvent::MTE3_V>(eventIdMte3ToV);
        AscendC::Cast(tmpBuf, kvCacheTensor, RoundMode::CAST_ROUND, eleCountPerKVCache_);
        int32_t eventIdVToMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::DataCopy(kvCacheOutGM_[kvCacheOffset], tmpBuf, eleCountPerKVCache_);
    }
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::WaitKVCacheSaved()
{
    int32_t eventIdMte3ToMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CopyOIntraIn(uint32_t attentionOffset)
{
    auto oIntraTensor = maskQueue_.AllocTensor<float>();
    AscendC::DataCopy(oIntraTensor, outIntraWorkspaceGM_[attentionOffset], eleCountOFinal_);
    maskQueue_.EnQue(oIntraTensor);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CalculateOFinal(const AscendC::LocalTensor<float> &oInterTensor,
                                                                     uint32_t attentionOffset)
{
    auto oIntraTensor = maskQueue_.DeQue<float>();
    auto oFinalTensor = attentionOutQueue_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Add(oFinalTensor, oIntraTensor, oInterTensor[attentionOffset], eleCountOFinal_);
    } else {
        AscendC::Add(oIntraTensor, oIntraTensor, oInterTensor[attentionOffset], eleCountOFinal_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(oFinalTensor, oIntraTensor, RoundMode::CAST_ROUND, eleCountOFinal_);
    }

    maskQueue_.FreeTensor(oIntraTensor);
    attentionOutQueue_.EnQue<T>(oFinalTensor);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CopyAttentionOut(uint32_t attentionOffset)
{
    auto oFinalTensor = attentionOutQueue_.DeQue<T>();
    AscendC::DataCopy(attentionOutGM_[attentionOffset], oFinalTensor, eleCountOFinal_);
    attentionOutQueue_.FreeTensor(oFinalTensor);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CopyMIn(uint32_t headIdx, uint32_t maskOffset, uint32_t copyRows)
{
    auto maskLocal = maskQueue_.AllocTensor<float>();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopyParams copyParams{
        (uint16_t)copyRows, (uint16_t)(tiling_->mm1TilingData.baseN * sizeof(float) / DEFAULT_C0_SIZE),
        (uint16_t)((tiling_->mm1TilingData.N - tiling_->mm1TilingData.baseN) * sizeof(float) / DEFAULT_C0_SIZE),
        (uint16_t)0};
    AscendC::DataCopy(maskLocal, maskGM_[maskOffset], copyParams);
    maskQueue_.EnQue<float>(maskLocal);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::ComputePSplit(uint32_t headIdx, uint32_t computeRound,
                                                                   const AscendC::LocalTensor<float> &pOutTensor)
{
    uint32_t multiplyRound = (eleCountPerSSplit_ + maskMaxSize_ - 1) / maskMaxSize_,
             multiplyEleCount = eleCountPerSSplit_ < maskMaxSize_ ? eleCountPerSSplit_ : maskMaxSize_,
             copyRows = multiplyEleCount / tiling_->mm1TilingData.baseN;
    uint32_t maskOffset = headIdx * blockSize_ * blockSize_, pOutTensorOffset = 0;
    maskOffset += computeRound % mm1RoundM_ * tiling_->mm1TilingData.baseM * tiling_->mm1TilingData.N +
                  computeRound / mm1RoundM_ * tiling_->mm1TilingData.baseN;
    for (uint32_t multiplyRoundIdx = 0; multiplyRoundIdx < multiplyRound;
         ++multiplyRoundIdx, pOutTensorOffset += multiplyEleCount, maskOffset += copyRows * tiling_->mm1TilingData.N) {
        CopyMIn(headIdx, maskOffset, copyRows);
        auto maskLocal = maskQueue_.DeQue<float>();
        AscendC::Mul(pOutTensor[pOutTensorOffset], pOutTensor[pOutTensorOffset], maskLocal, multiplyEleCount);
        maskQueue_.FreeTensor(maskLocal);
    }
    if constexpr (!IsSameType<T, float>::value) {
        auto tempOutTensor = castDataBuf_.Get<T>();
        AscendC::Cast(tempOutTensor, pOutTensor, RoundMode::CAST_ROUND, eleCountPerSSplit_);
    }
    pOutQueue_.EnQue<float>(pOutTensor);
}

template <typename T>
__aicore__ inline void LightningAttentionPrefill<T>::CopyPOut(uint32_t computeRound)
{
    auto pOutTensor = pOutQueue_.DeQue<float>();
    uint32_t offset = computeRound % mm1RoundM_ * tiling_->mm1TilingData.baseM * tiling_->mm1TilingData.N +
                      computeRound / mm1RoundM_ * tiling_->mm1TilingData.baseN;
    AscendC::DataCopyParams copyParams{
        (uint16_t)tiling_->mm1TilingData.baseM, (uint16_t)(tiling_->mm1TilingData.baseN * sizeof(T) / DEFAULT_C0_SIZE),
        (uint16_t)0,
        (uint16_t)((tiling_->mm1TilingData.N - tiling_->mm1TilingData.baseN) * sizeof(T) / DEFAULT_C0_SIZE)};
    if constexpr (IsSameType<T, float>::value) {
        AscendC::DataCopy(outPWorkspaceGM_[offset], pOutTensor, copyParams);
    } else {
        auto tempOutTensor = castDataBuf_.Get<T>();
        AscendC::DataCopy(outPWorkspaceGM_[offset], tempOutTensor, copyParams);
    }
    pOutQueue_.FreeTensor(pOutTensor);
}

} // namespace LightningAttention

#endif //LIGHTNING_ATTENTION_PREFILL_H
