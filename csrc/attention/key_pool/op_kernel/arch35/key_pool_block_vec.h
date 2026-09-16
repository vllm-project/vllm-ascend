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
 * \file key_pool_block_vec.h
 * \brief
 */

#ifndef KEY_POOL_BLOCK_VEC_H
#define KEY_POOL_BLOCK_VEC_H

#include "key_pool_comm_arch35.h"
#include "key_pool_tools.h"
#include "vf/key_pool_vf_softmax.h"
#include "vf/key_pool_vf_add.h"
#include "vf/key_pool_vf_mul.h"
#include "../key_pool_layer_norm.h"

using namespace AscendC;

namespace KeyPool {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP>
class KeyPoolBlockVector {
public:
    static constexpr bool HIDDEN_STATES_DTYPE = COMP::hiddenStatesDtype == HIDDEN_STATES_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = BUFFER_SIZE_BYTE_32K; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float SOFTMAX_MIN_NUM = -2e38;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using HIDDEN_STATES_T = typename AscendC::Conditional<HIDDEN_STATES_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline KeyPoolBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const KeyPoolTools<COMP> &tools);
    __aicore__ inline void Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk, __gm__ uint8_t *gateWeight,
                                __gm__ uint8_t *normWeight, __gm__ uint8_t *normBias, __gm__ uint8_t *stateCache,
                                __gm__ uint8_t *ape, __gm__ uint8_t *cacheBlockTable, __gm__ uint8_t *seqLens,
                                __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *pooledKeyOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline void SaveNormalizedTail(const Vec1RunInfo &info);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm,
                                                GlobalTensor<T> normalizedKvGm);

protected:
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;
    GlobalTensor<T> normalizedKvGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd,
                                             uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize,
                                           uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info, KeyPoolVec1SliceIterator<COMP> &sliceIterator,
                                             const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline void CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                         const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    __aicore__ inline void AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal, const LocalTensor<O> &srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyWithOutputQue(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyWithInputQue(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                                uint32_t copyRowCount, uint32_t copyColCount,
                                                uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void AddMultiDataToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                            uint32_t dealRowCount, uint32_t dealColCount, uint32_t srcSingleRowCount,
                                            uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset);
    __aicore__ inline void PadAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                   const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                   const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                   uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                             const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo,
                                             uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                             const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                             uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t stateIdx);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                              const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx,
                                              uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                              uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void SaveToWorkSpace(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &cacheTcGm,
                                           const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                           uint32_t dDealSize);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm,
                                             const GlobalTensor<T> &srcGm, const LocalTensor<T> &srcLocal,
                                             const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void RoundToHiddenDtype(const LocalTensor<T> &srcLocal, uint32_t elementCount);
    __aicore__ inline void AddApeToPooledScore(const LocalTensor<T> &scoreLocal, uint32_t poolCount,
                                               uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const Vec1RunInfo &info, const LoopInfo &loopInfo,
                                          const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo,
                                          uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t needDealTcSize);
    __aicore__ inline void GatherKvForNorm(const LocalTensor<T> &kvLocal, const Vec1RunInfo &info,
                                           const LoopInfo &loopInfo, const StatisticInfo &statisticInfo,
                                           const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                           uint32_t needDealTcSize);
    __aicore__ inline void PrepareNormalizedKv(const Vec1RunInfo &info, const Vec1SplitInfo &splitInfo,
                                               const LoopInfo &loopInfo);
    __aicore__ inline void CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb, const Vec1SliceInfo &sliceInfo,
                                                  uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
    template <bool IS_SCORE>
    __aicore__ inline void ReadState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t compressedCnt_ = 0;
    uint32_t prevApeDStartIdx_ = 0;
    uint32_t prevApeDDealSize_ = 0;
    bool apeIsLoad_ = false;
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    KeyPoolTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<T> normWeightGm_;
    GlobalTensor<T> normBiasGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<HIDDEN_STATES_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    // TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> apeUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> apeBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TQue<QuePosition::VECIN, 1> inputQue3;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;
    bool hasLayerNorm_ = false;
    uint64_t normalizedKvDbOffset_ = 0;
    uint32_t normalizedPoolBase_ = 0;
};

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::InitParams(const ConstInfo &constInfo, const KeyPoolTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk,
                                                      __gm__ uint8_t *gateWeight, __gm__ uint8_t *normWeight,
                                                      __gm__ uint8_t *normBias, __gm__ uint8_t *stateCache,
                                                      __gm__ uint8_t *ape, __gm__ uint8_t *cacheBlockTable,
                                                      __gm__ uint8_t *seqLens, __gm__ uint8_t *seqUsed,
                                                      __gm__ uint8_t *startPos, __gm__ uint8_t *pooledKeyOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)cacheBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T *)stateCache);
    hasLayerNorm_ = (normWeight != nullptr);
    if (hasLayerNorm_) {
        normWeightGm_.SetGlobalBuffer((__gm__ T *)normWeight);
        normBiasGm_.SetGlobalBuffer((__gm__ T *)normBias);
    }
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ HIDDEN_STATES_T *)pooledKeyOut);
    isExistSeqUsed_ = (seqUsed != nullptr);
    isExistStartPos_ = (startPos != nullptr);
    if constexpr (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
    }
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    // The current Norm-on tiling uses kBaseNum=1, so the alternating
    // multi-K fallback queue only needs the reduced reservation on A5.
    pipe->InitBuffer(inputQue3, 1, hasLayerNorm_ ? BUFFER_SIZE_BYTE_16K : BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    // Norm-on gathers one full FP32 compression group; the regular pooling
    // path only needs one 32 KiB group and must stay within the A5 Vector UB.
    pipe->InitBuffer(tmpBuff2, hasLayerNorm_ ? BUFFER_SIZE_BYTE_64K : BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_32K);
    apeUb = apeBuf.Get<T>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::FreeEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm,
                                                                      GlobalTensor<T> scoreMm1ResGm,
                                                                      GlobalTensor<T> kvCacheTcGm,
                                                                      GlobalTensor<T> scoreCacheTcGm,
                                                                      GlobalTensor<T> normalizedKvGm)
{
    this->kvMm1ResGm_ = kvMm1ResGm;
    this->scoreMm1ResGm_ = scoreMm1ResGm;
    this->kvCacheTcGm_ = kvCacheTcGm;
    this->scoreCacheTcGm_ = scoreCacheTcGm;
    this->normalizedKvGm_ = normalizedKvGm;
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVector<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx,
                                                           uint32_t dDealSize)
{
    if (apeIsLoad_ && prevApeDStartIdx_ == dStartIdx && prevApeDDealSize_ == dDealSize) {
        return;
    }

    uint32_t copyRowCount = coff_ * cmpRatio_;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;

    DataCopyWithInputQue(apeUb, apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);

    prevApeDStartIdx_ = dStartIdx;
    prevApeDDealSize_ = dDealSize;
    apeIsLoad_ = true;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
                                                               const LocalTensor<T> &apeUb,
                                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleRowElemNum;
        uint32_t row = tcDealSize == 1 ? sliceInfo.validSeqCnt : (cmpRatio_ - sliceInfo.headHolderSeqCnt);
        AddVF(scoreLocal[scoreOffset], apeUb[apeOffset], coff_ * row, dDealSize, dDealSize);
        scoreOffset += row * singleRowElemNum;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint64_t apeOffset = 0;
        uint32_t row = cmpRatio_ - sliceInfo.tailHolderSeqCnt;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * cmpRatio_ * singleRowElemNum;
        AddVF(scoreLocal[tailScoreOffset], apeUb[apeOffset], coff_ * row, dDealSize, dDealSize);
    }
    if (tcDealSize == 0) {
        return;
    }
    uint32_t row = cmpRatio_;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        AddVF(scoreLocal[scoreOffset + r * row * singleRowElemNum], apeUb, coff_ * row, dDealSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::AddSingleApeToScore(const LocalTensor<T> &scoreLocal,
                                                                     const LocalTensor<T> &apeUb,
                                                                     const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * coff_;
    uint32_t dealRowCount = min(sliceInfo.sIdx, cmpRatio_);
    uint64_t scoreOffset = (cmpRatio_ - dealRowCount) * singleRowElemNum;
    uint64_t apeOffset = (cmpRatio_ - dealRowCount) * singleRowElemNum;
    AddVF(scoreLocal[scoreOffset], apeUb[apeOffset], dealRowCount, dDealSize, singleRowElemNum);
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal,
                                                                     const LocalTensor<O> &srcLocal,
                                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                                     uint32_t srcSingleRowCount,
                                                                     uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal,
                                                                     const GlobalTensor<O> &srcGm,
                                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                                     uint32_t srcSingleRowCount,
                                                                     uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm,
                                                                     const LocalTensor<O> &srcLocal,
                                                                     uint32_t copyRowCount, uint32_t copyColCount,
                                                                     uint32_t srcSingleRowCount,
                                                                     uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstGm, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::DataCopyWithOutputQue(const GlobalTensor<O> &dstGm,
                                                                       const LocalTensor<O> &srcLocal,
                                                                       uint32_t copyRowCount, uint32_t copyColCount,
                                                                       uint32_t srcSingleRowCount,
                                                                       uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    uint32_t singleCopyRowCount = BUFFER_SIZE_BYTE_32K / (copyColCount * sizeof(O));
    for (uint32_t rowCount = 0; rowCount < copyRowCount; rowCount += singleCopyRowCount) {
        uint64_t srcOffset = rowCount * srcSingleRowCount;
        uint64_t dstOffset = rowCount * dstSingleRowCount;
        uint32_t curCopyRowCount = min(singleCopyRowCount, copyRowCount - rowCount);

        LocalTensor<O> outputUb = outputQue1.AllocTensor<O>();

        DataCopyAlignUbToUb(outputUb, srcLocal[srcOffset], curCopyRowCount, copyColCount, srcSingleRowCount,
                            copyColCount);

        outputQue1.EnQue(outputUb);
        outputQue1.DeQue<O>();

        DataCopyAlignUbToGm(dstGm[dstOffset], outputUb, curCopyRowCount, copyColCount, copyColCount, dstSingleRowCount);

        outputQue1.FreeTensor(outputUb);
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::DataCopyWithInputQue(const LocalTensor<O> &dstLocal,
                                                                      const GlobalTensor<O> &srcGm,
                                                                      uint32_t copyRowCount, uint32_t copyColCount,
                                                                      uint32_t srcSingleRowCount,
                                                                      uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    uint32_t singleCopyRowCount = BUFFER_SIZE_BYTE_32K / (copyColCount * sizeof(O));
    for (uint32_t rowCount = 0; rowCount < copyRowCount; rowCount += singleCopyRowCount) {
        uint64_t srcOffset = rowCount * srcSingleRowCount;
        uint64_t dstOffset = rowCount * dstSingleRowCount;
        uint32_t curCopyRowCount = min(singleCopyRowCount, copyRowCount - rowCount);

        LocalTensor<O> inputUb = inputQue2.AllocTensor<O>();

        DataCopyAlignGmToUb(inputUb, srcGm[srcOffset], curCopyRowCount, copyColCount, srcSingleRowCount, copyColCount);

        inputQue2.EnQue(inputUb);
        inputQue2.DeQue<O>();

        DataCopyAlignUbToUb(dstLocal[dstOffset], inputUb, curCopyRowCount, copyColCount, copyColCount,
                            dstSingleRowCount);

        inputQue2.FreeTensor(inputUb);
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVector<COMP>::AddMultiDataToUb(const LocalTensor<O> &dstLocal,
                                                                  const GlobalTensor<O> &srcGm, uint32_t dealRowCount,
                                                                  uint32_t dealColCount, uint32_t srcSingleRowCount,
                                                                  uint32_t dstSingleRowCount, uint32_t repeatTimes,
                                                                  uint64_t offset)
{
    if (dealRowCount == 0 || dealColCount == 0 || repeatTimes == 0) {
        return;
    }
    // A full D x D tile can exceed the 32 KiB input queue. Process bounded
    // row chunks so every DMA fits and every destination row is initialized.
    uint32_t maxRowsPerChunk = BUFFER_SIZE_BYTE_32K / (dstSingleRowCount * sizeof(O));
    maxRowsPerChunk = max(maxRowsPerChunk, 1U);
    for (uint32_t rowOffset = 0; rowOffset < dealRowCount; rowOffset += maxRowsPerChunk) {
        uint32_t curRows = min(maxRowsPerChunk, dealRowCount - rowOffset);
        uint32_t cnt = curRows * dstSingleRowCount;
        uint64_t dstOffset = static_cast<uint64_t>(rowOffset) * dstSingleRowCount;

        for (uint32_t repeatIdx = 0; repeatIdx < repeatTimes; repeatIdx++) {
            LocalTensor<O> inputLocal = inputQue2.AllocTensor<O>();
            uint64_t srcOffset =
                static_cast<uint64_t>(rowOffset) * srcSingleRowCount + static_cast<uint64_t>(repeatIdx) * offset;
            DataCopyAlignGmToUb(inputLocal, srcGm[srcOffset], curRows, dealColCount, srcSingleRowCount,
                                dstSingleRowCount);
            inputQue2.EnQue(inputLocal);
            inputQue2.DeQue<O>();

            PipeBarrier<PIPE_V>();
            if (repeatIdx == 0) {
                MultiAddVF<true>(dstLocal[dstOffset], inputLocal, curRows, dealColCount, dstSingleRowCount, 1, cnt);
            } else {
                MultiAddVF<false>(dstLocal[dstOffset], inputLocal, curRows, dealColCount, dstSingleRowCount, 1, cnt);
            }
            inputQue2.FreeTensor(inputLocal);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::PadAlign(const LocalTensor<T> &dstLocal,
                                                          const LocalTensor<T> &srcLocal,
                                                          const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                          uint32_t dDealSize)
{
    // Ub data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * cmpRatio_ - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = srcSingleRowElemNum; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum;

    uint64_t dstUbOffset = sliceInfo.key_pooledScCnt * cmpRatio_ * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + cmpRatio_) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                            copyRowCount - min(copyRowCount, cmpRatio_), copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dDealSize;
    }
    // 右侧
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount, srcSingleRowCount,
                        dstSingleRowCount);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVector<COMP>::OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                                         const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                                         const GlobalTensor<int32_t> &blockTableGm,
                                                         const GlobalTensor<T> &cacheTcGm, const Vec1RunInfo &info,
                                                         const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                                         uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if (!hasLayerNorm_) {
        SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));
    }
    ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                        static_cast<uint32_t>(IS_SCORE));

    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t nextC1V1DbIdx = (info.c1v1DbIdx + 1) % constInfo_.dbWorkspaceRatio;
        GlobalTensor<T> nextCacheTcGm = cacheTcGm[nextC1V1DbIdx * cmpRatio_ * constInfo_.headDim];
        SaveToWorkSpace(srcLocal, nextCacheTcGm, sliceInfo, loopInfo, dStartIdx, dDealSize);
    }
    if (sliceInfo.compressTcSize > 0) {
        PadAlign(dstLocal, srcLocal, sliceInfo, dStartIdx, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm[info.c1v1DbIdx * cmpRatio_ * constInfo_.headDim];
            LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx, globalSeqIdx,
                              dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal,
                                                                   const GlobalTensor<T> &srcGm,
                                                                   const Vec1SliceInfo &sliceInfo,
                                                                   const StatisticInfo &statisticInfo,
                                                                   uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = statisticInfo.dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = (uint64_t)sliceInfo.dealedSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    if (constInfo_.kBaseNum == 1) {
        DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
    } else {
        AddMultiDataToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount,
                         constInfo_.kBaseNum, constInfo_.mm1KvResSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::SaveToWorkSpace(const LocalTensor<T> &srcLocal,
                                                                 const GlobalTensor<T> &cacheTcGm,
                                                                 const Vec1SliceInfo &sliceInfo,
                                                                 const LoopInfo &loopInfo, uint32_t dStartIdx,
                                                                 uint32_t dDealSize)
{
    uint64_t curSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt;
    uint64_t totalSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.bSeqUsed;
    if (!loopInfo.isCoreRowLast || !loopInfo.isCoreLoopLast || !sliceInfo.isLast || totalSeqLen < cmpRatio_ ||
        curSeqLen > Trunc(totalSeqLen, (uint64_t)cmpRatio_) - cmpRatio_) {
        return;
    }
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset =
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, cmpRatio_)) * srcSingleRowElemNum;
    DataCopyWithOutputQue(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                          curSeqLen - max(curSeqLen - cmpRatio_, sliceInfo.bStartPos), dDealSize, coff_ * dDealSize,
                          constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::LoadFromWorkSpace(
    const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm, const GlobalTensor<T> &srcGm,
    const LocalTensor<T> &srcLocal, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, cmpRatio_);
    uint64_t dstLocalOffset = (sliceInfo.key_pooledScCnt * cmpRatio_ + cmpRatio_ - copyRowCount) * dstSingleRowElemNum;
    if (loopInfo.isCoreRowFirst && loopInfo.isCoreLoopFirst && sliceInfo.isFirst) { // 从cacheGm获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim;
        uint64_t srcLocalOffset = dStartIdx;

        DataCopyWithInputQue(dstLocal[dstLocalOffset], cacheTcGm[srcLocalOffset], copyRowCount, dDealSize,
                             srcSingleRowElemNum, coff_ * dDealSize);
    } else if (sliceInfo.isFirst) { // 从存放MatMul结果的WorkSpace中获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset =
            (globalSeqIdx + sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum + dStartIdx;

        if (constInfo_.kBaseNum == 1) {
            DataCopyWithInputQue(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                                 srcSingleRowElemNum, coff_ * dDealSize);
        } else {
            AddMultiDataToUb(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                             srcSingleRowElemNum, coff_ * dDealSize, constInfo_.kBaseNum, constInfo_.mm1KvResSize);
        }
    } else { // 从UB中获取
        uint32_t srcSingleRowElemNum = dDealSize * coff_;
        uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
        DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                            srcSingleRowElemNum, coff_ * dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
                                                                    const GlobalTensor<T> &state,
                                                                    const GlobalTensor<int32_t> &blockTableGm,
                                                                    uint32_t batchIdx, uint32_t startSeqIdx,
                                                                    uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                    uint32_t dDealSize, uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::LINEAR_BUFFER) {
        uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        while (copyFinishRowCnt < seqCnt) {
            uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
            uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
            uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCount > seqCnt) {
                copyRowCount = seqCnt - copyFinishRowCnt;
            }
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;

            DataCopyWithInputQue(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                 dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, coff_ * dDealSize);
            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    } else {
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        uint64_t idInBlockTable = blockTableGm.GetValue(batchIdx);
        while (copyFinishRowCnt < seqCnt) {
            uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
            uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCount > seqCnt) {
                copyRowCount = seqCnt - copyFinishRowCnt;
            }
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;

            DataCopyWithInputQue(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                 dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR, coff_ * dDealSize);
            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                   const GlobalTensor<int32_t> &blockTableGm,
                                                                   const LocalTensor<T> &input, uint32_t batchIdx,
                                                                   uint32_t startSeqIdx, uint32_t endSeqIdx,
                                                                   uint32_t dStartIdx, uint32_t dDealSize,
                                                                   uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::LINEAR_BUFFER) {
        uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        while (copyFinishRowCnt < seqCnt) {
            uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
            uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
            uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCount > seqCnt) {
                copyRowCount = seqCnt - copyFinishRowCnt;
            }
            if (idInBlockTable < constInfo_.blockNum) { // vLLM block 0 is valid.
                uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                       remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                       stateIdx * coff_ * constInfo_.headDim + dStartIdx;
                DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                      dDealSize, coff_ * dDealSize,
                                      coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR);
            }

            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    } else {
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        uint64_t idInBlockTable = blockTableGm.GetValue(batchIdx);
        while (copyFinishRowCnt < seqCnt) {
            uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
            uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCount > seqCnt) {
                copyRowCount = seqCnt - copyFinishRowCnt;
            }
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                  dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR);

            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::SaveState(const LocalTensor<T> &srcLocal,
                                                           const GlobalTensor<T> &stateGm,
                                                           const GlobalTensor<int32_t> &blockTableGm,
                                                           const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                           uint32_t dDealSize, uint32_t stateIdx)
{
    uint64_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dDealSize;

    {
        // Persist only the uncompressed tail. LayerNorm is applied to the
        // current projection before pooling, so the cache remains raw.
        uint64_t totalEndSeqIdx = sliceInfo.bStartPos + sliceInfo.bSeqUsed;
        uint32_t tailLen = totalEndSeqIdx % cmpRatio_;
        if (tailLen == 0) {
            return;
        }
        uint64_t tailStartSeqIdx = totalEndSeqIdx - tailLen;
        if (endSeqIdx <= tailStartSeqIdx) {
            return;
        }
        if (startSeqIdx < tailStartSeqIdx) {
            srcBaseOffset += (tailStartSeqIdx - startSeqIdx) * coff_ * dDealSize;
            startSeqIdx = tailStartSeqIdx;
        }
    }

    if constexpr (COMP::cacheMode == CACHE_MODE::RING_BUFFER) {
        uint64_t compressSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.bSeqUsed, (uint64_t)cmpRatio_);
        uint32_t writeSeqStartIdx =
            compressSeqIdx > (coff_ - 1) * cmpRatio_ ? compressSeqIdx - (coff_ - 1) * cmpRatio_ : 0;
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, (uint64_t)writeSeqStartIdx) - startSeqIdx) * coff_ * dDealSize;
        startSeqIdx = max(startSeqIdx, (uint64_t)writeSeqStartIdx);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                          dStartIdx, dDealSize, stateIdx);
        srcBaseOffset += dDealSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx,
                      dDealSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVector<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal,
                                                                     uint32_t duplicateRowCount,
                                                                     uint32_t duplicateColCount,
                                                                     uint32_t singleRowCount)
{
    for (uint32_t offset = 0; offset < duplicateColCount; offset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curDuplicateColCount = min(duplicateColCount - offset, FP32_REPEAT_ELEMENT_NUM);
        if constexpr (IS_SCORE) {
            Duplicate(dstLocal[offset], SOFTMAX_MIN_NUM, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        } else {
            Duplicate(dstLocal[offset], FLOAT_ZERO, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        }
    }
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVector<COMP>::ReadState(const LocalTensor<T> &dstLocal,
                                                           const GlobalTensor<T> &stateGm,
                                                           const GlobalTensor<int32_t> &blockTableGm,
                                                           const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                           uint32_t dDealSize, uint32_t stateIdx)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    // 填充右边
    if (sliceInfo.headHolderSeqCnt > 0) {
        // 整个batch的第一块
        uint64_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * cmpRatio_ * coff_ * dDealSize;
        if constexpr (COMP::coff == KeyPool::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == KeyPool::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < cmpRatio_;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * cmpRatio_ * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], cmpRatio_, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < cmpRatio_ && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx = sliceInfo.bStartPos < cmpRatio_ ?
                                       0 :
                                       Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_) - cmpRatio_;
            uint32_t endSeqIdx = min(
                Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, (uint64_t)cmpRatio_) - cmpRatio_,
                sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * cmpRatio_ * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += cmpRatio_ * coff_ * dDealSize;
            }
            ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal, uint32_t tcDealSize,
                                                           uint32_t dDealSize)
{
    float minValue = SOFTMAX_MIN_VALUE;
    uint32_t ReduceSize = coff_ * cmpRatio_;
    FaVectorApi::SoftmaxDnVF<T>(scoreLocal, scoreLocal, dDealSize, ReduceSize, tcDealSize, minValue, dDealSize);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::RoundToHiddenDtype(const LocalTensor<T> &srcLocal,
                                                                    uint32_t elementCount)
{
    constexpr uint32_t maxRoundCount = BUFFER_SIZE_BYTE_16K / sizeof(HIDDEN_STATES_T);
    for (uint32_t offset = 0; offset < elementCount; offset += maxRoundCount) {
        uint32_t curCount = min(maxRoundCount, elementCount - offset);
        LocalTensor<HIDDEN_STATES_T> roundLocal = outputQue2.AllocTensor<HIDDEN_STATES_T>();
        Cast(roundLocal, srcLocal[offset], RoundMode::CAST_ROUND, curCount);
        PipeBarrier<PIPE_V>();
        Cast(srcLocal[offset], roundLocal, RoundMode::CAST_NONE, curCount);
        PipeBarrier<PIPE_V>();
        outputQue2.FreeTensor(roundLocal);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::AddApeToPooledScore(const LocalTensor<T> &scoreLocal,
                                                                     uint32_t poolCount, uint32_t dDealSize)
{
    uint32_t poolElementCount = cmpRatio_ * dDealSize;
    for (uint32_t pool = 0; pool < poolCount; pool++) {
        AddVF(scoreLocal[pool * poolElementCount], apeUb, cmpRatio_, dDealSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                  const LocalTensor<T> &scoreLocal,
                                                                  const LocalTensor<T> &dstLocal, uint32_t tcDealSize,
                                                                  uint32_t dDealSize)
{
    if (hasLayerNorm_ && cmpRatio_ == 128U) {
        PipeBarrier<PIPE_ALL>();
    }
    MulReduceSumbaseVF(kvLocal, scoreLocal, dstLocal, coff_, cmpRatio_, dDealSize, tcDealSize);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb,
                                                                        const Vec1SliceInfo &sliceInfo,
                                                                        uint32_t compressTcSize, uint32_t dStartIdx,
                                                                        uint32_t dDealSize)
{
    LocalTensor<HIDDEN_STATES_T> outputUb = outputQue2.AllocTensor<HIDDEN_STATES_T>();
    Cast(outputUb, comperssoredUb, RoundMode::CAST_ROUND, compressTcSize * dDealSize);
    outputQue2.EnQue(outputUb);
    outputQue2.DeQue<HIDDEN_STATES_T>();
    if constexpr (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::BSH) {
        uint32_t outputPoolCapacity = CeilDivT(GetSeqLength(sliceInfo.bIdx), cmpRatio_);
        uint32_t bIdx = sliceInfo.bIdx;
        uint32_t sIdx = sliceInfo.sIdx;
        uint64_t ubOffset = 0;
        while (compressTcSize > 0) {
            uint32_t bStartPos = GetStartPos(bIdx);
            uint32_t preScSize = (sIdx + bStartPos) / cmpRatio_;
            uint32_t totalScSize = (GetSeqUsed(bIdx) + bStartPos) / cmpRatio_;
            if (preScSize < totalScSize) {
                uint32_t curScSize = min(compressTcSize, totalScSize - preScSize);
                uint64_t outputPoolIdx =
                    static_cast<uint64_t>(bIdx) * outputPoolCapacity + preScSize - bStartPos / cmpRatio_;
                uint64_t outGmOffset = outputPoolIdx * constInfo_.headDim + dStartIdx;
                DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb[ubOffset], curScSize, dDealSize, dDealSize,
                                    constInfo_.headDim);
                compressTcSize -= curScSize;
                ubOffset += curScSize * dDealSize;
            }
            bIdx++;
            sIdx = 0;
        }
    } else {
        uint64_t outGmOffset = static_cast<uint64_t>(compressedCnt_) * constInfo_.headDim + dStartIdx;
        DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb, compressTcSize, dDealSize, dDealSize,
                            constInfo_.headDim);
    }
    outputQue2.FreeTensor(outputUb);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
    uint32_t dBaseSize, uint32_t needDealTcSize)
{
    KeyPoolVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = scoreMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(scoreUb, scoreDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(scoreUb);
    inputQue1.DeQue<T>();
    RoundToHiddenDtype(scoreUb, statisticInfo.dealSeqCnt * coff_ * dDealSize);
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, scoreCacheTcGm_, info,
                      overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(scoreUb);

    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0 &&
            (!loopInfo.isCoreRowFirst || !loopInfo.isCoreLoopFirst)) {
            PipeBarrier<PIPE_V>();
            AddSingleApeToScore(scoreLocal, apeUb, originSliceInfo, dDealSize);
        }
    }

    if (hasLayerNorm_) {
        uint64_t normalizedOffset =
            normalizedKvDbOffset_ +
            static_cast<uint64_t>(compressedCnt_ - normalizedPoolBase_) * cmpRatio_ * constInfo_.headDim + dStartIdx;
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyAlignGmToUb(kvLocal, normalizedKvGm_[normalizedOffset], statisticInfo.key_poolScCnt * cmpRatio_,
                            dDealSize, constInfo_.headDim, dDealSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        PipeBarrier<PIPE_ALL>();
    } else {
        GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
        LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
        FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
        inputQue1.EnQue(kvUb);
        inputQue1.DeQue<T>();
        RoundToHiddenDtype(kvUb, statisticInfo.dealSeqCnt * coff_ * dDealSize);
        overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
        overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
        while (!overLapSliceIterator.IsEnd()) {
            overLapSliceIterator.GetSlice();
            OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, kvCacheTcGm_, info,
                           overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
            overLapSliceIterator.IteratorSlice();
        }
        inputQue1.FreeTensor(kvUb);
    }

    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::GatherKvForNorm(const LocalTensor<T> &kvLocal, const Vec1RunInfo &info,
                                                                 const LoopInfo &loopInfo,
                                                                 const StatisticInfo &statisticInfo,
                                                                 const Vec1SliceInfo &originSliceInfo,
                                                                 uint32_t dStartIdx, uint32_t dDealSize,
                                                                 uint32_t needDealTcSize)
{
    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];

    KeyPoolVec1SliceIterator<COMP> overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        if (overLapSliceInfo.dealTcSize > 0) {
            ReadState<false>(kvLocal, stateCacheGm_, stateBlockTableGm_, overLapSliceInfo, dStartIdx, dDealSize, 0U);

            if constexpr (COMP::coff == COFF::DISABLE) {
                if (overLapSliceInfo.compressTcSize > 0) {
                    uint32_t copyRowCount =
                        overLapSliceInfo.compressTcSize * cmpRatio_ - overLapSliceInfo.headHolderSeqCnt;
                    uint64_t dstUbOffset = (static_cast<uint64_t>(overLapSliceInfo.key_pooledScCnt) * cmpRatio_ +
                                            overLapSliceInfo.headHolderSeqCnt) *
                                           dDealSize;
                    uint64_t srcGmOffset =
                        (static_cast<uint64_t>(originSliceInfo.dealedSeqCnt) + overLapSliceInfo.dealedSeqCnt) *
                            constInfo_.headDim +
                        dStartIdx;
                    uint32_t maxRowsPerChunk =
                        max(1U, static_cast<uint32_t>(BUFFER_SIZE_BYTE_32K / (dDealSize * sizeof(T))));
                    for (uint32_t rowOffset = 0; rowOffset < copyRowCount; rowOffset += maxRowsPerChunk) {
                        uint32_t curRows = min(maxRowsPerChunk, copyRowCount - rowOffset);
                        LocalTensor<T> inputChunk = inputQue1.AllocTensor<T>();
                        GlobalTensor<T> srcChunk =
                            kvDBMm1ResGm[srcGmOffset + static_cast<uint64_t>(rowOffset) * constInfo_.headDim];
                        if (constInfo_.kBaseNum == 1) {
                            DataCopyAlignGmToUb(inputChunk, srcChunk, curRows, dDealSize, constInfo_.headDim,
                                                dDealSize);
                        } else {
                            AddMultiDataToUb(inputChunk, srcChunk, curRows, dDealSize, constInfo_.headDim, dDealSize,
                                             constInfo_.kBaseNum, constInfo_.mm1KvResSize);
                        }
                        inputQue1.EnQue(inputChunk);
                        inputQue1.DeQue<T>();
                        RoundToHiddenDtype(inputChunk, curRows * dDealSize);
                        DataCopyAlignUbToUb(kvLocal[dstUbOffset + static_cast<uint64_t>(rowOffset) * dDealSize],
                                            inputChunk, curRows, dDealSize, dDealSize, dDealSize);
                        inputQue1.FreeTensor(inputChunk);
                    }
                }
            } else {
                LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
                FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
                inputQue1.EnQue(kvUb);
                inputQue1.DeQue<T>();
                RoundToHiddenDtype(kvUb, statisticInfo.dealSeqCnt * dDealSize);
                OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, kvCacheTcGm_, info,
                               overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
                inputQue1.FreeTensor(kvUb);
            }
        }
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::PrepareNormalizedKv(const Vec1RunInfo &info,
                                                                     const Vec1SplitInfo &splitInfo,
                                                                     const LoopInfo &loopInfo)
{
    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    normalizedKvDbOffset_ =
        (static_cast<uint64_t>(info.c1v1DbIdx) * constInfo_.usedCoreNum * 2U + GetBlockIdx()) * constInfo_.mm1KvResSize;
    normalizedPoolBase_ = compressedCnt_;

    KeyPoolVec1SliceIterator<COMP> sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
    for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize;) {
        uint32_t maxPoolCount = BUFFER_SIZE_BYTE_64K / (cmpRatio_ * constInfo_.headDim * sizeof(T));
        maxPoolCount = max(maxPoolCount, 1U);
        uint32_t actDealTcSize = min(maxPoolCount, splitInfo.dealTcSize - tcIdx);
        sliceIterator.SetNeedDealTcSize(actDealTcSize);
        Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
        uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
        StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
        if (statisticInfo.key_poolScCnt > 0) {
            LocalTensor<T> kvLocal = tmpBuff2.Get<T>();
            GatherKvForNorm(kvLocal, info, loopInfo, statisticInfo, originSliceInfo, 0U, constInfo_.headDim,
                            needDealTcSize);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            KeyPoolVec1SliceIterator<COMP> normalizedSliceIterator(tools_);
            normalizedSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
            normalizedSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
            normalizedSliceIterator.SetNeedDealTcSize(needDealTcSize);
            while (!normalizedSliceIterator.IsEnd()) {
                Vec1SliceInfo normalizedSlice = normalizedSliceIterator.GetSlice();
                if (normalizedSlice.compressTcSize > 0) {
                    uint32_t historicalRows = normalizedSlice.headHolderSeqCnt;
                    uint32_t currentRows = normalizedSlice.compressTcSize * cmpRatio_ - historicalRows;
                    uint64_t currentOffset =
                        static_cast<uint64_t>(normalizedSlice.key_pooledScCnt * cmpRatio_ + historicalRows) *
                        constInfo_.headDim;
                    LocalTensor<T> normScratch = inputQue1.AllocTensor<T>();
                    KeyPoolLayerNormRowsInplace(kvLocal[currentOffset], normScratch, normWeightGm_, normBiasGm_,
                                                constInfo_.normEps, currentRows, constInfo_.headDim,
                                                constInfo_.headDim);
                    inputQue1.FreeTensor(normScratch);
                    PipeBarrier<PIPE_ALL>();
                    ApplyKeyPoolRotaryPlaceholder(kvLocal[currentOffset], currentRows * constInfo_.headDim);
                }
                normalizedSliceIterator.IteratorSlice();
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            uint64_t normalizedOffset =
                normalizedKvDbOffset_ +
                static_cast<uint64_t>(compressedCnt_ - normalizedPoolBase_) * cmpRatio_ * constInfo_.headDim;
            DataCopyAlignUbToGm(normalizedKvGm_[normalizedOffset], kvLocal, statisticInfo.key_poolScCnt * cmpRatio_,
                                constInfo_.headDim, constInfo_.headDim, constInfo_.headDim);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            compressedCnt_ += statisticInfo.key_poolScCnt;
        }

        tcIdx += actDealTcSize;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::SaveNormalizedTail(const Vec1RunInfo &info)
{
    if (!hasLayerNorm_ || info.dealTcNum == 0) {
        return;
    }
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    if (splitInfo.dealTcSize == 0) {
        return;
    }
    // The caller's cross-core read barrier protects aliased ring pages.
    // Only the final incomplete group writes state, after every history read.
    KeyPoolVec1SliceIterator<COMP> tailIterator(tools_);
    tailIterator.SetMaxBatchSize(constInfo_.batchSize);
    tailIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
    tailIterator.SetNeedDealTcSize(splitInfo.dealTcSize);
    while (!tailIterator.IsEnd()) {
        Vec1SliceInfo tailSlice = tailIterator.GetSlice();
        uint32_t tailRows = (tailSlice.bStartPos + tailSlice.bSeqUsed) % cmpRatio_;
        tailRows = min(tailRows, tailSlice.validSeqCnt);
        if (tailRows > 0 && tailSlice.sIdx + tailSlice.validSeqCnt == tailSlice.bSeqUsed) {
            uint64_t tailOffset = static_cast<uint64_t>(tailSlice.dealedSeqCnt +
                                                       tailSlice.validSeqCnt - tailRows) * constInfo_.headDim;
            Vec1SliceInfo tailCacheSlice = tailSlice;
            tailCacheSlice.sIdx += tailSlice.validSeqCnt - tailRows;
            tailCacheSlice.validSeqCnt = tailRows;
            tailCacheSlice.dealedSeqCnt = 0U;
            for (uint32_t stateIdx = 0; stateIdx < STATE_INTERLEAVE_FACTOR; ++stateIdx) {
                GlobalTensor<T> sourceGm = stateIdx == 0 ? kvMm1ResGm_ : scoreMm1ResGm_;
                LocalTensor<T> tail = tmpBuff2.Get<T>();
                if (stateIdx == 0 && constInfo_.kBaseNum > 1) {
                    // K is split across Cube partials on this architecture.
                    AddMultiDataToUb(tail, sourceGm[info.c1v1DbIdx * constInfo_.dbSize + tailOffset],
                                     tailRows, constInfo_.headDim, constInfo_.headDim, constInfo_.headDim,
                                     constInfo_.kBaseNum, constInfo_.mm1KvResSize);
                } else {
                    DataCopyAlignGmToUb(tail, sourceGm[info.c1v1DbIdx * constInfo_.dbSize + tailOffset],
                                        tailRows, constInfo_.headDim, constInfo_.headDim, constInfo_.headDim);
                }
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                RoundToHiddenDtype(tail, tailRows * constInfo_.headDim);
                if (stateIdx == 0) {
                    LocalTensor<T> normScratch = inputQue1.AllocTensor<T>();
                    KeyPoolLayerNormRowsInplace(tail, normScratch, normWeightGm_, normBiasGm_, constInfo_.normEps,
                                                tailRows, constInfo_.headDim, constInfo_.headDim);
                    inputQue1.FreeTensor(normScratch);
                    ApplyKeyPoolRotaryPlaceholder(tail, tailRows * constInfo_.headDim);
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                SaveState(tail, stateCacheGm_, stateBlockTableGm_, tailCacheSlice, 0U, constInfo_.headDim, stateIdx);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        tailIterator.IteratorSlice();
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::DealVec1BaseBlock(const Vec1RunInfo &info,
                                                                   KeyPoolVec1SliceIterator<COMP> &sliceIterator,
                                                                   const LoopInfo &loopInfo, uint32_t dStartIdx,
                                                                   uint32_t dDealSize, uint32_t dBaseSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = tmpBuff1.Get<T>();
    LocalTensor<T> kvLocal = tmpBuff2.Get<T>();

    OverLapScoreKv(scoreLocal, kvLocal, info, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dDealSize, dBaseSize,
                   needDealTcSize);

    if (statisticInfo.key_poolScCnt > 0) {
        AddApeToPooledScore(scoreLocal, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        SoftmaxDN(scoreLocal, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        RoundToHiddenDtype(scoreLocal, statisticInfo.key_poolScCnt * cmpRatio_ * dDealSize);
        LocalTensor<T> comperssoredUb = scoreLocal;
        bool independentReductionOutput = false;
        if (hasLayerNorm_ && cmpRatio_ == 128U) {
            comperssoredUb = outputQue1.AllocTensor<T>();
            independentReductionOutput = true;
        }
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.key_poolScCnt, dStartIdx, dDealSize);
        if (independentReductionOutput) {
            outputQue1.FreeTensor(comperssoredUb);
        }
    }
    compressedCnt_ += statisticInfo.key_poolScCnt;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    if (hasLayerNorm_) {
        splitInfo.dBaseSize = constInfo_.headDim;
    } else {
        splitInfo.dBaseSize =
            constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, info.dealTcNum)));
    }
    if (constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(HIDDEN_STATES_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<HIDDEN_STATES_T>());
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), info.dealTcNum);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t dealTcNum = info.dealTcNum;

    if (blockIdx < groupSize * (dealTcNum % groupNum)) {
        splitInfo.dealTcSize = dealTcNum / groupNum + 1;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize);
    } else if (blockIdx < groupSize * groupNum) {
        splitInfo.dealTcSize = dealTcNum / groupNum;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize) + dealTcNum % groupNum;
    } else {
        splitInfo.dealTcSize = 0;
        splitInfo.preDealTcSize = dealTcNum;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    KeyPoolVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.Reset(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();

    // 处理前序任务量，更新起始索引
    if (splitInfo.preDealTcSize > 0) {
        sliceIterator.SetNeedDealTcSize(splitInfo.preDealTcSize);
        StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
        splitInfo.curCompressedCnt = statisticInfo.key_poolScCnt;
        splitInfo.dealSeqStartIdx = sliceInfo.dealedSeqCnt;
        splitInfo.curBStart = sliceInfo.bIdx;
        splitInfo.curSStart = sliceInfo.sIdx;
    } else {
        splitInfo.curCompressedCnt = 0;
        splitInfo.dealSeqStartIdx = 0;
        splitInfo.curBStart = info.bStart;
        splitInfo.curSStart = info.sStart;
    }

    // 处理当前核实际要跑的任务量
    sliceIterator.SetNeedDealTcSize(info.dealTcNum - splitInfo.preDealTcSize);
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    splitInfo.totalCompressedCnt = splitInfo.curCompressedCnt + statisticInfo.key_poolScCnt;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
{
    // 计算headDim和Tc方向切分大小
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (cmpRatio_ * coff_ * sizeof(T));

    // 切块逻辑
    if (maxDealColNum < splitInfo.dBaseSize) {
        splitInfo.tcSplitSize = 1;
        splitInfo.dLoopCount = CeilDivT(splitInfo.dBaseSize, maxDealColNum);
        splitInfo.dSplitSize = splitInfo.dBaseSize / splitInfo.dLoopCount;
    } else {
        splitInfo.dSplitSize = splitInfo.dBaseSize;
        splitInfo.dLoopCount = splitInfo.dBaseSize / splitInfo.dSplitSize; // 此处常等于1，保留原逻辑
        splitInfo.tcSplitSize = maxDealColNum / splitInfo.dBaseSize;
    }
}

template <typename COMP>
__aicore__ inline Vec1SplitInfo KeyPoolBlockVector<COMP>::SplitCoreV1(const Vec1RunInfo &info)
{
    Vec1SplitInfo splitInfo;

    // 1. 计算基础分组和分片大小
    CalcGroupInfo(info, splitInfo);

    // 2. 根据当前的 BlockIdx 计算任务分配（负载均衡）
    CalcTaskDistribution(info, splitInfo);

    // 3. 刷新迭代器并获取当前核的起始位置状态
    UpdateIteratorState(info, splitInfo);

    if (splitInfo.dealTcSize == 0) {
        return splitInfo;
    }

    // 4. 计算具体在内存中的切块（Tiling）逻辑
    CalcTilingStrategy(splitInfo);

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVector<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    if (info.dealTcNum == 0) {
        return;
    }
    uint32_t preCompressedCnt = compressedCnt_;
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    // 计算当前VecCore的任务量
    if (splitInfo.dealTcSize == 0) {
        compressedCnt_ += splitInfo.totalCompressedCnt;
        return;
    }

    LoopInfo loopInfo;
    loopInfo.groupSize = splitInfo.vec1GroupSize;
    loopInfo.groupNum = splitInfo.vec1GroupNum;
    loopInfo.coreRowIdx = GetBlockIdx() / splitInfo.vec1GroupSize;
    loopInfo.coreColIdx = GetBlockIdx() % splitInfo.vec1GroupSize;
    loopInfo.isCoreRowLast = loopInfo.coreRowIdx == splitInfo.vec1GroupNum - 1;
    loopInfo.isCoreRowFirst = loopInfo.coreRowIdx == 0;

    KeyPoolVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    if (hasLayerNorm_) {
        compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
        PrepareNormalizedKv(info, splitInfo, loopInfo);
        compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
    }
    // 切块循环
    uint64_t baseOffset = loopInfo.coreColIdx * splitInfo.dBaseSize;
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;

        CopyInApe(apeUb, dBaseOffset, splitInfo.dSplitSize);

        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
        for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize; tcIdx += splitInfo.tcSplitSize) {
            uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcSize - tcIdx);

            loopInfo.isCoreLoopFirst = tcIdx == 0;
            loopInfo.isCoreLoopLast = tcIdx + splitInfo.tcSplitSize >= splitInfo.dealTcSize;
            // 处理单个切块
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            sliceIterator.SetDealedTcCnt(0U);
            DealVec1BaseBlock(info, sliceIterator, loopInfo, dBaseOffset, splitInfo.dSplitSize, splitInfo.dBaseSize);
        }
    }
    compressedCnt_ = preCompressedCnt + splitInfo.totalCompressedCnt;
}

} // namespace KeyPool
#endif // KEY_POOL_BLOCK_VECTOR_H
