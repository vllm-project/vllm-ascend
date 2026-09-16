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
 * \file key_pool_block_vec_perf.h
 * \brief
 */

#ifndef KEY_POOL_BLOCK_VEC_PERF_H
#define KEY_POOL_BLOCK_VEC_PERF_H

#include "key_pool_comm_arch22.h"
#include "key_pool_tools.h"
#include "key_pool_vector_comm.h"
#include "../key_pool_layer_norm.h"
#include "key_pool_soft_max.h"

using namespace AscendC;

namespace KeyPool {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP>
class KeyPoolBlockVectorPerf {
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

    __aicore__ inline KeyPoolBlockVectorPerf(){};
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
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info, KeyPoolVec1SliceIterator<COMP> &sliceIterator,
                                             const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline void CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                         const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    __aicore__ inline void AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O> dstLocal, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O> dstLocal, const GlobalTensor<O> srcGm,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyWithOutputQue(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void PadAlign(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
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
    __aicore__ inline void SaveToWorkSpace(const LocalTensor<T> srcLocal, const GlobalTensor<T> &cacheTcGm,
                                           const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                           uint32_t dDealSize);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T> dstLocal, const GlobalTensor<T> &cacheTcGm,
                                             const GlobalTensor<T> &srcGm, const LocalTensor<T> srcLocal,
                                             const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                     uint32_t dDealSize);
    __aicore__ inline void RoundToHiddenDtype(const LocalTensor<T> &srcLocal, uint32_t elementCount);
    __aicore__ inline void AddApeToPooledScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeLocal,
                                               uint32_t poolCount, uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb,
                                            uint32_t tcDealSize, uint32_t dDealSize);
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
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t compressedCnt_ = 0;
    uint32_t v1SplitSize_ = 0;
    uint32_t v1ScLoopTimes_ = 0;
    uint32_t v1DLoopTimes_ = 0;
    uint32_t dealTcNum_ = 0;
    bool apeIsLoad_ = false;
    bool isExistSeqUsed = false;
    bool isExistStartPos = false;
    // vec2
    uint32_t mmResColSize_ = 128;
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
    LocalTensor<T> mm1ResTensor;
    LocalTensor<T> leftStateTensor;
    LocalTensor<T> rightStateTensor;
    LocalTensor<T> apeUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> apeBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    bool hasLayerNorm_ = false;
    uint64_t normalizedKvDbOffset_ = 0;
    uint32_t normalizedPoolBase_ = 0;
};

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::InitParams(const ConstInfo &constInfo,
                                                                const KeyPoolTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk,
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
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if constexpr (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
    }
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_32K);
    apeUb = apeBuf.Get<T>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::FreeEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm,
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
__aicore__ inline uint32_t KeyPoolBlockVectorPerf<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed) {
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
__aicore__ inline uint32_t KeyPoolBlockVectorPerf<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVectorPerf<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx,
                                                               uint32_t dDealSize)
{
    LocalTensor<T> apeUbTmp = inputQue1.AllocTensor<T>();

    uint32_t copyRowCount = coff_ * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;
    DataCopyAlignGmToUb(apeUbTmp, apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    inputQue1.EnQue(apeUbTmp);
    inputQue1.DeQue<T>();
    DataCopy(apeUb, apeUbTmp, coff_ * dDealSize * constInfo_.cmpRatio);
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(apeUbTmp);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
                                                                   const LocalTensor<T> &apeUb,
                                                                   const Vec1SliceInfo &sliceInfo, uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleRowElemNum;
        uint32_t rCnt = tcDealSize == 1 ? sliceInfo.validSeqCnt * singleRowElemNum :
                                          (constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt) * singleRowElemNum;
        Add(scoreLocal[scoreOffset], scoreLocal[scoreOffset], apeUb[apeOffset], rCnt);
        scoreOffset += rCnt;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint64_t apeOffset = 0;
        uint32_t rCnt = (constInfo_.cmpRatio - sliceInfo.tailHolderSeqCnt) * singleRowElemNum;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * constInfo_.cmpRatio * singleRowElemNum;
        Add(scoreLocal[tailScoreOffset], scoreLocal[tailScoreOffset], apeUb[apeOffset], rCnt);
    }
    if (tcDealSize == 0) {
        return;
    }
    uint32_t rCnt = constInfo_.cmpRatio * singleRowElemNum;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        Add(scoreLocal[scoreOffset + r * rCnt], scoreLocal[scoreOffset + r * rCnt], apeUb, rCnt);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::AddSingleApeToScore(const LocalTensor<T> &scoreLocal,
                                                                         const LocalTensor<T> &apeUb,
                                                                         const Vec1SliceInfo &sliceInfo,
                                                                         uint32_t dDealSize)
{
    uint32_t SingleRowElemNum = dDealSize * coff_;
    uint32_t dealRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t scoreOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    uint64_t apeOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    for (uint32_t dOffset = 0; dOffset < dDealSize; dOffset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curAddColCount = min(dDealSize - dOffset, FP32_REPEAT_ELEMENT_NUM);
        Add(scoreLocal[scoreOffset + dOffset], scoreLocal[scoreOffset + dOffset], apeUb[apeOffset + dOffset],
            curAddColCount, dealRowCount,
            {1, 1, 1, static_cast<uint8_t>(SingleRowElemNum / BlockElementNum<T>()),
             static_cast<uint8_t>(SingleRowElemNum / BlockElementNum<T>()),
             static_cast<uint8_t>(SingleRowElemNum / BlockElementNum<T>())});
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> dstLocal,
                                                                         const LocalTensor<O> srcLocal,
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> dstLocal,
                                                                         const GlobalTensor<O> srcGm,
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> dstGm,
                                                                         const LocalTensor<O> srcLocal,
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DataCopyWithOutputQue(const GlobalTensor<O> dstGm,
                                                                           const LocalTensor<O> srcLocal,
                                                                           uint32_t copyRowCount, uint32_t copyColCount,
                                                                           uint32_t srcSingleRowCount,
                                                                           uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    uint32_t singleCopyRowCount = BUFFER_SIZE_BYTE_16K / (copyColCount * sizeof(O));
    for (uint32_t rowCount = 0; rowCount < copyRowCount; rowCount += singleCopyRowCount) {
        uint64_t srcOffset = rowCount * srcSingleRowCount;
        uint64_t dstOffset = rowCount * dstSingleRowCount;
        uint32_t curCopyRowCount = min(singleCopyRowCount, copyRowCount - rowCount);

        LocalTensor<O> outputUb = outputQue1.AllocTensor<O>();

        DataCopyAlignUbToUb(outputUb, srcLocal[srcOffset], curCopyRowCount, copyColCount, srcSingleRowCount,
                            copyColCount);
        PipeBarrier<PIPE_V>();

        outputQue1.EnQue(outputUb);
        outputQue1.DeQue<O>();

        DataCopyAlignUbToGm(dstGm[dstOffset], outputUb, curCopyRowCount, copyColCount, copyColCount, dstSingleRowCount);

        outputQue1.FreeTensor(outputUb);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::PadAlign(const LocalTensor<T> dstLocal,
                                                              const LocalTensor<T> srcLocal,
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
    uint32_t copyRowCount = sliceInfo.compressTcSize * constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = srcSingleRowElemNum; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum;

    uint64_t dstUbOffset = sliceInfo.key_pooledScCnt * constInfo_.cmpRatio * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + constInfo_.cmpRatio) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                            copyRowCount - min(copyRowCount, constInfo_.cmpRatio), copyColCount, srcSingleRowCount,
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::OverLap(
    const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if (!hasLayerNorm_) {
        SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));
    }

    event_t eventId_V_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                        static_cast<uint32_t>(IS_SCORE));

    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t nextC1V1DbIdx = (info.c1v1DbIdx + 1) % constInfo_.dbWorkspaceRatio;
        GlobalTensor<T> nextCacheTcGm = cacheTcGm[nextC1V1DbIdx * constInfo_.cmpRatio * constInfo_.headDim];
        SaveToWorkSpace(srcLocal, nextCacheTcGm, sliceInfo, loopInfo, dStartIdx, dDealSize);
    }
    if (sliceInfo.compressTcSize > 0) {
        PadAlign(dstLocal, srcLocal, sliceInfo, dStartIdx, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            GlobalTensor<T> curCacheTcGm = cacheTcGm[info.c1v1DbIdx * constInfo_.cmpRatio * constInfo_.headDim];
            LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx, globalSeqIdx,
                              dDealSize);
        }
    }
    event_t eventId_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal,
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
    DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::SaveToWorkSpace(const LocalTensor<T> srcLocal,
                                                                     const GlobalTensor<T> &cacheTcGm,
                                                                     const Vec1SliceInfo &sliceInfo,
                                                                     const LoopInfo &loopInfo, uint32_t dStartIdx,
                                                                     uint32_t dDealSize)
{
    uint64_t curSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt;
    uint64_t totalSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.bSeqUsed;
    if (!loopInfo.isCoreRowLast || !loopInfo.isCoreLoopLast || !sliceInfo.isLast || totalSeqLen < constInfo_.cmpRatio ||
        curSeqLen > Trunc(totalSeqLen, (uint64_t)constInfo_.cmpRatio) - constInfo_.cmpRatio) {
        return;
    }
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset =
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, constInfo_.cmpRatio)) *
        srcSingleRowElemNum;
    DataCopyWithOutputQue(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                          curSeqLen - max(curSeqLen - constInfo_.cmpRatio, sliceInfo.bStartPos), dDealSize,
                          coff_ * dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::LoadFromWorkSpace(
    const LocalTensor<T> dstLocal, const GlobalTensor<T> &cacheTcGm, const GlobalTensor<T> &srcGm,
    const LocalTensor<T> srcLocal, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.key_pooledScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    if (loopInfo.isCoreRowFirst && loopInfo.isCoreLoopFirst && sliceInfo.isFirst) { // 从cacheGm获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset = dStartIdx;
        DataCopyAlignGmToUb(dstLocal[dstLocalOffset], cacheTcGm[srcLocalOffset], copyRowCount, dDealSize,
                            constInfo_.headDim, coff_ * dDealSize);
    } else if (sliceInfo.isFirst) { // 从存放MatMul结果的WorkSpace中获取
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset =
            (globalSeqIdx + sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum + dStartIdx;
        DataCopyAlignGmToUb(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                            coff_ * constInfo_.headDim, coff_ * dDealSize);
    } else { // 从UB中获取
        uint32_t srcSingleRowElemNum = dDealSize * coff_;
        uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
        DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                            coff_ * dDealSize, coff_ * dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
                                                                        const GlobalTensor<T> &state,
                                                                        const GlobalTensor<int32_t> &blockTableGm,
                                                                        uint32_t batchIdx, uint32_t startSeqIdx,
                                                                        uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                        uint32_t dDealSize, uint32_t stateIdx)
{
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

        DataCopyAlignGmToUb(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount, dDealSize,
                            STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim, coff_ * dDealSize);
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                       const GlobalTensor<int32_t> &blockTableGm,
                                                                       const LocalTensor<T> &input, uint32_t batchIdx,
                                                                       uint32_t startSeqIdx, uint32_t endSeqIdx,
                                                                       uint32_t dStartIdx, uint32_t dDealSize,
                                                                       uint32_t stateIdx)
{
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
                                  dDealSize, coff_ * dDealSize, STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim);
        }

        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::SaveState(const LocalTensor<T> &srcLocal,
                                                               const GlobalTensor<T> &stateGm,
                                                               const GlobalTensor<int32_t> &blockTableGm,
                                                               const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                               uint32_t dDealSize, uint32_t stateIdx)
{
    uint64_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dDealSize;

    {
        // Keep the cache raw and persist only the uncompressed tail.
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal,
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::ReadState(const LocalTensor<T> &dstLocal,
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
        uint64_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)constInfo_.cmpRatio);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
        if constexpr (COMP::coff == KeyPool::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == KeyPool::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < constInfo_.cmpRatio;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], constInfo_.cmpRatio, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < constInfo_.cmpRatio && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx =
                sliceInfo.bStartPos < constInfo_.cmpRatio ?
                    0 :
                    Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)constInfo_.cmpRatio) - constInfo_.cmpRatio;
            uint32_t endSeqIdx =
                min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, (uint64_t)constInfo_.cmpRatio) -
                        constInfo_.cmpRatio,
                    sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.key_pooledScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += constInfo_.cmpRatio * coff_ * dDealSize;
            }
            ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal,
                                                               const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                                               uint32_t dDealSize)
{
    float minValue = SOFTMAX_MIN_VALUE;
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::RoundToHiddenDtype(const LocalTensor<T> &srcLocal,
                                                                        uint32_t elementCount)
{
    constexpr uint32_t maxRoundCount = BUFFER_SIZE_BYTE_16K / sizeof(HIDDEN_STATES_T);
    for (uint32_t offset = 0; offset < elementCount; offset += maxRoundCount) {
        uint32_t curCount = min(maxRoundCount, elementCount - offset);
        LocalTensor<HIDDEN_STATES_T> roundLocal = outputQue1.AllocTensor<HIDDEN_STATES_T>();
        Cast(roundLocal, srcLocal[offset], RoundMode::CAST_ROUND, curCount);
        PipeBarrier<PIPE_V>();
        Cast(srcLocal[offset], roundLocal, RoundMode::CAST_NONE, curCount);
        PipeBarrier<PIPE_V>();
        outputQue1.FreeTensor(roundLocal);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::AddApeToPooledScore(const LocalTensor<T> &scoreLocal,
                                                                         const LocalTensor<T> &apeLocal,
                                                                         uint32_t poolCount, uint32_t dDealSize)
{
    uint32_t poolElementCount = cmpRatio_ * dDealSize;
    for (uint32_t pool = 0; pool < poolCount; pool++) {
        Add(scoreLocal[pool * poolElementCount], scoreLocal[pool * poolElementCount], apeLocal, poolElementCount);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                      const LocalTensor<T> &scoreLocal,
                                                                      const LocalTensor<T> &dstLocal,
                                                                      const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                                                      uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    Mul(kvLocal, kvLocal, scoreLocal, tcDealSize * rCnt);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSum(dstLocal[r * dDealSize], kvLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb,
                                                                            const Vec1SliceInfo &sliceInfo,
                                                                            uint32_t compressTcSize, uint32_t dStartIdx,
                                                                            uint32_t dDealSize)
{
    LocalTensor<HIDDEN_STATES_T> outputUb = outputQue1.AllocTensor<HIDDEN_STATES_T>();
    Cast(outputUb, comperssoredUb, RoundMode::CAST_ROUND, compressTcSize * dDealSize);
    outputQue1.EnQue(outputUb);
    outputQue1.DeQue<HIDDEN_STATES_T>();
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
    PipeBarrier<PIPE_MTE3>();
    outputQue1.FreeTensor(outputUb);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::OverLapScoreKv(
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::GatherKvForNorm(const LocalTensor<T> &kvLocal,
                                                                     const Vec1RunInfo &info, const LoopInfo &loopInfo,
                                                                     const StatisticInfo &statisticInfo,
                                                                     const Vec1SliceInfo &originSliceInfo,
                                                                     uint32_t dStartIdx, uint32_t dDealSize,
                                                                     uint32_t needDealTcSize)
{
    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(kvUb);
    inputQue1.DeQue<T>();
    RoundToHiddenDtype(kvUb, statisticInfo.dealSeqCnt * dDealSize);

    KeyPoolVec1SliceIterator<COMP> overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, kvCacheTcGm_, info,
                       overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(kvUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::PrepareNormalizedKv(const Vec1RunInfo &info,
                                                                         const Vec1SplitInfo &splitInfo,
                                                                         const LoopInfo &loopInfo)
{
    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    normalizedKvDbOffset_ =
        (static_cast<uint64_t>(info.c1v1DbIdx) * constInfo_.usedCoreNum * 2U + GetBlockIdx()) * constInfo_.vec1ResSize;
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::SaveNormalizedTail(const Vec1RunInfo &info)
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
                DataCopyAlignGmToUb(tail, sourceGm[info.c1v1DbIdx * constInfo_.dbSize + tailOffset],
                                    tailRows, constInfo_.headDim, constInfo_.headDim, constInfo_.headDim);
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::DealVec1BaseBlock(const Vec1RunInfo &info,
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
        LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
        AddApeToPooledScore(scoreLocal, apeUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        SoftmaxDN(scoreLocal, tmpUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        RoundToHiddenDtype(scoreLocal, statisticInfo.key_poolScCnt * cmpRatio_ * dDealSize);
        LocalTensor<T> comperssoredUb = scoreLocal;
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, tmpUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.key_poolScCnt, dStartIdx, dDealSize);
    }
    compressedCnt_ += statisticInfo.key_poolScCnt;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    if (hasLayerNorm_) {
        splitInfo.dBaseSize = constInfo_.headDim;
    } else {
        splitInfo.dBaseSize =
            constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, info.dealTcNum)));
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(HIDDEN_STATES_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<HIDDEN_STATES_T>());
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), info.dealTcNum);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::CalcTaskDistribution(const Vec1RunInfo &info,
                                                                          Vec1SplitInfo &splitInfo)
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::UpdateIteratorState(const Vec1RunInfo &info,
                                                                         Vec1SplitInfo &splitInfo)
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
{
    // 计算headDim和Tc方向切分大小
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (constInfo_.cmpRatio * coff_ * sizeof(T));

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
__aicore__ inline Vec1SplitInfo KeyPoolBlockVectorPerf<COMP>::SplitCoreV1(const Vec1RunInfo &info)
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
__aicore__ inline void KeyPoolBlockVectorPerf<COMP>::ComputeVec1(const Vec1RunInfo &info)
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
#endif // KEY_POOL_BLOCK_VECTOR_PREF_H
