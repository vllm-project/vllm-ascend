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
 * \file key_pool_block_vec_full_load.h
 * \brief
 */

#ifndef KEY_POOL_BLOCK_VEC_FULL_LOAD_H
#define KEY_POOL_BLOCK_VEC_FULL_LOAD_H

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
class KeyPoolBlockVectorFullLoad {
public:
    static constexpr bool HIDDEN_STATES_DTYPE = COMP::hiddenStatesDtype == HIDDEN_STATES_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float SOFTMAX_MIN_NUM = -2e38;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using HIDDEN_STATES_T = typename AscendC::Conditional<HIDDEN_STATES_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline KeyPoolBlockVectorFullLoad(){};
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
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm);
    __aicore__ inline void PrepareForComputeVec();
    __aicore__ inline void ComputeVec1();

protected:
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
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
    __aicore__ inline void CopyInApe(uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_FULLLOAD>
    __aicore__ inline void AddApe(const LocalTensor<T> &scoreLocal, uint32_t dealRowCount, uint32_t dealColCount,
                                  uint32_t scoreSingleRowCount, uint32_t apeSingleRowCount, uint64_t scoreOffset,
                                  uint64_t apeOffset);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const Vec1SliceInfo &sliceInfo,
                                         uint32_t dDealSize, uint32_t dBaseSize, uint32_t dStartIdx,
                                         bool isApeFullLoad);
    __aicore__ inline void FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                             uint32_t preDealSeqCnt, uint32_t dealSeqCnt, uint32_t dStartIdx,
                                             uint32_t dDealSize);

    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                             const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                             uint32_t endSeqIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                             uint32_t stateIdx);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                              const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx,
                                              uint64_t startSeqIdx, uint64_t endSeqIdx, uint32_t dStartIdx,
                                              uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void ReadState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void PadAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dBaseOffset, uint32_t dDealSize,
                                    uint32_t dBaseSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                   const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                   const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                   uint32_t dBaseOffset, uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t dBaseSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const LoopInfo &loopInfo, const StatisticInfo &statisticInfo,
                                          const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx,
                                          uint32_t dBaseOffset, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t dealSeqStartIdx, uint32_t needDealTcSize);
    __aicore__ inline void SaveToWorkSpace(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &cacheTcGm,
                                           const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                           uint32_t dDealSize);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm,
                                             const GlobalTensor<T> &srcGm, const LocalTensor<T> &srcLocal,
                                             const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                     uint32_t dDealSize);
    __aicore__ inline void RoundToHiddenDtype(const LocalTensor<T> &srcLocal, uint32_t elementCount);
    __aicore__ inline void AddApeToPooledScore(const LocalTensor<T> &scoreLocal, uint32_t poolCount,
                                               uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb,
                                            uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb, const Vec1SliceInfo &sliceInfo,
                                                  uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void DealVec1BaseBlock(KeyPoolVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t dBaseOffset, uint32_t dDealSize,
                                             uint32_t dBaseSize, uint32_t dealSeqStartIdx);
    __aicore__ inline void CalcGroupInfo(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1();
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t compressedCnt_ = 0;
    uint32_t kvStateIdx_ = 0;
    uint32_t scoreStateIdx_ = 1;
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    Vec1SplitInfo vec1SplitInfo_;
    LoopInfo loopInfo_;
    KeyPoolTools<COMP> tools_;
    ConstInfo constInfo_ = {};
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
    LocalTensor<T> scoreUb;
    LocalTensor<T> kvUb;
    LocalTensor<T> apeUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TQue<QuePosition::VECIN, 1> inputQueApe;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    bool hasLayerNorm_ = false;
};

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::InitParams(const ConstInfo &constInfo,
                                                                    const KeyPoolTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk,
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQueApe, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(tmpBuf1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuf2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_16K);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::FreeEventID()
{}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm,
                                                                              GlobalTensor<T> scoreMm1ResGm,
                                                                              GlobalTensor<T> kvCacheTcGm,
                                                                              GlobalTensor<T> scoreCacheTcGm)
{
    this->kvMm1ResGm_ = kvMm1ResGm;
    this->scoreMm1ResGm_ = scoreMm1ResGm;
    this->kvCacheTcGm_ = kvCacheTcGm;
    this->scoreCacheTcGm_ = scoreCacheTcGm;
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVectorFullLoad<COMP>::GetSeqUsed(uint32_t bIdx)
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
__aicore__ inline uint32_t KeyPoolBlockVectorFullLoad<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t KeyPoolBlockVectorFullLoad<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::hiddenStatesLayout == HIDDEN_STATES_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DataCopyAlignUbToUb(
    const LocalTensor<O> &dstLocal, const LocalTensor<O> &srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DataCopyAlignGmToUb(
    const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DataCopyAlignUbToGm(
    const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DataCopyWithOutputQue(
    const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
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

        outputQue1.EnQue(outputUb);
        outputQue1.DeQue<O>();

        DataCopyAlignUbToGm(dstGm[dstOffset], outputUb, curCopyRowCount, copyColCount, copyColCount, dstSingleRowCount);

        outputQue1.FreeTensor(outputUb);
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DataCopyWithInputQue(
    const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
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
        PipeBarrier<PIPE_V>();

        inputQue2.FreeTensor(inputUb);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::CopyInApe(uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t copyRowCount = coff_ * cmpRatio_;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;

    DataCopyWithInputQue(apeUb, apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
template <typename O>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::AddMultiDataToUb(
    const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm, uint32_t dealRowCount, uint32_t dealColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset)
{
    uint32_t cnt = dealRowCount * dealColCount;
    uint32_t groupSize = BUFFER_SIZE_BYTE_32K / (cnt * sizeof(O));
    uint32_t loopTimes = CeilDivT(repeatTimes, groupSize);
    uint64_t srcGmOffset = 0;
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        auto &inputQue = inputQue2;
        uint32_t curGroupSize = min(groupSize, (repeatTimes - groupSize * idx));
        LocalTensor<O> splitLocal = inputQue.AllocTensor<O>();
        if (srcSingleRowCount == dstSingleRowCount && dstSingleRowCount == dealRowCount) {
            for (uint32_t groupIdx = 0; groupIdx < curGroupSize; groupIdx++) {
                DataCopy(splitLocal[groupIdx * cnt], srcGm[srcGmOffset], cnt);
                srcGmOffset += offset;
            }
        } else {
            for (uint32_t groupIdx = 0; groupIdx < curGroupSize; groupIdx++) {
                DataCopyAlignGmToUb(splitLocal[groupIdx * cnt], srcGm[srcGmOffset], dealRowCount, dealColCount,
                                    srcSingleRowCount, dstSingleRowCount);
                srcGmOffset += offset;
            }
        }

        inputQue.EnQue(splitLocal);
        inputQue.DeQue<O>();

        PipeBarrier<PIPE_V>();
        if (idx == 0) {
            DataCopy(dstLocal, splitLocal, cnt);
            PipeBarrier<PIPE_V>();
            for (uint32_t groupIdx = 1; groupIdx < curGroupSize; groupIdx++) {
                Add(dstLocal, dstLocal, splitLocal[groupIdx * cnt], cnt);
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t groupIdx = 0; groupIdx < curGroupSize; groupIdx++) {
                Add(dstLocal, dstLocal, splitLocal[groupIdx * cnt], cnt);
                PipeBarrier<PIPE_V>();
            }
        }
        inputQue.FreeTensor(splitLocal);
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
template <bool IS_FULLLOAD>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::AddApe(const LocalTensor<T> &scoreLocal, uint32_t dealRowCount,
                                                                uint32_t dealColCount, uint32_t scoreSingleRowCount,
                                                                uint32_t apeSingleRowCount, uint64_t scoreOffset,
                                                                uint64_t apeOffset)
{
    if constexpr (IS_FULLLOAD) {
        Add(scoreLocal[scoreOffset], scoreLocal[scoreOffset], apeUb[apeOffset], coff_ * dealRowCount * dealColCount);
    } else {
        apeUb = inputQueApe.AllocTensor<T>();
        DataCopyAlignGmToUb(apeUb, apeGm_[apeOffset], coff_ * dealRowCount, dealColCount, constInfo_.headDim,
                            apeSingleRowCount);
        inputQueApe.EnQue(apeUb);
        inputQueApe.DeQue<T>();
        Add(scoreLocal[scoreOffset], scoreLocal[scoreOffset], apeUb, coff_ * dealRowCount * dealColCount);
        inputQueApe.FreeTensor(apeUb);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
                                                                       const Vec1SliceInfo &sliceInfo,
                                                                       uint32_t dDealSize, uint32_t dBaseSize,
                                                                       uint32_t dStartIdx, bool isApeFullLoad)
{
    uint32_t singleUbRowElemNum = dBaseSize * coff_;
    uint32_t singleApeRowElemNum = isApeFullLoad ? singleUbRowElemNum : constInfo_.headDim * coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleUbRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint32_t row = tcDealSize == 1 ? sliceInfo.validSeqCnt : (cmpRatio_ - sliceInfo.headHolderSeqCnt);

        if (isApeFullLoad) {
            uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleApeRowElemNum;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dBaseSize, scoreOffset, apeOffset);

        } else {
            uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleApeRowElemNum + dStartIdx;
            AddApe<false>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, scoreOffset, apeOffset);
        }
        scoreOffset += row * singleUbRowElemNum;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint32_t row = cmpRatio_ - sliceInfo.tailHolderSeqCnt;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * cmpRatio_ * singleUbRowElemNum;
        if (isApeFullLoad) {
            uint64_t apeOffset = 0;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dBaseSize, tailScoreOffset, apeOffset);

        } else {
            uint64_t apeOffset = dStartIdx;
            AddApe<false>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, tailScoreOffset, apeOffset);
        }
    }
    if (tcDealSize == 0) {
        return;
    }

    if (isApeFullLoad) {
        uint32_t row = cmpRatio_;
        for (uint32_t r = 0; r < tcDealSize; r++) {
            uint64_t curScoreOffset = scoreOffset + r * row * singleUbRowElemNum;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, curScoreOffset, 0U);
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal,
                                                                           const GlobalTensor<T> &srcGm,
                                                                           uint32_t preDealSeqCnt, uint32_t dealSeqCnt,
                                                                           uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = preDealSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    if (constInfo_.kBaseNum == 1) {
        DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
    } else {
        AddMultiDataToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount,
                         constInfo_.kBaseNum, constInfo_.mm1KvResSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::PadAlign(const LocalTensor<T> &dstLocal,
                                                                  const LocalTensor<T> &srcLocal,
                                                                  const Vec1SliceInfo &sliceInfo, uint32_t dBaseOffset,
                                                                  uint32_t dDealSize, uint32_t dBaseSize)
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
    uint32_t srcSingleRowElemNum = dBaseSize * coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * cmpRatio_ - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize * coff_; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum + dBaseOffset;

    uint64_t dstUbOffset = sliceInfo.key_pooledScCnt * cmpRatio_ * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + cmpRatio_) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                            copyRowCount - min(copyRowCount, cmpRatio_), copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dBaseSize;
    }
    // 右侧
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount, srcSingleRowCount,
                        dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                           const GlobalTensor<int32_t> &blockTableGm,
                                                                           const LocalTensor<T> &input,
                                                                           uint32_t batchIdx, uint32_t startSeqIdx,
                                                                           uint32_t endSeqIdx, uint32_t dDealSize,
                                                                           uint32_t dBaseSize, uint32_t stateIdx)
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
        // copyRowCount *= coff_;
        if (idInBlockTable < constInfo_.blockNum) { // vLLM block 0 is valid.
            uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                   remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim;
            uint64_t ubOffset = copyFinishRowCnt * coff_ * dBaseSize;
            DataCopyWithOutputQue(state[stateOffset], input[ubOffset], copyRowCount, dDealSize, coff_ * dBaseSize,
                                  coff_ * constInfo_.headDim * 2);
        }

        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
                                                                            const GlobalTensor<T> &state,
                                                                            const GlobalTensor<int32_t> &blockTableGm,
                                                                            uint32_t batchIdx, uint64_t startSeqIdx,
                                                                            uint64_t endSeqIdx, uint32_t dStartIdx,
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
                               remainRowCnt * 2 * coff_ * constInfo_.headDim + stateIdx * coff_ * constInfo_.headDim +
                               dStartIdx;

        DataCopyWithInputQue(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount, dDealSize,
                             coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal,
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::SaveState(
    const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx)
{
    uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dBaseSize;

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
            srcBaseOffset += (tailStartSeqIdx - startSeqIdx) * coff_ * dBaseSize;
            startSeqIdx = tailStartSeqIdx;
        }
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm[dStartIdx], blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx,
                          endSeqIdx, dDealSize, dBaseSize, stateIdx);
        srcBaseOffset += dBaseSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm[dStartIdx], blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                      dDealSize, dBaseSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::ReadState(const LocalTensor<T> &dstLocal,
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
            uint64_t startSeqIdx = sliceInfo.bStartPos < cmpRatio_ ?
                                       0 :
                                       Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_) - cmpRatio_;
            uint64_t endSeqIdx = min(
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
template <bool IS_SCORE>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::OverLap(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx + dBaseOffset, dDealSize,
                        static_cast<uint32_t>(IS_SCORE));

    if (sliceInfo.compressTcSize > 0) {
        PadAlign(dstLocal, srcLocal, sliceInfo, dBaseOffset, dDealSize, dBaseSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm;
            LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx, globalSeqIdx,
                              dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::SaveToWorkSpace(const LocalTensor<T> &srcLocal,
                                                                         const GlobalTensor<T> &cacheTcGm,
                                                                         const Vec1SliceInfo &sliceInfo,
                                                                         const LoopInfo &loopInfo, uint32_t dStartIdx,
                                                                         uint32_t dDealSize)
{
    uint32_t curSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt;
    uint32_t totalSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.bSeqUsed;
    if (!loopInfo.isCoreRowLast || !loopInfo.isCoreLoopLast || !sliceInfo.isLast || totalSeqLen < cmpRatio_ ||
        curSeqLen > Trunc(totalSeqLen, cmpRatio_) - cmpRatio_) {
        return;
    }
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset =
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, cmpRatio_)) * srcSingleRowElemNum;
    DataCopyWithOutputQue(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                          curSeqLen - max((uint64_t)(curSeqLen - cmpRatio_), sliceInfo.bStartPos), dDealSize,
                          coff_ * dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::LoadFromWorkSpace(
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx, uint32_t needDealTcSize)
{
    KeyPoolVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = scoreMm1ResGm_;
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, scoreCacheTcGm_,
                      overLapSliceInfo, loopInfo, dStartIdx, dBaseOffset,
                      originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize, dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }

    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_;
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, kvCacheTcGm_, overLapSliceInfo,
                       loopInfo, dStartIdx, dBaseOffset, originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize,
                       dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal,
                                                                   const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                                                   uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::RoundToHiddenDtype(const LocalTensor<T> &srcLocal,
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::AddApeToPooledScore(const LocalTensor<T> &scoreLocal,
                                                                             uint32_t poolCount, uint32_t dDealSize)
{
    uint32_t poolElementCount = cmpRatio_ * dDealSize;
    for (uint32_t pool = 0; pool < poolCount; pool++) {
        Add(scoreLocal[pool * poolElementCount], scoreLocal[pool * poolElementCount], apeUb, poolElementCount);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                          const LocalTensor<T> &scoreLocal,
                                                                          const LocalTensor<T> &dstLocal,
                                                                          const LocalTensor<T> &tmpUb,
                                                                          uint32_t tcDealSize, uint32_t dDealSize)
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb,
                                                                                const Vec1SliceInfo &sliceInfo,
                                                                                uint32_t compressTcSize,
                                                                                uint32_t dStartIdx, uint32_t dDealSize)
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
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::DealVec1BaseBlock(
    KeyPoolVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = tmpBuf1.Get<T>();
    LocalTensor<T> kvLocal = scoreLocal[BUFFER_SIZE_BYTE_16K / sizeof(T)];
    OverLapScoreKv(scoreLocal, kvLocal, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dBaseOffset, dDealSize,
                   dBaseSize, dealSeqStartIdx, needDealTcSize);
    if (hasLayerNorm_ && coff_ == 1 && dDealSize == constInfo_.headDim && statisticInfo.key_poolScCnt > 0) {
        LocalTensor<T> normScratch = inputQue2.AllocTensor<T>();
        KeyPoolLayerNormRowsInplace(kvLocal, normScratch, normWeightGm_, normBiasGm_, constInfo_.normEps,
                                    statisticInfo.key_poolScCnt * cmpRatio_, dDealSize, constInfo_.headDim);
        inputQue2.FreeTensor(normScratch);
    }
    if (statisticInfo.key_poolScCnt > 0) {
        LocalTensor<T> tmpUb = tmpBuf2.Get<T>();
        AddApeToPooledScore(scoreLocal, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        SoftmaxDN(scoreLocal, tmpUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        RoundToHiddenDtype(scoreLocal, statisticInfo.key_poolScCnt * cmpRatio_ * dDealSize);
        LocalTensor<T> comperssoredUb = scoreLocal;
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, tmpUb, statisticInfo.key_poolScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.key_poolScCnt, dStartIdx + dBaseOffset,
                               dDealSize);
    }
    compressedCnt_ += statisticInfo.key_poolScCnt;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::CalcGroupInfo(Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    if (hasLayerNorm_ || cmpRatio_ <= 4) {
        splitInfo.dBaseSize = constInfo_.headDim;
    } else {
        splitInfo.dBaseSize =
            constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, constInfo_.batchSize)));
        if (constInfo_.kBaseNum > 1) {
            splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
        }
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(HIDDEN_STATES_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<HIDDEN_STATES_T>());
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), constInfo_.batchSize);
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::CalcTaskDistribution(Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t totalDealBatchNum = constInfo_.batchSize;

    if (blockIdx < groupSize * (totalDealBatchNum % groupNum)) {
        splitInfo.dealBatchNum = totalDealBatchNum / groupNum + 1;
        splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (blockIdx / groupSize);
    } else if (blockIdx < groupSize * groupNum) {
        splitInfo.dealBatchNum = totalDealBatchNum / groupNum;
        splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (blockIdx / groupSize) + totalDealBatchNum % groupNum;
    } else {
        splitInfo.dealBatchNum = 0;
        splitInfo.preDealBatchNum = totalDealBatchNum;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::UpdateIteratorState(Vec1SplitInfo &splitInfo)
{
    splitInfo.preCompressedCnt = 0;
    splitInfo.dealSeqStartIdx = splitInfo.preDealBatchNum * constInfo_.sSize;
    splitInfo.curBStart = splitInfo.preDealBatchNum;
    splitInfo.dealSeqCnt = splitInfo.dealBatchNum * constInfo_.sSize;
    splitInfo.curSStart = 0;
    for (uint32_t curB = 0; curB < constInfo_.batchSize; curB++) {
        uint32_t startPos = GetStartPos(curB);
        uint32_t seqLength = GetSeqLength(curB);
        if (curB < splitInfo.curBStart) {
            splitInfo.preCompressedCnt += (startPos + seqLength) / cmpRatio_ - startPos / cmpRatio_;
        }
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
{
    // 计算headDim和Tc方向切分大小
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_16K / (cmpRatio_ * coff_ * sizeof(T));

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
__aicore__ inline Vec1SplitInfo KeyPoolBlockVectorFullLoad<COMP>::SplitCoreV1()
{
    Vec1SplitInfo splitInfo;

    // 1. 计算基础分组和分片大小
    CalcGroupInfo(splitInfo);

    // 2. 根据当前的 BlockIdx 计算任务分配（负载均衡）
    CalcTaskDistribution(splitInfo);

    // 3. 刷新迭代器并获取当前核的起始位置状态
    UpdateIteratorState(splitInfo);

    if (splitInfo.dealBatchNum == 0) {
        return splitInfo;
    }

    // 4. 计算具体在内存中的切块（Tiling）逻辑
    CalcTilingStrategy(splitInfo);

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::PrepareForComputeVec()
{
    vec1SplitInfo_ = SplitCoreV1();
    if (vec1SplitInfo_.dealBatchNum == 0) {
        return;
    }
    loopInfo_.groupSize = vec1SplitInfo_.vec1GroupSize;
    loopInfo_.groupNum = vec1SplitInfo_.vec1GroupNum;
    loopInfo_.coreRowIdx = GetBlockIdx() / vec1SplitInfo_.vec1GroupSize;
    loopInfo_.coreColIdx = GetBlockIdx() % vec1SplitInfo_.vec1GroupSize;
    loopInfo_.isCoreRowLast = loopInfo_.coreRowIdx == vec1SplitInfo_.vec1GroupNum - 1;
    loopInfo_.isCoreRowFirst = loopInfo_.coreRowIdx == 0;

    bool isApeFullLoad = coff_ * cmpRatio_ * vec1SplitInfo_.dBaseSize * sizeof(T) <= BUFFER_SIZE_BYTE_16K;
    if (isApeFullLoad) {
        uint64_t baseOffset = loopInfo_.coreColIdx * vec1SplitInfo_.dBaseSize;
        apeUb = inputQueApe.AllocTensor<T>();
        CopyInApe(baseOffset, vec1SplitInfo_.dBaseSize);
        inputQueApe.EnQue(apeUb);
        inputQueApe.DeQue<T>();
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolBlockVectorFullLoad<COMP>::ComputeVec1()
{
    // 计算当前VecCore的任务量
    if (vec1SplitInfo_.dealBatchNum == 0) {
        return;
    }

    KeyPoolVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    // 切块循环
    uint64_t baseOffset = loopInfo_.coreColIdx * vec1SplitInfo_.dBaseSize;

    uint32_t cnt = constInfo_.sSize * vec1SplitInfo_.dBaseSize * coff_;
    uint32_t singleLoopBatchNum = BUFFER_SIZE_BYTE_16K / (cnt * sizeof(T));
    uint32_t loopTimes = CeilDivT(vec1SplitInfo_.dealBatchNum, singleLoopBatchNum);
    bool isApeFullLoad = coff_ * cmpRatio_ * vec1SplitInfo_.dBaseSize * sizeof(T) <= BUFFER_SIZE_BYTE_16K;
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint32_t curLoopBatchNum = min(singleLoopBatchNum, vec1SplitInfo_.dealBatchNum - singleLoopBatchNum * idx);
        scoreUb = inputQue1.AllocTensor<T>();
        kvUb = scoreUb[BUFFER_SIZE_BYTE_16K / sizeof(T)];
        FromWokrSpaceToUb(scoreUb, scoreMm1ResGm_, vec1SplitInfo_.dealSeqStartIdx, curLoopBatchNum * constInfo_.sSize,
                          baseOffset, vec1SplitInfo_.dBaseSize);
        FromWokrSpaceToUb(kvUb, kvMm1ResGm_, vec1SplitInfo_.dealSeqStartIdx, curLoopBatchNum * constInfo_.sSize,
                          baseOffset, vec1SplitInfo_.dBaseSize);
        inputQue1.EnQue(scoreUb);
        inputQue1.DeQue<T>();
        uint32_t projectionElementCount = curLoopBatchNum * constInfo_.sSize * vec1SplitInfo_.dBaseSize * coff_;
        RoundToHiddenDtype(scoreUb, projectionElementCount);
        RoundToHiddenDtype(kvUb, projectionElementCount);
        vec1SplitInfo_.dealTcNum = 0;
        uint32_t curLoopCompressedCnt = 0;
        for (uint32_t curB = vec1SplitInfo_.curBStart; curB < vec1SplitInfo_.curBStart + curLoopBatchNum; curB++) {
            uint32_t startPos = GetStartPos(curB);
            uint32_t seqLength = GetSeqLength(curB);
            vec1SplitInfo_.dealTcNum += CeilDivT(startPos + seqLength, cmpRatio_) - (startPos / cmpRatio_);
            curLoopCompressedCnt += (startPos + seqLength) / cmpRatio_ - startPos / cmpRatio_;
        }
        sliceIterator.Reset(vec1SplitInfo_.curBStart, vec1SplitInfo_.curSStart, 0U, 0U);
        sliceIterator.SetNeedDealTcSize(vec1SplitInfo_.dealTcNum);
        sliceIterator.SetDealedTcCnt(0U);
        Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();
        while (!sliceIterator.IsEnd()) {
            sliceIterator.GetSlice();
            SaveState(kvUb, stateCacheGm_, stateBlockTableGm_, sliceInfo, baseOffset, vec1SplitInfo_.dBaseSize,
                      vec1SplitInfo_.dBaseSize, kvStateIdx_);

            SaveState(scoreUb, stateCacheGm_, stateBlockTableGm_, sliceInfo, baseOffset, vec1SplitInfo_.dBaseSize,
                      vec1SplitInfo_.dBaseSize, scoreStateIdx_);
            sliceIterator.IteratorSlice();
        }
        if (curLoopCompressedCnt == 0) {
            inputQue1.FreeTensor(scoreUb);
            continue;
        }
        for (uint32_t dLoopIdx = 0; dLoopIdx < vec1SplitInfo_.dLoopCount; dLoopIdx++) {
            uint64_t dBaseOffset = baseOffset + dLoopIdx * vec1SplitInfo_.dSplitSize;
            loopInfo_.dLoopIdx = dLoopIdx;

            sliceIterator.Reset(vec1SplitInfo_.curBStart, vec1SplitInfo_.curSStart, 0U, 0U);
            compressedCnt_ = vec1SplitInfo_.preCompressedCnt;
            for (uint32_t tcIdx = 0; tcIdx < vec1SplitInfo_.dealTcNum; tcIdx += vec1SplitInfo_.tcSplitSize) {
                uint32_t actDealTcSize = min(vec1SplitInfo_.tcSplitSize, vec1SplitInfo_.dealTcNum - tcIdx);

                loopInfo_.isCoreLoopFirst = tcIdx == 0;
                loopInfo_.isCoreLoopLast = tcIdx + vec1SplitInfo_.tcSplitSize >= vec1SplitInfo_.dealTcNum;
                // 处理单个切块
                sliceIterator.SetNeedDealTcSize(actDealTcSize);
                sliceIterator.SetDealedTcCnt(0U);
                DealVec1BaseBlock(sliceIterator, loopInfo_, baseOffset, dLoopIdx * vec1SplitInfo_.dSplitSize,
                                  vec1SplitInfo_.dSplitSize, vec1SplitInfo_.dBaseSize, vec1SplitInfo_.dealSeqStartIdx);
            }
        }
        inputQue1.FreeTensor(scoreUb);
        vec1SplitInfo_.curBStart += curLoopBatchNum;
        vec1SplitInfo_.dealSeqStartIdx += curLoopBatchNum * constInfo_.sSize;
        vec1SplitInfo_.preCompressedCnt += curLoopCompressedCnt;
    }
    if (isApeFullLoad) {
        inputQueApe.FreeTensor(apeUb);
    }
}

} // namespace KeyPool
#endif // KEY_POOL_BLOCK_VEC_FULL_LOAD_H
