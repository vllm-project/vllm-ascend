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
 * \file compressor_block_vec.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_H
#define COMPRESSOR_BLOCK_VEC_H

#include "compressor_comm.h"
#include "compressor_tools.h"
#include "vf/vf_softmax.h"
#include "vf/vf_add.h"
#include "vf/vf_mul.h"
#include "vf/vf_rms_norm.h"
#include "vf/vf_rope.h"


using namespace AscendC;

namespace Compressor {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP>
class CompressorBlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    float SOFTMAX_MIN_NUM = static_cast<float>(-1.0 / 0.0);
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const CompressorTools<COMP> &tools);
    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm,
                                                GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void ComputeVec2(const Vec2RunInfo &info);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> vec2InputGm_;
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd,
                                             uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize,
                                           uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info, CompressorVec1SliceIterator<COMP> &sliceIterator,
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
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const Vec1RunInfo &info, const LoopInfo &loopInfo,
                                          const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo,
                                          uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t needDealTcSize);
    __aicore__ inline void CopyOutVec1Res(const GlobalTensor<T> &resGm, const Vec1RunInfo &info,
                                          const LocalTensor<T> &comperssoredUb, uint32_t compressTcSize,
                                          uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    __aicore__ inline Vec2SplitInfo SplitCoreV2(const Vec2RunInfo &info);
    __aicore__ inline void MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb,
                                          const LocalTensor<T> &normWeightUb, const LocalTensor<T> &tempLocal,
                                          uint32_t dealRowCount);
    __aicore__ inline void CalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
                                   const Vec2SliceInfo &originSliceInfo, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb,
                                              CompressorVec2SliceIterator<COMP> &sliceIterator);
    __aicore__ inline void DealVec2BaseBlock(const Vec2RunInfo &info, const Vec2SplitInfo &splitInfo,
                                             CompressorVec2SliceIterator<COMP> &sliceIterator);
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
    CompressorTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<T> normWeightGm_;
    GlobalTensor<T> ropeSinGm_;
    GlobalTensor<T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    // TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> normWeightUb;
    LocalTensor<T> apeUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> apeBuf;
    TBuf<TPosition::VECCALC> normWeightBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TQue<QuePosition::VECIN, 1> inputQue3;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;
};


template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo,
                                                               const CompressorTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                                         __gm__ uint8_t *stateCache, __gm__ uint8_t *ape,
                                                         __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin,
                                                         __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                                         __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed,
                                                         __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T *)stateCache);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    normWeightGm_.SetGlobalBuffer((__gm__ T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
    isExistSeqUsed_ = (seqUsed != nullptr);
    isExistStartPos_ = (startPos != nullptr);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue3, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_32K);
    normWeightUb = normWeightBuf.Get<T>();
    apeUb = apeBuf.Get<T>();
    LocalTensor<T> normweightInUb = inputQue2.AllocTensor<T>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    inputQue2.EnQue(normweightInUb);
    inputQue2.DeQue<T>();
    DataCopy(normWeightUb, normweightInUb, constInfo_.headDim);
    inputQue2.FreeTensor(normweightInUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::AllocEventID()
{
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::FreeEventID()
{
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                  GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm,
                                                  GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm)
{
    this->kvMm1ResGm_ = kvMm1ResGm;
    this->scoreMm1ResGm_ = scoreMm1ResGm;
    this->kvCacheTcGm_ = kvCacheTcGm;
    this->scoreCacheTcGm_ = scoreCacheTcGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec2InputGm_ = vec2InputGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx,
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
__aicore__ inline void CompressorBlockVector<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
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
__aicore__ inline void
CompressorBlockVector<COMP>::AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
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
__aicore__ inline void
CompressorBlockVector<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal, const LocalTensor<O> &srcLocal,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorBlockVector<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorBlockVector<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstGm, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorBlockVector<COMP>::DataCopyWithOutputQue(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
                                                   uint32_t copyRowCount, uint32_t copyColCount,
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
__aicore__ inline void
CompressorBlockVector<COMP>::DataCopyWithInputQue(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                                  uint32_t copyRowCount, uint32_t copyColCount,
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

        inputQue2.FreeTensor(inputUb);
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorBlockVector<COMP>::AddMultiDataToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                              uint32_t dealRowCount, uint32_t dealColCount, uint32_t srcSingleRowCount,
                                              uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset)
{
    uint32_t cnt = dealRowCount * dstSingleRowCount;
    uint32_t groupSize = BUFFER_SIZE_BYTE_32K / (cnt * sizeof(O));
    uint32_t loopTimes = CeilDivT(repeatTimes, groupSize);
    uint64_t srcGmOffset = 0;
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        auto &inputQue = idx % 2 == 0 ? inputQue2 : inputQue3;
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
            MultiAddVF<true>(dstLocal, splitLocal, dealRowCount, dealColCount, dstSingleRowCount, curGroupSize, cnt);
        } else {
            MultiAddVF<false>(dstLocal, splitLocal, dealRowCount, dealColCount, dstSingleRowCount, curGroupSize, cnt);
        }
        inputQue.FreeTensor(splitLocal);
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::PadAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                      const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
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

    uint64_t dstUbOffset = sliceInfo.compressoredScCnt * cmpRatio_ * dstSingleRowCount;
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
__aicore__ inline void
CompressorBlockVector<COMP>::OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                     const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                     const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                     uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if constexpr (IS_SCORE) {
        AddApeToScore(srcLocal, apeUb, sliceInfo, dDealSize);
        PipeBarrier<PIPE_V>();
    }
    SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));
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
__aicore__ inline void
CompressorBlockVector<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                               const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo,
                                               uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = statisticInfo.dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    if (constInfo_.kBaseNum == 1) {
        DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
    } else {
        AddMultiDataToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount,
                         constInfo_.kBaseNum, constInfo_.mm1KvResSize);
    }
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::SaveToWorkSpace(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &cacheTcGm,
                                             const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t dDealSize)
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
                          curSeqLen - max(curSeqLen - cmpRatio_, sliceInfo.bStartPos), dDealSize, coff_ * dDealSize,
                          constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::LoadFromWorkSpace(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm,
                                               const GlobalTensor<T> &srcGm, const LocalTensor<T> &srcLocal,
                                               const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                               uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, cmpRatio_);
    uint64_t dstLocalOffset =
        (sliceInfo.compressoredScCnt * cmpRatio_ + cmpRatio_ - copyRowCount) * dstSingleRowElemNum;
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
__aicore__ inline void
CompressorBlockVector<COMP>::ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                                const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx,
                                                uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                                uint32_t dDealSize, uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::CONTINUOUS) {
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
                                   remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;

            DataCopyWithInputQue(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                 dDealSize, coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
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
                                   remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;

            DataCopyWithInputQue(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                 dDealSize, coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                               const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                               uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize,
                                               uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::CONTINUOUS) {
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
            if (idInBlockTable != 0) { // 32
                uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                       remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                       stateIdx * coff_ * constInfo_.headDim + dStartIdx;
                DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                      dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * 2);
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
                                   remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                  dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * 2);

            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                       const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                       uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dDealSize;

    if constexpr (COMP::cacheMode == CACHE_MODE::CYCLE) {
        uint32_t compressSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.bSeqUsed, cmpRatio_);
        uint32_t writeSeqStartIdx = compressSeqIdx > (coff_ - 1) * cmpRatio_ ?
                                    compressSeqIdx - (coff_ - 1) * cmpRatio_ : 0;
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, writeSeqStartIdx) - startSeqIdx) * coff_ * dDealSize;
        startSeqIdx = max(startSeqIdx, writeSeqStartIdx);
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
__aicore__ inline void
CompressorBlockVector<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                                 uint32_t duplicateColCount, uint32_t singleRowCount)
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
__aicore__ inline void
CompressorBlockVector<COMP>::ReadState(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &stateGm,
                                       const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                       uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    // 填充右边
    if (sliceInfo.headHolderSeqCnt > 0) {
        // 整个batch的第一块
        uint32_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, cmpRatio_);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
        if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < cmpRatio_;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], cmpRatio_, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < cmpRatio_ && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx = sliceInfo.bStartPos < cmpRatio_ ?
                                       0 :
                                       Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, cmpRatio_) - cmpRatio_;
            uint32_t endSeqIdx =
                min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, cmpRatio_) - cmpRatio_,
                    sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += cmpRatio_ * coff_ * dDealSize;
            }
            ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal, uint32_t tcDealSize,
                                                              uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = coff_ * cmpRatio_;
    FaVectorApi::SoftmaxDnVF<T>(scoreLocal, scoreLocal, dDealSize, ReduceSize, tcDealSize, minValue, dDealSize);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                              const LocalTensor<T> &dstLocal, uint32_t tcDealSize, uint32_t dDealSize)
{
    MulReduceSumbaseVF(kvLocal, scoreLocal, dstLocal, coff_, cmpRatio_, dDealSize, tcDealSize);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::CopyOutVec1Res(const GlobalTensor<T> &resGm, const Vec1RunInfo &info,
                                            const LocalTensor<T> &comperssoredUb, uint32_t compressTcSize,
                                            uint32_t dStartIdx, uint32_t dDealSize)
{
    uint64_t outGmOffset = compressedCnt_ * constInfo_.headDim + dStartIdx;
    DataCopyAlignUbToGm(resGm[outGmOffset], comperssoredUb, compressTcSize, dDealSize, dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
    uint32_t dBaseSize, uint32_t needDealTcSize)
{
    CompressorVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = scoreMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(scoreUb, scoreDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(scoreUb);
    inputQue1.DeQue<T>();
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

    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(kvUb);
    inputQue1.DeQue<T>();
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
__aicore__ inline void CompressorBlockVector<COMP>::DealVec1BaseBlock(const Vec1RunInfo &info,
                                                                      CompressorVec1SliceIterator<COMP> &sliceIterator,
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

    if (statisticInfo.compressorScCnt > 0) {
        SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        LocalTensor<T> comperssoredUb = outputQue2.AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        outputQue2.EnQue(comperssoredUb);
        outputQue2.DeQue<T>();
        GlobalTensor<T> resGm = vec1ResGm_[info.v1v2DbIdx * constInfo_.dbSize];
        CopyOutVec1Res(resGm, info, comperssoredUb, statisticInfo.compressorScCnt, dStartIdx, dDealSize);
        outputQue2.FreeTensor(comperssoredUb);
    }
    compressedCnt_ += statisticInfo.compressorScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize = constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, info.dealTcNum)));
    if (constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_BLOCK_ELEMENT_NUM);
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), info.dealTcNum);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcTaskDistribution(const Vec1RunInfo &info,
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
__aicore__ inline void CompressorBlockVector<COMP>::UpdateIteratorState(const Vec1RunInfo &info,
                                                                        Vec1SplitInfo &splitInfo)
{
    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.Reset(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();

    // 处理前序任务量，更新起始索引
    if (splitInfo.preDealTcSize > 0) {
        sliceIterator.SetNeedDealTcSize(splitInfo.preDealTcSize);
        StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
        splitInfo.curCompressedCnt = statisticInfo.compressorScCnt;
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
    splitInfo.totalCompressedCnt = splitInfo.curCompressedCnt + statisticInfo.compressorScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
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
__aicore__ inline Vec1SplitInfo CompressorBlockVector<COMP>::SplitCoreV1(const Vec1RunInfo &info)
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
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    if (info.dealTcNum == 0) {
        return;
    }
    if (info.resetResFlag) {
        compressedCnt_ = 0;
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


    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
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
template <typename COMP>
__aicore__ inline Vec2SplitInfo CompressorBlockVector<COMP>::SplitCoreV2(const Vec2RunInfo &info)
{
    Vec2SplitInfo splitInfo;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;

    if (blockIdx < (info.dealScSize % aiCoreNum)) {
        splitInfo.dealScNum = info.dealScSize / aiCoreNum + 1;
        splitInfo.preScCnt = splitInfo.dealScNum * blockIdx;
    } else if (blockIdx < aiCoreNum) {
        splitInfo.dealScNum = info.dealScSize / aiCoreNum;
        splitInfo.preScCnt = splitInfo.dealScNum * blockIdx + info.dealScSize % aiCoreNum;
    } else {
        splitInfo.dealScNum = 0;
        splitInfo.preScCnt = info.dealScSize;
    }
    if (splitInfo.dealScNum == 0) {
        return splitInfo;
    }

    uint32_t preScCnt = splitInfo.preScCnt;
    for (uint32_t curB = info.bStart, curSc = info.bCompressedId; curB < constInfo_.batchSize; curB++, curSc = 0) {
        uint32_t startPos = GetStartPos(curB);
        uint32_t seqUsed = GetSeqUsed(curB);
        uint32_t curScNum = ((startPos + seqUsed) / cmpRatio_) - (startPos / cmpRatio_) - curSc;
        if (preScCnt < curScNum) {
            splitInfo.curBStart = curB;
            splitInfo.curScStart = curSc + preScCnt;
            break;
        }
        preScCnt -= curScNum;
    }

    splitInfo.dealedScCnt = info.compressedId - info.dealScSize;

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const Vec2RunInfo &info)
{
    Vec2SplitInfo splitInfo = SplitCoreV2(info);
    if (splitInfo.dealScNum == 0) {
        return;
    }
    CompressorVec2SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.Reset(splitInfo.curBStart, splitInfo.curScStart, splitInfo.dealedScCnt + splitInfo.preScCnt);

    uint32_t singleLoopScNum = BUFFER_SIZE_BYTE_32K / (constInfo_.headDim * sizeof(T));
    uint32_t loopTimes = CeilDivT(splitInfo.dealScNum, singleLoopScNum);

    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint32_t curLoopScNum = min(singleLoopScNum, splitInfo.dealScNum - singleLoopScNum * idx);
        sliceIterator.SetNeedDealScSize(curLoopScNum);
        sliceIterator.ResetLoopDealedScCnt();
        DealVec2BaseBlock(info, splitInfo, sliceIterator);
        splitInfo.preScCnt += curLoopScNum;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::DealVec2BaseBlock(const Vec2RunInfo &info,
                                                                      const Vec2SplitInfo &splitInfo,
                                                                      CompressorVec2SliceIterator<COMP> &sliceIterator)
{
    Vec2SliceInfo &sliceInfo = sliceIterator.GetSlice();
    uint32_t needDealScSize = sliceIterator.GetNeedDealScSize();
    uint32_t computeSize = needDealScSize * constInfo_.headDim;
    int64_t inGmOffset = splitInfo.preScCnt * constInfo_.headDim;
    GlobalTensor<T> vec2InputGm = vec2InputGm_[info.v2DbIdx * constInfo_.dbSize];
    // CopyIn
    LocalTensor<T> vec1ResUb = inputQue1.AllocTensor<T>();
    DataCopy(vec1ResUb, vec2InputGm[inGmOffset], computeSize);
    inputQue1.EnQue(vec1ResUb);
    inputQue1.DeQue<T>();

    // RmsNorm
    LocalTensor<T> normResUb = tmpBuff1.Get<T>();
    LocalTensor<T> tempLocal = tmpBuff2.Get<T>();
    PipeBarrier<PIPE_V>();
    MultRowRmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, needDealScSize);
    inputQue1.FreeTensor(vec1ResUb);

    // rope: 只对后RD进行rope
    LocalTensor<X_T> outputUb = outputQue1.AllocTensor<X_T>();
    PipeBarrier<PIPE_V>();
    CalRope(outputUb, normResUb, sliceInfo, needDealScSize);
    PipeBarrier<PIPE_V>();

    // CopyOut
    outputQue1.EnQue(outputUb);
    outputQue1.DeQue<X_T>();
    CopyFinalResultOut(outputUb, sliceIterator);
    outputQue1.FreeTensor(outputUb);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVector<COMP>::MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb,
                                            const LocalTensor<T> &normWeightUb, const LocalTensor<T> &tempLocal,
                                            uint32_t dealRowCount)
{
    uint32_t row = 1;
    uint32_t col = constInfo_.headDim;
    float reciprocal = 1.0f / col;
    float epsilon = constInfo_.normEps;
    for (uint32_t i = 0; i < dealRowCount; ++i) {
        RmsNormVF(normResUb[i * col], vec1ResUb[i * col], normWeightUb, reciprocal, epsilon, row, col);
    }
}


template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalRope(const LocalTensor<X_T> &outputUb,
                                                            const LocalTensor<T> &normResUb,
                                                            const Vec2SliceInfo &originSliceInfo, uint32_t dealRowCount)
{
    CompressorVec2SliceIterator<COMP> ropeSliceIterator(tools_);
    ropeSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    ropeSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.scIdx, originSliceInfo.dealedScCnt);
    Vec2SliceInfo &sliceInfo = ropeSliceIterator.GetSlice();

    ropeSliceIterator.SetNeedDealScSize(dealRowCount);
    while (!ropeSliceIterator.IsEnd()) {
        ropeSliceIterator.GetSlice();
        if (sliceInfo.curDealScNum > 0) {
            uint32_t computeSize = sliceInfo.curDealScNum * constInfo_.ropeHeadDim;
            uint64_t SinCosOffset = sliceInfo.padScIdx * constInfo_.ropeHeadDim;

            // sin与cos各占一半, 实际分别最多只会用16K,总占用32K
            LocalTensor<T> cosUb = inputQue2.AllocTensor<T>();
            LocalTensor<T> sinUb = cosUb[BUFFER_SIZE_BYTE_16K / sizeof(T)];
            DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
            DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
            inputQue2.EnQue(sinUb);
            inputQue2.DeQue<T>();

            RopeVF<COMP::rotaryMode>(sinUb, cosUb, normResUb[sliceInfo.loopDealedScCnt * constInfo_.headDim],
                                     outputUb[sliceInfo.loopDealedScCnt * constInfo_.headDim], sliceInfo.curDealScNum,
                                     constInfo_.ropeHeadDim, constInfo_.headDim,
                                     constInfo_.headDim - constInfo_.ropeHeadDim);
            inputQue2.FreeTensor(sinUb);
        }
        ropeSliceIterator.IteratorSlice();
    }
}


template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb,
                                                                       CompressorVec2SliceIterator<COMP> &sliceIterator)
{
    Vec2SliceInfo &sliceInfo = sliceIterator.GetSlice();
    while (!sliceIterator.IsEnd()) {
        sliceIterator.GetSlice();
        if (sliceInfo.curDealScNum > 0) {
            DataCopy(cmpKvOutGm_[sliceInfo.padScIdx * constInfo_.headDim],
                     cmpKvOutUb[sliceInfo.loopDealedScCnt * constInfo_.headDim],
                     sliceInfo.curDealScNum * constInfo_.headDim);
        }
        sliceIterator.IteratorSlice();
    }
}

} // namespace Compressor
#endif // COMPRESSOR_BLOCK_VECTOR_H
