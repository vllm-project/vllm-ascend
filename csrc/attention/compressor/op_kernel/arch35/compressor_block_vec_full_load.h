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
 * \file compressor_block_vec_full_load.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_FULL_LOAD_H
#define COMPRESSOR_BLOCK_VEC_FULL_LOAD_H

#include "compressor_comm.h"
#include "compressor_tools.h"
#include "vf/vf_softmax.h"
#include "vf/vf_add.h"
#include "vf/vf_mul.h"
#include "vf/vf_rms_norm.h"
#include "vf/vf_rope.h"
#include <cstdint>


using namespace AscendC;

namespace Compressor {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP>
class CompressorBlockVectorFullLoad {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float SOFTMAX_MIN_NUM = -2e38;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVectorFullLoad(){};
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
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm,
                                                GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void ComputeVec1();
    __aicore__ inline void ComputeVec2();

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
                                              uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                              uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                     uint32_t stateIdx);
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
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutVec1Res(const GlobalTensor<T> &resGm, const LocalTensor<T> &comperssoredUb,
                                          uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void DealVec1BaseBlock(CompressorVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t dBaseOffset, uint32_t dDealSize,
                                             uint32_t dBaseSize, uint32_t dealSeqStartIdx);
    __aicore__ inline void MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb,
                                          const LocalTensor<T> &normWeightUb, const LocalTensor<T> &tempLocal,
                                          uint32_t dealRowCount);
    __aicore__ inline void CalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
                                   const Vec2SliceInfo &originSliceInfo, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb,
                                              CompressorVec2SliceIterator<COMP> &sliceIterator);
    __aicore__ inline void DealVec2BaseBlock(const Vec2SplitInfo &splitInfo,
                                             CompressorVec2SliceIterator<COMP> &sliceIterator);
    __aicore__ inline void CalcGroupInfo(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1();
    __aicore__ inline Vec2SplitInfo SplitCoreV2();
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t compressedCnt_ = 0;
    uint32_t totalCompressedCnt_ = 0;
    uint32_t kvStateIdx_ = 0;
    uint32_t scoreStateIdx_ = 1;
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    CompressorTools<COMP> tools_;
    ConstInfo constInfo_ = {};
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
    LocalTensor<T> scoreUb;
    LocalTensor<T> kvUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> apeBuf;
    TBuf<TPosition::VECCALC> normWeightBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TQue<QuePosition::VECIN, 1> inputQue3;
    TQue<QuePosition::VECIN, 1> inputQueApe;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;
};


template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::InitParams(const ConstInfo &constInfo,
                                                                       const CompressorTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::Init(
    __gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate, __gm__ uint8_t *stateCache, __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut)
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQue3, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(inputQueApe, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(tmpBuf1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuf2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_16K);
    normWeightUb = normWeightBuf.Get<T>();
    LocalTensor<T> normweightInUb = inputQue2.AllocTensor<T>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    inputQue2.EnQue(normweightInUb);
    inputQue2.DeQue<T>();
    DataCopy(normWeightUb, normweightInUb, constInfo_.headDim);
    inputQue2.FreeTensor(normweightInUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::AllocEventID()
{
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::FreeEventID()
{
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
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
__aicore__ inline uint32_t CompressorBlockVectorFullLoad<COMP>::GetSeqUsed(uint32_t bIdx)
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
__aicore__ inline uint32_t CompressorBlockVectorFullLoad<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorFullLoad<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}


template <typename COMP>
template <typename O>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal, const LocalTensor<O> &srcLocal,
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
CompressorBlockVectorFullLoad<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
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
CompressorBlockVectorFullLoad<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
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
CompressorBlockVectorFullLoad<COMP>::DataCopyWithOutputQue(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
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
CompressorBlockVectorFullLoad<COMP>::DataCopyWithInputQue(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CopyInApe(uint32_t dStartIdx,
                                                                      uint32_t dDealSize)
{
    apeUb = apeBuf.Get<T>();

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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::AddMultiDataToUb(
    const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm, uint32_t dealRowCount, uint32_t dealColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset)
{
    uint32_t cnt = dealRowCount * dealColCount;
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
            MultiAddVF<true>(dstLocal, splitLocal, dealRowCount, dealColCount, dealColCount, curGroupSize, cnt);
        } else {
            MultiAddVF<false>(dstLocal, splitLocal, dealRowCount, dealColCount, dealColCount, curGroupSize, cnt);
        }
        inputQue.FreeTensor(splitLocal);
    }
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
template <bool IS_FULLLOAD>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::AddApe(const LocalTensor<T> &scoreLocal,
                                                                   uint32_t dealRowCount, uint32_t dealColCount,
                                                                   uint32_t scoreSingleRowCount,
                                                                   uint32_t apeSingleRowCount,
                                                                   uint64_t scoreOffset, uint64_t apeOffset)
{
    if constexpr (IS_FULLLOAD) {
        AddVF(scoreLocal[scoreOffset], apeUb[apeOffset], coff_ * dealRowCount, dealColCount, scoreSingleRowCount,
              apeSingleRowCount);
    } else {
        apeUb = inputQueApe.AllocTensor<T>();
        DataCopyAlignGmToUb(apeUb, apeGm_[apeOffset], coff_ * dealRowCount, dealColCount, constInfo_.headDim,
                            apeSingleRowCount);
        inputQueApe.EnQue(apeUb);
        inputQueApe.DeQue<T>();
        AddVF(scoreLocal[scoreOffset], apeUb, coff_ * dealRowCount, dealColCount, scoreSingleRowCount,
              apeSingleRowCount);
        inputQueApe.FreeTensor(apeUb);
    }
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal, const Vec1SliceInfo &sliceInfo,
                                                   uint32_t dDealSize, uint32_t dBaseSize, uint32_t dStartIdx,
                                                   bool isApeFullLoad)
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
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                                       uint32_t preDealSeqCnt, uint32_t dealSeqCnt, uint32_t dStartIdx,
                                                       uint32_t dDealSize)
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
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::PadAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                              const Vec1SliceInfo &sliceInfo, uint32_t dBaseOffset, uint32_t dDealSize,
                                              uint32_t dBaseSize)
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

    uint64_t dstUbOffset = sliceInfo.compressoredScCnt * cmpRatio_ * dstSingleRowCount;
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                              const GlobalTensor<int32_t> &blockTableGm,
                                                                              const LocalTensor<T> &input,
                                                                              uint32_t batchIdx, uint32_t startSeqIdx,
                                                                              uint32_t endSeqIdx, uint32_t dDealSize,
                                                                              uint32_t dBaseSize, uint32_t stateIdx)
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
            // copyRowCount *= coff_;
            if (idInBlockTable != 0) { // 32
                uint64_t stateOffset =
                    idInBlockTable * constInfo_.stateCacheStrideDim0 + remainRowCnt * 2 * coff_ * constInfo_.headDim +
                    stateIdx * coff_ * constInfo_.headDim;
                uint64_t ubOffset = copyFinishRowCnt * coff_ * dBaseSize;
                DataCopyWithOutputQue(state[stateOffset], input[ubOffset], copyRowCount, dDealSize, coff_ * dBaseSize,
                                      coff_ * constInfo_.headDim * 2);
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
                                   stateIdx * coff_ * constInfo_.headDim;
            uint64_t ubOffset = copyFinishRowCnt * coff_ * dBaseSize;
            DataCopyWithOutputQue(state[stateOffset], input[ubOffset], copyRowCount,
                                  dDealSize, coff_ * dBaseSize, coff_ * constInfo_.headDim * 2);

            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
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
template <bool IS_SCORE>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
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
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                               const GlobalTensor<int32_t> &blockTableGm,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                               uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx)
{
    uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dBaseSize;

    if constexpr (COMP::cacheMode == CACHE_MODE::CYCLE) {
        uint32_t compressSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.bSeqUsed, cmpRatio_);
        uint32_t writeSeqStartIdx = compressSeqIdx > (coff_ - 1) * cmpRatio_ ?
                                    compressSeqIdx - (coff_ - 1) * cmpRatio_ : 0;
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, writeSeqStartIdx) - startSeqIdx) * coff_ * dBaseSize;
        startSeqIdx = max(startSeqIdx, writeSeqStartIdx);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm[dStartIdx], blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx,
                          startSeqIdx, endSeqIdx, dDealSize, dBaseSize, stateIdx);
        srcBaseOffset += dBaseSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm[dStartIdx], blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx,
                      startSeqIdx, endSeqIdx, dDealSize, dBaseSize, stateIdx);
}


template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::ReadState(
    const LocalTensor<T> &dstLocal, const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx)
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
            uint32_t startSeqIdx =
                sliceInfo.bStartPos < cmpRatio_ ?
                    0 :
                    Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, cmpRatio_) - cmpRatio_;
            uint32_t endSeqIdx =
                min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, cmpRatio_) -
                        cmpRatio_,
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
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::OverLap(
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
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::SaveToWorkSpace(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &cacheTcGm,
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
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, cmpRatio_)) *
        srcSingleRowElemNum;
    DataCopyWithOutputQue(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                          curSeqLen - max(curSeqLen - cmpRatio_, sliceInfo.bStartPos), dDealSize,
                          coff_ * dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::LoadFromWorkSpace(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm,
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx, uint32_t needDealTcSize)
{
    CompressorVec1SliceIterator overLapSliceIterator(tools_);
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal,
                                                                      uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * cmpRatio_;
    FaVectorApi::SoftmaxDnVF<T>(scoreLocal, scoreLocal, dDealSize, ReduceSize, tcDealSize, SOFTMAX_MIN_NUM, dDealSize);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                             const LocalTensor<T> &scoreLocal,
                                                                             const LocalTensor<T> &dstLocal,
                                                                             uint32_t tcDealSize, uint32_t dDealSize)
{
    MulReduceSumbaseVF(kvLocal, scoreLocal, dstLocal, coff_, cmpRatio_, dDealSize, tcDealSize);
}

template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::CopyOutVec1Res(const GlobalTensor<T> &resGm, const LocalTensor<T> &comperssoredUb,
                                                    uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint64_t outGmOffset = compressedCnt_ * constInfo_.headDim + dStartIdx;
    DataCopyAlignUbToGm(resGm[outGmOffset], comperssoredUb, compressTcSize, dDealSize, dDealSize, constInfo_.headDim);
}


template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::DealVec1BaseBlock(
    CompressorVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t dBaseOffset, uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = tmpBuf1.Get<T>();
    LocalTensor<T> kvLocal = tmpBuf2.Get<T>();
    OverLapScoreKv(scoreLocal, kvLocal, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dBaseOffset, dDealSize,
                   dBaseSize, dealSeqStartIdx, needDealTcSize);
    if (statisticInfo.compressorScCnt > 0) {
        SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        LocalTensor<T> comperssoredUb = outputQue2.AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        outputQue2.EnQue(comperssoredUb);
        outputQue2.DeQue<T>();
        GlobalTensor<T> resGm = vec1ResGm_;
        CopyOutVec1Res(resGm, comperssoredUb, statisticInfo.compressorScCnt, dStartIdx + dBaseOffset, dDealSize);
        outputQue2.FreeTensor(comperssoredUb);
    }
    compressedCnt_ += statisticInfo.compressorScCnt;
}


template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb,
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
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::CalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
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
                   constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
            inputQue2.FreeTensor(sinUb);
        }
        ropeSliceIterator.IteratorSlice();
    }
}


template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::CopyFinalResultOut(const LocalTensor<X_T> &cmpKvOutUb,
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


template <typename COMP>
__aicore__ inline void
CompressorBlockVectorFullLoad<COMP>::DealVec2BaseBlock(const Vec2SplitInfo &splitInfo,
                                                       CompressorVec2SliceIterator<COMP> &sliceIterator)
{
    Vec2SliceInfo &sliceInfo = sliceIterator.GetSlice();
    uint32_t needDealScSize = sliceIterator.GetNeedDealScSize();
    uint32_t computeSize = needDealScSize * constInfo_.headDim;
    int64_t inGmOffset = splitInfo.preScCnt * constInfo_.headDim;
    GlobalTensor<T> vec2InputGm = vec2InputGm_;
    // CopyIn
    LocalTensor<T> vec1ResUb = inputQue1.AllocTensor<T>();
    DataCopy(vec1ResUb, vec2InputGm[inGmOffset], computeSize);
    inputQue1.EnQue(vec1ResUb);
    inputQue1.DeQue<T>();

    // RmsNorm
    LocalTensor<T> normResUb = tmpBuf1.Get<T>();
    LocalTensor<T> tempLocal = tmpBuf2.Get<T>();
    PipeBarrier<PIPE_V>();
    MultRowRmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, needDealScSize);
    inputQue1.FreeTensor(vec1ResUb);


    // rope: 只对后RD进行rope; 将normResUb每行前headDim -
    // ropeHeadDim个元素cast到X_T，然后再与rope后的结果组合存到outputUb
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CalcGroupInfo(Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize =
        constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, constInfo_.batchSize)));
    if (constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), constInfo_.batchSize);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CalcTaskDistribution(Vec1SplitInfo &splitInfo)
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::UpdateIteratorState(Vec1SplitInfo &splitInfo)
{
    splitInfo.preCompressedCnt = 0;
    splitInfo.dealSeqStartIdx = splitInfo.preDealBatchNum * constInfo_.sSize;
    splitInfo.curBStart = splitInfo.preDealBatchNum;
    splitInfo.dealSeqCnt = splitInfo.dealBatchNum * constInfo_.sSize;
    splitInfo.curSStart = 0;
    totalCompressedCnt_ = 0;
    uint32_t endB = splitInfo.preDealBatchNum + splitInfo.dealBatchNum;
    for (uint32_t curB = 0; curB < constInfo_.batchSize; curB++) {
        uint32_t startPos = GetStartPos(curB);
        uint32_t seqLength = GetSeqLength(curB);
        if (curB < splitInfo.curBStart) {
            splitInfo.preCompressedCnt += (startPos + seqLength) / cmpRatio_ - startPos / cmpRatio_;
        } else {
            totalCompressedCnt_ += (startPos + seqLength) / cmpRatio_ - startPos / cmpRatio_;
        }
    }
    totalCompressedCnt_ += splitInfo.preCompressedCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
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
__aicore__ inline Vec1SplitInfo CompressorBlockVectorFullLoad<COMP>::SplitCoreV1()
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
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::ComputeVec1()
{
    Vec1SplitInfo splitInfo = SplitCoreV1();
    // 计算当前VecCore的任务量
    if (splitInfo.dealBatchNum == 0) {
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


    uint32_t cnt = constInfo_.sSize * splitInfo.dBaseSize * coff_;
    uint32_t singleLoopBatchNum = BUFFER_SIZE_BYTE_16K / (cnt * sizeof(T));
    uint32_t loopTimes = CeilDivT(splitInfo.dealBatchNum, singleLoopBatchNum);
    bool isApeFullLoad = coff_ * cmpRatio_ * splitInfo.dBaseSize * sizeof(T) <= BUFFER_SIZE_BYTE_16K;
    if (isApeFullLoad) {
        CopyInApe(baseOffset, splitInfo.dBaseSize);
    }
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint32_t curLoopBatchNum = min(singleLoopBatchNum, splitInfo.dealBatchNum - singleLoopBatchNum * idx);
        scoreUb = inputQue1.AllocTensor<T>();
        kvUb = scoreUb[BUFFER_SIZE_BYTE_16K / sizeof(T)];
        FromWokrSpaceToUb(scoreUb, scoreMm1ResGm_, splitInfo.dealSeqStartIdx, curLoopBatchNum * constInfo_.sSize,
                          baseOffset, splitInfo.dBaseSize);
        FromWokrSpaceToUb(kvUb, kvMm1ResGm_, splitInfo.dealSeqStartIdx, curLoopBatchNum * constInfo_.sSize, baseOffset,
                          splitInfo.dBaseSize);
        inputQue1.EnQue(scoreUb);
        inputQue1.DeQue<T>();
        splitInfo.dealTcNum = 0;
        uint32_t curLoopCompressedCnt = 0;
        for (uint32_t curB = splitInfo.curBStart; curB < splitInfo.curBStart + curLoopBatchNum; curB++) {
            uint32_t startPos = GetStartPos(curB);
            uint32_t seqLength = GetSeqLength(curB);
            splitInfo.dealTcNum +=
                CeilDivT(startPos + seqLength, cmpRatio_) - (startPos / cmpRatio_);
            curLoopCompressedCnt += (startPos + seqLength) / cmpRatio_ - startPos / cmpRatio_;
        }
        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
        sliceIterator.SetNeedDealTcSize(splitInfo.dealTcNum);
        sliceIterator.SetDealedTcCnt(0U);
        Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();
        while (!sliceIterator.IsEnd()) {
            sliceIterator.GetSlice();
            SaveState(kvUb, stateCacheGm_, stateBlockTableGm_, sliceInfo, baseOffset, splitInfo.dBaseSize,
                      splitInfo.dBaseSize, kvStateIdx_);

            AddApeToScore(scoreUb, sliceInfo, splitInfo.dBaseSize, splitInfo.dBaseSize, baseOffset, isApeFullLoad);
            SaveState(scoreUb, stateCacheGm_, stateBlockTableGm_, sliceInfo, baseOffset, splitInfo.dBaseSize,
                      splitInfo.dBaseSize, scoreStateIdx_);
            sliceIterator.IteratorSlice();
        }

        if (curLoopCompressedCnt == 0) {
            inputQue1.FreeTensor(scoreUb);
            continue;
        }
        for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
            uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;
            loopInfo.dLoopIdx = dLoopIdx;

            sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
            compressedCnt_ = splitInfo.preCompressedCnt;
            for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcNum; tcIdx += splitInfo.tcSplitSize) {
                uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcNum - tcIdx);

                loopInfo.isCoreLoopFirst = tcIdx == 0;
                loopInfo.isCoreLoopLast = tcIdx + splitInfo.tcSplitSize >= splitInfo.dealTcNum;
                // 处理单个切块
                sliceIterator.SetNeedDealTcSize(actDealTcSize);
                sliceIterator.SetDealedTcCnt(0U);
                DealVec1BaseBlock(sliceIterator, loopInfo, baseOffset, dLoopIdx * splitInfo.dSplitSize,
                                  splitInfo.dSplitSize, splitInfo.dBaseSize, splitInfo.dealSeqStartIdx);
            }
        }
        inputQue1.FreeTensor(scoreUb);
        splitInfo.curBStart += curLoopBatchNum;
        splitInfo.dealSeqStartIdx += curLoopBatchNum * constInfo_.sSize;
        splitInfo.preCompressedCnt += curLoopCompressedCnt;
    }
}


template <typename COMP>
__aicore__ inline Vec2SplitInfo CompressorBlockVectorFullLoad<COMP>::SplitCoreV2()
{
    Vec2SplitInfo splitInfo;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;

    if (blockIdx < (totalCompressedCnt_ % aiCoreNum)) {
        splitInfo.dealScNum = totalCompressedCnt_ / aiCoreNum + 1;
        splitInfo.preScCnt = splitInfo.dealScNum * blockIdx;
    } else if (blockIdx < aiCoreNum) {
        splitInfo.dealScNum = totalCompressedCnt_ / aiCoreNum;
        splitInfo.preScCnt = splitInfo.dealScNum * blockIdx + totalCompressedCnt_ % aiCoreNum;
    } else {
        splitInfo.dealScNum = 0;
        splitInfo.preScCnt = totalCompressedCnt_;
    }

    if (splitInfo.dealScNum == 0) {
        return splitInfo;
    }

    uint32_t preScCnt = splitInfo.preScCnt;
    for (uint32_t curB = 0; curB < constInfo_.batchSize; curB++) {
        uint32_t startPos = GetStartPos(curB);
        uint32_t seqUsed = GetSeqUsed(curB);
        uint32_t curScNum = (startPos + seqUsed) / cmpRatio_ - startPos / cmpRatio_;
        if (preScCnt < curScNum) {
            splitInfo.curBStart = curB;
            splitInfo.curScStart = preScCnt;
            break;
        }
        preScCnt -= curScNum;
    }

    splitInfo.dealedScCnt = 0;

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::ComputeVec2()
{
    Vec2SplitInfo splitInfo = SplitCoreV2();
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
        DealVec2BaseBlock(splitInfo, sliceIterator);
        splitInfo.preScCnt += curLoopScNum;
    }
}


} // namespace Compressor
#endif // COMPRESSOR_BLOCK_VECTOR_PREF_H
