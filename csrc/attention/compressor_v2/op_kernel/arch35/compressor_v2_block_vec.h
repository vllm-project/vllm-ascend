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

#include "compressor_v2_comm_arch35.h"
#include "compressor_v2_tools.h"
#include "vf/vf_softmax_compressor_v2.h"
#include "vf/vf_add_compressor_v2.h"
#include "vf/vf_mul_compressor_v2.h"
#include "limits"

using namespace AscendC;

namespace CompressorV2 {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP>
class CompressorV2BlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = BUFFER_SIZE_BYTE_32K; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float SOFTMAX_MIN_NUM = -std::numeric_limits<float>::infinity();
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorV2BlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const CompressorV2Tools<COMP> &tools);
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                __gm__ uint8_t *stateCache, __gm__ uint8_t *stateBlockTable, __gm__ uint8_t *cuSeqlens,
                                __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut);
    // =================================资源管理=================================
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm);

protected:
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;

protected:
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
    __aicore__ inline void FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                             uint32_t preDealSeqCnt, uint32_t dealSeqCnt, uint32_t dStartIdx,
                                             uint32_t dDealSize);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                             const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                             uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize, uint32_t stateIdx);
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
    __aicore__ inline void PadAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dBaseOffset, uint32_t dDealSize,
                                    uint32_t dBaseSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb, const Vec1SliceInfo &sliceInfo,
                                                  uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx);
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
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    CompressorV2Tools<COMP> tools_;
    ConstInfo constInfo_ = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    // TBuf<TPosition::VECIN> mm1ResUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TQue<QuePosition::VECIN, 1> inputQue3;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;
};

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::InitParams(const ConstInfo &constInfo,
                                                                 const CompressorV2Tools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    coff_ = static_cast<uint32_t>(COMP::coff);
    cmpRatio_ = constInfo.cmpRatio;
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv,
                                                           __gm__ uint8_t *wGate, __gm__ uint8_t *stateCache,
                                                           __gm__ uint8_t *stateBlockTable, __gm__ uint8_t *cuSeqlens,
                                                           __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                                           __gm__ uint8_t *cmpKvOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T *)stateCache);
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
__aicore__ inline void CompressorV2BlockVector<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::FreeEventID()
{}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm,
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
__aicore__ inline uint32_t CompressorV2BlockVector<COMP>::GetSeqUsed(uint32_t bIdx)
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
__aicore__ inline uint32_t CompressorV2BlockVector<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorV2BlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorV2BlockVector<COMP>::DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::DataCopyWithOutputQue(
    const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::DataCopyWithInputQue(const LocalTensor<O> &dstLocal,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::AddMultiDataToUb(
    const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm, uint32_t dealRowCount, uint32_t dealColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount, uint32_t repeatTimes, uint64_t offset)
{
    uint32_t cnt = dealRowCount * dstSingleRowCount;
    if (cnt == 0) {
        return;
    }
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
__aicore__ inline void CompressorV2BlockVector<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal,
                                                                        const GlobalTensor<T> &srcGm,
                                                                        uint32_t preDealSeqCnt, uint32_t dealSeqCnt,
                                                                        uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = (uint64_t)preDealSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    if (constInfo_.kBaseNum == 1) {
        DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
    } else {
        AddMultiDataToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount,
                         constInfo_.kBaseNum, constInfo_.mm1KvResSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::SaveToWorkSpace(const LocalTensor<T> &srcLocal,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::LoadFromWorkSpace(
    const LocalTensor<T> &dstLocal, const GlobalTensor<T> &cacheTcGm, const GlobalTensor<T> &srcGm,
    const LocalTensor<T> &srcLocal, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
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
__aicore__ inline void CompressorV2BlockVector<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                        const GlobalTensor<int32_t> &blockTableGm,
                                                                        const LocalTensor<T> &input, uint32_t batchIdx,
                                                                        uint32_t startSeqIdx, uint32_t endSeqIdx,
                                                                        uint32_t dStartIdx, uint32_t dDealSize,
                                                                        uint32_t dBaseSize, uint32_t stateIdx)
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
            if (idInBlockTable != 0) { // 32
                uint64_t stateOffset = idInBlockTable * constInfo_.stateCacheStrideDim0 +
                                       remainRowCnt * STATE_INTERLEAVE_FACTOR * coff_ * constInfo_.headDim +
                                       stateIdx * coff_ * constInfo_.headDim + dStartIdx;
                DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dBaseSize], copyRowCount,
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
            DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dBaseSize], copyRowCount,
                                  dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * STATE_INTERLEAVE_FACTOR);

            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::SaveState(
    const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize, uint32_t stateIdx)
{
    uint64_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dBaseSize;

    if constexpr (COMP::cacheMode == CACHE_MODE::RING_BUFFER) {
        // 写窗口 = 本 batch 本次调用序列末尾 min(bSeqUsed, blockSize) 个 token
        uint64_t seqEnd = sliceInfo.bStartPos + sliceInfo.bSeqUsed;
        uint64_t writeSeqStartIdx =
            seqEnd > (uint64_t)constInfo_.blockSize ? seqEnd - (uint64_t)constInfo_.blockSize : 0ULL;
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, writeSeqStartIdx) - startSeqIdx) * coff_ * dBaseSize;
        startSeqIdx = max(startSeqIdx, writeSeqStartIdx);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                          dStartIdx, dDealSize, dBaseSize, stateIdx);
        srcBaseOffset += dBaseSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx,
                      dDealSize, dBaseSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorV2BlockVector<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::ReadState(const LocalTensor<T> &dstLocal,
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
        uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * cmpRatio_ * coff_ * dDealSize;
        if constexpr (COMP::coff == CompressorV2::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // 填充左边
    if constexpr (COMP::coff == CompressorV2::COFF::OVERLAP) {
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
                                       Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, (uint64_t)cmpRatio_) - cmpRatio_;
            uint32_t endSeqIdx = min(
                Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, (uint64_t)cmpRatio_) - cmpRatio_,
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
__aicore__ inline void CompressorV2BlockVector<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal, uint32_t tcDealSize,
                                                                uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * cmpRatio_;
    FaVectorApi::SoftmaxDnVF<T>(scoreLocal, scoreLocal, dDealSize, ReduceSize, tcDealSize, SOFTMAX_MIN_NUM, dDealSize);
}

template <typename COMP>

__aicore__ inline void CompressorV2BlockVector<COMP>::PadAlign(const LocalTensor<T> &dstLocal,
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
    uint32_t srcSingleRowElemNum = dBaseSize * this->coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * this->cmpRatio_ - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize * this->coff_; // left和right在seq方向是交错存储的
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum + dBaseOffset;

    uint64_t dstUbOffset = sliceInfo.compressoredScCnt * this->cmpRatio_ * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + this->cmpRatio_) * dstSingleRowCount;
        this->DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                                  copyRowCount - min(copyRowCount, this->cmpRatio_), copyColCount, srcSingleRowCount,
                                  dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dBaseSize;
    }
    // 右侧
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    this->DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount,
                              srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                       const LocalTensor<T> &scoreLocal,
                                                                       const LocalTensor<T> &dstLocal,
                                                                       uint32_t tcDealSize, uint32_t dDealSize)
{
    MulReduceSumbaseVF(kvLocal, scoreLocal, dstLocal, coff_, cmpRatio_, dDealSize, tcDealSize);
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::CopyOutVec1ResToOutput(const LocalTensor<T> &comperssoredUb,
                                                                             const Vec1SliceInfo &sliceInfo,
                                                                             uint32_t compressTcSize,
                                                                             uint32_t dStartIdx, uint32_t dDealSize)
{
    LocalTensor<X_T> outputUb = outputQue2.AllocTensor<X_T>();
    Cast(outputUb, comperssoredUb, RoundMode::CAST_ROUND, compressTcSize * dDealSize);
    outputQue2.EnQue(outputUb);
    outputQue2.DeQue<X_T>();
    if constexpr (COMP::xLayout == X_LAYOUT::BSH) {
        uint32_t bOutputScLen = CeilDivT(GetSeqLength(sliceInfo.bIdx), cmpRatio_);
        uint32_t bIdx = sliceInfo.bIdx;
        uint32_t sIdx = sliceInfo.sIdx;
        uint64_t ubOffset = 0;
        while (compressTcSize > 0) {
            uint32_t bStartPos = GetStartPos(bIdx);
            uint32_t preScSize = (sIdx + bStartPos) / cmpRatio_;
            uint32_t totalScSize = (GetSeqUsed(bIdx) + bStartPos) / cmpRatio_;
            if (preScSize < totalScSize) {
                uint32_t curScSize = min(compressTcSize, (totalScSize - preScSize));
                uint64_t padScIdx = bIdx * bOutputScLen + preScSize - (bStartPos / cmpRatio_);
                uint64_t outGmOffset = padScIdx * constInfo_.headDim + dStartIdx;
                DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb[ubOffset], curScSize, dDealSize, dDealSize,
                                    constInfo_.headDim);
                compressTcSize -= curScSize;
                ubOffset += curScSize * dDealSize;
            }
            bIdx++;
            sIdx = 0;
        }
    } else {
        uint64_t outGmOffset = compressedCnt_ * constInfo_.headDim + dStartIdx;
        DataCopyAlignUbToGm(cmpKvOutGm_[outGmOffset], outputUb, compressTcSize, dDealSize, dDealSize,
                            constInfo_.headDim);
    }
    outputQue2.FreeTensor(outputUb);
}

template <typename COMP>
__aicore__ inline void CompressorV2BlockVector<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
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

} // namespace CompressorV2
#endif // COMPRESSOR_BLOCK_VEC_H
