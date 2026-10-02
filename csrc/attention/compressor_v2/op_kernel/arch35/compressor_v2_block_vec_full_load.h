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

#include "compressor_v2_block_vec.h"
#include "compressor_v2_tools.h"
#include "vf/vf_softmax_compressor_v2.h"
#include "vf/vf_add_compressor_v2.h"
#include "vf/vf_mul_compressor_v2.h"
#include "limits"

using namespace AscendC;

namespace CompressorV2 {

template <typename COMP>
class CompressorV2BlockVectorFullLoad : public CompressorV2BlockVector<COMP> {
public:
    using Base = CompressorV2BlockVector<COMP>;
    using T = typename Base::T;
    using X_T = typename Base::X_T;

    __aicore__ inline CompressorV2BlockVectorFullLoad(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ComputeVec1();

private:
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
    __aicore__ inline void DealVec1BaseBlock(CompressorV2Vec1SliceIterator<COMP> &sliceIterator,
                                             const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
                                             uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx);
    __aicore__ inline void CalcGroupInfo(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1();

    uint32_t totalCompressedCnt_ = 0;
    uint32_t kvStateIdx_ = 0;
    uint32_t scoreStateIdx_ = 1;
    LocalTensor<T> scoreUb;
    LocalTensor<T> kvUb;
};

template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->inputQue3, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->tmpBuff2, BUFFER_SIZE_BYTE_32K);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::OverLap(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    this->template ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx + dBaseOffset, dDealSize,
                                       static_cast<uint32_t>(IS_SCORE));

    if (sliceInfo.compressTcSize > 0) {
        this->PadAlign(dstLocal, srcLocal, sliceInfo, dBaseOffset, dDealSize, dBaseSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm;
            this->LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx,
                                    globalSeqIdx, dDealSize);
        }
    }
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx, uint32_t needDealTcSize)
{
    CompressorV2Vec1SliceIterator overLapSliceIterator(this->tools_);
    overLapSliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = this->scoreMm1ResGm_;
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_,
                      this->scoreCacheTcGm_, overLapSliceInfo, loopInfo, dStartIdx, dBaseOffset,
                      originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize, dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }

    GlobalTensor<T> kvDBMm1ResGm = this->kvMm1ResGm_;
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_, this->kvCacheTcGm_,
                       overLapSliceInfo, loopInfo, dStartIdx, dBaseOffset,
                       originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize, dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::DealVec1BaseBlock(
    CompressorV2Vec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t dBaseOffset, uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = this->tmpBuff1.template Get<T>();
    LocalTensor<T> kvLocal = this->tmpBuff2.template Get<T>();
    OverLapScoreKv(scoreLocal, kvLocal, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dBaseOffset, dDealSize,
                   dBaseSize, dealSeqStartIdx, needDealTcSize);
    if (statisticInfo.compressorScCnt > 0) {
        this->SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> comperssoredUb = scoreLocal;
        PipeBarrier<PIPE_V>();
        this->KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        this->CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.compressorScCnt,
                                     dStartIdx + dBaseOffset, dDealSize);
    }
    this->compressedCnt_ += statisticInfo.compressorScCnt;
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::CalcGroupInfo(Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = this->constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize =
        this->constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, this->constInfo_.batchSize)));
    // 32B(8个FP32)对齐的UB列窗口上限（同 NORMAL 模板）：dBaseSize 超过它时
    // this->CalcTilingStrategy 的 dSplitSize = dBaseSize/dLoopCount 整数除法会切出非32B
    // 对齐的列窗口（DataCopy blockLen/srcGap 整数除法错位 → 数据错乱）。
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (this->cmpRatio_ * this->coff_ * sizeof(T));
    splitInfo.dBaseSize = min(splitInfo.dBaseSize, FloorPow2(Trunc(maxDealColNum, BlockElementNum<T>())));
    if (this->constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(X_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<X_T>());
    splitInfo.vec1GroupSize = this->constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum =
        min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), this->constInfo_.batchSize);
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::CalcTaskDistribution(Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t totalDealBatchNum = this->constInfo_.batchSize;

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
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::UpdateIteratorState(Vec1SplitInfo &splitInfo)
{
    splitInfo.preCompressedCnt = 0;
    splitInfo.dealSeqStartIdx = splitInfo.preDealBatchNum * this->constInfo_.sSize;
    splitInfo.curBStart = splitInfo.preDealBatchNum;
    splitInfo.dealSeqCnt = splitInfo.dealBatchNum * this->constInfo_.sSize;
    splitInfo.curSStart = 0;
    totalCompressedCnt_ = 0;
    uint32_t endB = splitInfo.preDealBatchNum + splitInfo.dealBatchNum;
    for (uint32_t curB = 0; curB < this->constInfo_.batchSize; curB++) {
        uint32_t startPos = this->GetStartPos(curB);
        uint32_t seqLength = this->GetSeqLength(curB);
        if (curB < splitInfo.curBStart) {
            splitInfo.preCompressedCnt += (startPos + seqLength) / this->cmpRatio_ - startPos / this->cmpRatio_;
        } else {
            totalCompressedCnt_ += (startPos + seqLength) / this->cmpRatio_ - startPos / this->cmpRatio_;
        }
    }
    totalCompressedCnt_ += splitInfo.preCompressedCnt;
}
template <typename COMP>
__aicore__ inline Vec1SplitInfo CompressorV2BlockVectorFullLoad<COMP>::SplitCoreV1()
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
    this->CalcTilingStrategy(splitInfo);

    return splitInfo;
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorFullLoad<COMP>::ComputeVec1()
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

    CompressorV2Vec1SliceIterator sliceIterator(this->tools_);
    sliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    // 切块循环
    uint64_t baseOffset = loopInfo.coreColIdx * splitInfo.dBaseSize;

    uint32_t cnt = this->constInfo_.sSize * splitInfo.dBaseSize * this->coff_;
    uint32_t singleLoopBatchNum = BUFFER_SIZE_BYTE_16K / (cnt * sizeof(T));
    uint32_t loopTimes = CeilDivT(splitInfo.dealBatchNum, singleLoopBatchNum);
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint32_t curLoopBatchNum = min(singleLoopBatchNum, splitInfo.dealBatchNum - singleLoopBatchNum * idx);
        scoreUb = this->inputQue1.template AllocTensor<T>();
        kvUb = scoreUb[BUFFER_SIZE_BYTE_16K / sizeof(T)];
        this->FromWokrSpaceToUb(scoreUb, this->scoreMm1ResGm_, splitInfo.dealSeqStartIdx,
                                curLoopBatchNum * this->constInfo_.sSize, baseOffset, splitInfo.dBaseSize);
        this->FromWokrSpaceToUb(kvUb, this->kvMm1ResGm_, splitInfo.dealSeqStartIdx,
                                curLoopBatchNum * this->constInfo_.sSize, baseOffset, splitInfo.dBaseSize);
        this->inputQue1.template EnQue(scoreUb);
        this->inputQue1.template DeQue<T>();
        splitInfo.dealTcNum = 0;
        uint32_t curLoopCompressedCnt = 0;
        for (uint32_t curB = splitInfo.curBStart; curB < splitInfo.curBStart + curLoopBatchNum; curB++) {
            uint32_t startPos = this->GetStartPos(curB);
            uint32_t seqLength = this->GetSeqLength(curB);
            uint32_t seqUsed = this->GetSeqUsed(curB);
            splitInfo.dealTcNum += CeilDivT(startPos + seqLength, this->cmpRatio_) - (startPos / this->cmpRatio_);
            curLoopCompressedCnt += (startPos + seqUsed) / this->cmpRatio_ - startPos / this->cmpRatio_;
        }
        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
        sliceIterator.SetNeedDealTcSize(splitInfo.dealTcNum);
        sliceIterator.SetDealedTcCnt(0U);
        Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();
        while (!sliceIterator.IsEnd()) {
            sliceIterator.GetSlice();
            this->SaveState(kvUb, this->stateCacheGm_, this->stateBlockTableGm_, sliceInfo, baseOffset,
                            splitInfo.dBaseSize, splitInfo.dBaseSize, kvStateIdx_);

            this->SaveState(scoreUb, this->stateCacheGm_, this->stateBlockTableGm_, sliceInfo, baseOffset,
                            splitInfo.dBaseSize, splitInfo.dBaseSize, scoreStateIdx_);
            sliceIterator.IteratorSlice();
        }

        if (curLoopCompressedCnt == 0) {
            this->inputQue1.template FreeTensor(scoreUb);
            continue;
        }
        for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
            uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;
            loopInfo.dLoopIdx = dLoopIdx;

            sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
            this->compressedCnt_ = splitInfo.preCompressedCnt;
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
        this->inputQue1.template FreeTensor(scoreUb);
        splitInfo.curBStart += curLoopBatchNum;
        splitInfo.dealSeqStartIdx += curLoopBatchNum * this->constInfo_.sSize;
        splitInfo.preCompressedCnt += curLoopCompressedCnt;
    }
}
} // namespace CompressorV2

#endif // COMPRESSOR_BLOCK_VEC_FULL_LOAD_H
