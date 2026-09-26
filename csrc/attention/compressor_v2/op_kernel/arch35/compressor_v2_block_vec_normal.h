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
 * \file compressor_block_vec_normal.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_NORMAL_H
#define COMPRESSOR_BLOCK_VEC_NORMAL_H

#include "compressor_v2_block_vec.h"
#include "compressor_v2_tools.h"
#include "vf/vf_softmax_compressor_v2.h"
#include "vf/vf_add_compressor_v2.h"
#include "vf/vf_mul_compressor_v2.h"
#include "limits"

using namespace AscendC;

namespace CompressorV2 {

template <typename COMP>
class CompressorV2BlockVectorNormal : public CompressorV2BlockVector<COMP> {
public:
    using Base = CompressorV2BlockVector<COMP>;
    using T = typename Base::T;
    using X_T = typename Base::X_T;

    __aicore__ inline CompressorV2BlockVectorNormal(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);

private:
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                   const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                   const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                   uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const Vec1RunInfo &info, const LoopInfo &loopInfo,
                                          const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo,
                                          uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t needDealTcSize);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info,
                                             CompressorV2Vec1SliceIterator<COMP> &sliceIterator,
                                             const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    MSplitInfo mSplitInfo = {};
};

template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->inputQue2, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->inputQue3, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->tmpBuff2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    PipeBarrier<PIPE_V>();
}
template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::OverLap(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if constexpr (IS_SCORE) {
        PipeBarrier<PIPE_V>();
    }
    this->SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, dDealSize,
                    static_cast<uint32_t>(IS_SCORE));
    this->template ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                                       static_cast<uint32_t>(IS_SCORE));

    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t nextC1V1DbIdx = (info.c1v1DbIdx + 1) % this->constInfo_.dbWorkspaceRatio;
        GlobalTensor<T> nextCacheTcGm = cacheTcGm[nextC1V1DbIdx * this->cmpRatio_ * this->constInfo_.headDim];
        this->SaveToWorkSpace(srcLocal, nextCacheTcGm, sliceInfo, loopInfo, dStartIdx, dDealSize);
    }
    if (sliceInfo.compressTcSize > 0) {
        this->PadAlign(dstLocal, srcLocal, sliceInfo, 0, dDealSize, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm[info.c1v1DbIdx * this->cmpRatio_ * this->constInfo_.headDim];
            this->LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx,
                                    globalSeqIdx, dDealSize);
        }
    }
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
    uint32_t dBaseSize, uint32_t needDealTcSize)
{
    CompressorV2Vec1SliceIterator overLapSliceIterator(this->tools_);
    overLapSliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = this->scoreMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    LocalTensor<T> scoreUb = this->inputQue1.template AllocTensor<T>();
    this->FromWokrSpaceToUb(scoreUb, scoreDBMm1ResGm, originSliceInfo.dealedSeqCnt, statisticInfo.dealSeqCnt, dStartIdx,
                            dDealSize);
    this->inputQue1.template EnQue(scoreUb);
    this->inputQue1.template DeQue<T>();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_,
                      this->scoreCacheTcGm_, info, overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt,
                      dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    this->inputQue1.template FreeTensor(scoreUb);

    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0 &&
            (!loopInfo.isCoreRowFirst || !loopInfo.isCoreLoopFirst)) {
            PipeBarrier<PIPE_V>();
        }
    }

    GlobalTensor<T> kvDBMm1ResGm = this->kvMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    LocalTensor<T> kvUb = this->inputQue1.template AllocTensor<T>();
    this->FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo.dealedSeqCnt, statisticInfo.dealSeqCnt, dStartIdx,
                            dDealSize);
    this->inputQue1.template EnQue(kvUb);
    this->inputQue1.template DeQue<T>();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_, this->kvCacheTcGm_,
                       info, overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    this->inputQue1.template FreeTensor(kvUb);

    PipeBarrier<PIPE_V>();
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::DealVec1BaseBlock(
    const Vec1RunInfo &info, CompressorV2Vec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo,
    uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = this->tmpBuff1.template Get<T>();
    LocalTensor<T> kvLocal = this->tmpBuff2.template Get<T>();

    OverLapScoreKv(scoreLocal, kvLocal, info, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dDealSize, dBaseSize,
                   needDealTcSize);

    if (statisticInfo.compressorScCnt > 0) {
        this->SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<T> comperssoredUb = scoreLocal;
        this->KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        this->CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.compressorScCnt, dStartIdx,
                                     dDealSize);
    }
    this->compressedCnt_ += statisticInfo.compressorScCnt;
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::CalcGroupInfo(const Vec1RunInfo &info,
                                                                          Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = this->constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize =
        this->constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, info.dealTcNum)));
    // 32B(8个FP32)对齐的UB列窗口上限。dBaseSize 超过它时 this->CalcTilingStrategy 会走
    // dSplitSize = dBaseSize/dLoopCount 整数除法拆分，切出的列窗口非32B对齐
    // （DataCopy 的 blockLen/srcGap 整数除法错位 → 行间源偏移错误 → 数据错乱）。
    // clamp 到“不超过 maxDealColNum 的最大2的幂”后，dSplitSize 恒等于 dBaseSize
    // （2的幂，天然8元素对齐），不再进入拆分分支。
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (this->cmpRatio_ * this->coff_ * sizeof(T));
    splitInfo.dBaseSize = min(splitInfo.dBaseSize, FloorPow2(Trunc(maxDealColNum, BlockElementNum<T>())));
    if (this->constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(X_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<X_T>());
    splitInfo.vec1GroupSize = this->constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), info.dealTcNum);
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::CalcTaskDistribution(const Vec1RunInfo &info,
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
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::UpdateIteratorState(const Vec1RunInfo &info,
                                                                                Vec1SplitInfo &splitInfo)
{
    CompressorV2Vec1SliceIterator sliceIterator(this->tools_);
    sliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
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
__aicore__ inline Vec1SplitInfo CompressorV2BlockVectorNormal<COMP>::SplitCoreV1(const Vec1RunInfo &info)
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
    this->CalcTilingStrategy(splitInfo);

    return splitInfo;
}
template <typename COMP>
__aicore__ inline void CompressorV2BlockVectorNormal<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    if (info.dealTcNum == 0) {
        return;
    }
    uint32_t preCompressedCnt = this->compressedCnt_;
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    // 计算当前VecCore的任务量
    if (splitInfo.dealTcSize == 0) {
        this->compressedCnt_ += splitInfo.totalCompressedCnt;
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
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;

        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        this->compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
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
    this->compressedCnt_ = preCompressedCnt + splitInfo.totalCompressedCnt;
}
} // namespace CompressorV2

#endif // COMPRESSOR_BLOCK_VEC_NORMAL_H
