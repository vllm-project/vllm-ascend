/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../compressor_comm.h"
#include "arch32.h"

using namespace AscendC;

namespace Compressor 
{
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
template <typename COMP> class CompressorBlockVector {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVector(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init( 
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Compressor::RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline void GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize);
    __aicore__ inline void CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm, GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void ComputeVec2(const Compressor::RunInfo &info);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
        const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dDealSize);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
        uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dDealSize);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> vec2InputGm_;
    GlobalTensor<T> preMm1ResGm_;
    GlobalTensor<T> curMm1ResGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetBsLength(uint32_t index);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd, uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize, uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const RunInfo &info, BlockInfo &blockInfo, uint32_t startTcIdx, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void UpdateBlockInfo(BlockInfo &blockInfo);
    __aicore__ inline void CopyInApe(LocalTensor<T> apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(LocalTensor<T> &scoreLocal, LocalTensor<T> &apeUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void DataCopyAlignUbToUb(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal,
        uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void DataCopyAlignGmToUb(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm,
        uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void OverLapScore(LocalTensor<T> scoreLocal, const RunInfo &info, 
        uint32_t tcStartIdx, uint32_t tcDealSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void OverLapKv(LocalTensor<T> kvLocal, const RunInfo &info, 
        uint32_t tcStartIdx, uint32_t tcDealSize, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void UpdateState(LocalTensor<T> kvLocal, LocalTensor<T> scoreLocal,
        uint32_t startTcIdx, const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(LocalTensor<T> &scoreLocal, LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(LocalTensor<T> &kvLocal, LocalTensor<T>& scoreLocal, LocalTensor<T> &dstLocal, LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutVec1Res(const RunInfo &info, LocalTensor<T> comporessedUb, uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);

    __aicore__ inline void SplitCoreV2(const Compressor::RunInfo& info);
    __aicore__ inline void CopyFinalResultOut(const Compressor::RunInfo& info, const LocalTensor<X_T> &cmpKvOutUb,
        uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void MultRowRmsNorm(LocalTensor<T> &normResUb, LocalTensor<T> &vec1ResUb, LocalTensor<T> &normWeightUb,
    LocalTensor<T> &tempLocal, const Compressor::RunInfo& info, uint32_t dealRowCount);
    __aicore__ inline void CalRope(const Compressor::RunInfo& info, LocalTensor<X_T> &outputUb, LocalTensor<T> &normResUb,
        uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void SaveLeftFirst(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
        const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void SaveState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
        const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void ReadState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
        const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize);
    // static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    // static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
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
    uint32_t v2MBaseSize = 16; // Tc块数量：32 * 1024 / (512 * 4)
    uint32_t v2TcStartIdx = 0U;
    uint32_t v2TcEndIdx = 0U;
    uint32_t mmResColSize_ = 128;
    int64_t vec1ResGmStart = 0U;
    uint32_t usedCoreNum = 16;
    uint32_t OutputBStartIdx, OutputSStartIdx, OutputSize;
    uint32_t mmResBaseOffset_ = 0;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> scoreBlockTableGm_;
    GlobalTensor<T> kvStateGm_;
    GlobalTensor<T> scoreStateGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    LocalTensor<T> mm1ResTensor;
    LocalTensor<T> leftStateTensor;
    LocalTensor<T> rightStateTensor;
    LocalTensor<T> normWeightUb;
    LocalTensor<uint32_t> gatherOffsetCastUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    TBuf<TPosition::VECIN> normWeightBuf;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut)
{
    kvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)kvBlockTable);
    scoreBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)scoreBlockTable);
    kvStateGm_.SetGlobalBuffer((__gm__ T *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ T *)scoreState);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_2K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(gatherOffsetBuf, BUFFER_SIZE_BYTE_2K);
    normWeightUb = normWeightBuf.Get<T>();
    LocalTensor<X_T> normweightInUb = inputQue2.AllocTensor<X_T>();
    LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf.Get<int32_t>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    inputQue2.EnQue(normweightInUb);
    inputQue2.DeQue<X_T>();
    Cast(normWeightUb, normweightInUb, RoundMode::CAST_NONE, constInfo_.headDim);
    inputQue2.FreeTensor(normweightInUb);
    if constexpr (COMP::rotaryMode == Compressor::ROTARY_MODE::INTERLEAVE) {
        SetGatherSrcOffset<float>(gatherOffsetUb, constInfo_.headDim);
    }
    gatherOffsetCastUb = gatherOffsetUb.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::AllocEventID()
{
    // SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    // SetFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::FreeEventID()
{
    // WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    // WaitFlag<HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm,
                                                                         GlobalTensor<T> curMm1ResGm,
                                                                         GlobalTensor<T> vec1ResGm,
                                                                         GlobalTensor<T> vec2InputGm) {
    this->preMm1ResGm_ = preMm1ResGm;
    this->curMm1ResGm_ = curMm1ResGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec2InputGm_ = vec2InputGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed) {
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
    if (isExistStartPos) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return sequsedGm_.GetValue(bIdx);
    } else if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBsLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index);
    } else {
        return index * constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength_ - headSize + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVector<COMP>::GetScSize() {
    uint32_t curBasicNum = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    return curBasicNum;
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd) {
    uint32_t accScSize = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;

        curActSeqLength_ = GetSeqLength(bIdx);
        curStartPos_ = GetStartPos(bIdx);
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            accScSize += GetScSize() - scStart;  // 需要减去本batch已经处理完的sc
            if (accScSize >= dealScSize) {
                scEnd = scStart + dealScSize;
                return;
            }
        } else {
            uint32_t curBasicNum = GetScSize();
            uint32_t curBasicNumEnd = dealScSize - accScSize;
            
            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = curBasicNumEnd;
                return;
            }
            accScSize += curBasicNum;
        }
    }
}

// 根据sc的开始索引计算vec输出时的b、sc的索引
template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize)
{
    outputScSize = v2TcEnd - v2TcStart;
    uint32_t scEnd = 0;
    uint32_t bEnd = 0;
    CalcScEndIdx(bStart, scStart, v2TcStart, bEnd, scEnd);
    outputSStart = scEnd;
    outputBStart = bEnd;
    // 处理跳batch
    curActSeqLength_ = GetSeqLength(bEnd);
    curStartPos_ = GetStartPos(bEnd);
    uint32_t curScSize = GetScSize();
    if (curScSize == scEnd) {
        outputSStart = 0;
        outputBStart++;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyInApe(LocalTensor<T> apeUb,
    uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t copyRowCount = constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = ((uint32_t)COMP::coff) * dDealSize;
    uint32_t srcSingleRowCount = ((uint32_t)COMP::coff) * constInfo_.headDim;

    uint32_t gmOffset = (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize + dStartIdx;
    uint32_t dstUbOffset = 0;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        DataCopyAlignGmToUb(apeUb[dstUbOffset], apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);

        gmOffset += constInfo_.headDim;
        dstUbOffset += dDealSize;
    }
    DataCopyAlignGmToUb(apeUb[dstUbOffset], apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::AddApeToScore(LocalTensor<T> &scoreLocal, LocalTensor<T> &apeUb,
    uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio :
        constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        Add(scoreLocal[r * rCnt], scoreLocal[r * rCnt], apeUb, rCnt);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlignUbToUb(LocalTensor<T> dstLocal, LocalTensor<T> srcLocal,
    uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / (32 / sizeof(T));
    intriParams.dstStride = (dstSingleRowCount - copyColCount) / (32 / sizeof(T));
    intriParams.srcStride = (srcSingleRowCount - copyColCount) / (32 / sizeof(T));
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::DataCopyAlignGmToUb(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm,
    uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / (32 / sizeof(T));
    intriParams.dstStride = (dstSingleRowCount - copyColCount) / (32 / sizeof(T));
    intriParams.srcStride = (srcSingleRowCount - copyColCount) / (32 / sizeof(T));
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::OverLapScore(LocalTensor<T> scoreLocal,
    const RunInfo &info, uint32_t tcStartIdx, uint32_t tcDealSize, uint32_t dStartIdx, uint32_t dDealSize)
{
    // scoreUb data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = constInfo_.dBaseSize * 2; // 2: kv和score各一份
    uint32_t copyRowCount = tcDealSize * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = ((uint32_t)COMP::coff) * dDealSize; // left和right在seq方向是交错存储的
    uint32_t srcScoreUbOffset = (tcStartIdx * constInfo_.cmpRatio) * srcSingleRowElemNum + constInfo_.dBaseSize + dStartIdx;
    if (GetBlockIdx() % 2 == 1) {
        srcScoreUbOffset += 128 * srcSingleRowElemNum;
    }

    uint32_t dstUbOffset = 0;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        DataCopyAlignGmToUb(scoreLocal[dstUbOffset], preMm1ResGm_[srcScoreUbOffset],
            copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
        dstUbOffset += dDealSize;
    }
    DataCopyAlignGmToUb(scoreLocal[dstUbOffset], curMm1ResGm_[srcScoreUbOffset],
        copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::OverLapKv(LocalTensor<T> kvLocal,
    const RunInfo &info, uint32_t tcStartIdx, uint32_t tcDealSize, uint32_t dStartIdx, uint32_t dDealSize)
{
    // kvUb data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = constInfo_.dBaseSize * 2; // 2: kv和score各一份
    uint32_t copyRowCount = tcDealSize * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = ((uint32_t)COMP::coff) * dDealSize; // left和right在seq方向是交错存储的
    uint32_t srcKvUbOffset = (tcStartIdx * constInfo_.cmpRatio) * srcSingleRowElemNum + dStartIdx;
    if (GetBlockIdx() % 2 == 1) {
        srcKvUbOffset += 128 * srcSingleRowElemNum;
    }

    uint32_t dstUbOffset = 0;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        DataCopyAlignGmToUb(kvLocal[dstUbOffset], preMm1ResGm_[srcKvUbOffset],
            copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
        dstUbOffset += dDealSize;
    }
    DataCopyAlignGmToUb(kvLocal[dstUbOffset], curMm1ResGm_[srcKvUbOffset],
        copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
    const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx,
    uint32_t dStart, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }

        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * coff * constInfo_.headDim +
            remainRowCnt * coff * constInfo_.headDim +
            (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize +
            dStart;

        DataCopyParams copyParams;
        copyParams.blockCount = copyRowCnt;
        copyParams.blockLen = dDealSize / (32 / sizeof(T));
        copyParams.dstStride = (coff * dDealSize - dDealSize) / (32 / sizeof(T));
        copyParams.srcStride = (coff * constInfo_.headDim - dDealSize) / (32 / sizeof(T));
        DataCopy(output[copyFinishRowCnt * coff * dDealSize], state[stateOffset], copyParams);

        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
    const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx,
    uint32_t dStart, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }

        if (idInBlockTable != 0) { // 32
            uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * coff * constInfo_.headDim +
                remainRowCnt * coff * constInfo_.headDim +
                (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize + dStart;

            DataCopyParams copyParams;
            copyParams.blockCount = copyRowCnt;
            copyParams.blockLen = dDealSize / (32 / sizeof(T));
            copyParams.dstStride = (coff * constInfo_.headDim - dDealSize) / (32 / sizeof(T));
            copyParams.srcStride = (coff * dDealSize - dDealSize) / (32 / sizeof(T));
            DataCopy(state[stateOffset], input[copyFinishRowCnt * coff * dDealSize], copyParams);
        }

        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SaveLeftFirst(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    uint32_t preBIdx = 0;
    // 左边为上一个batch或者最后一个batch的数据
    preBIdx = (blockInfo.bIdx - 1 + constInfo_.batchSize) % constInfo_.batchSize;

    uint32_t bSeqUsed = GetSeqUsed(preBIdx);
    uint32_t bStartPos = GetStartPos(preBIdx);
    // S=0时，跳B
    while (bSeqUsed == 0) {
        preBIdx = (preBIdx - 1 + constInfo_.batchSize) % constInfo_.batchSize;
        bSeqUsed = GetSeqUsed(preBIdx);
        bStartPos = GetStartPos(preBIdx);
    }

    uint32_t endIdxInBlock = (bStartPos + bSeqUsed) % constInfo_.cmpRatio;
    if (endIdxInBlock == 0) {
        endIdxInBlock = constInfo_.cmpRatio;
    }

    uint32_t copySeqCnt = (endIdxInBlock > bSeqUsed) ? bSeqUsed : endIdxInBlock;
    uint64_t endSeqIdx = bStartPos + bSeqUsed;
    uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
    uint64_t srcBaseOffset = (endIdxInBlock - copySeqCnt) * coff * dDealSize;
    WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], preBIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
    WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], preBIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SaveState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    if (COMP::coff == COFF::OVERLAP) {
        uint32_t coff = static_cast<uint32_t>(COMP::coff);
        // 存右边
        if (blockInfo.sIdx + blockInfo.validSeqCnt == blockInfo.bSeqUsed) {
            uint32_t copySeqCnt = constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt;
            if (blockInfo.validSeqCnt < copySeqCnt) {
                    copySeqCnt = blockInfo.validSeqCnt;
            } else {
                if (blockInfo.tailHolderSeqCnt > 0) {
                    if (blockInfo.validSeqCnt - copySeqCnt > constInfo_.cmpRatio) {
                        copySeqCnt += constInfo_.cmpRatio;
                    } else {
                        copySeqCnt += blockInfo.validSeqCnt - copySeqCnt;
                    }
                }
            }
            uint64_t endSeqIdx = blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt;
            uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
            uint64_t srcBaseOffset = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt - copySeqCnt) * coff * dDealSize;
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset + dDealSize], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset + dDealSize], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
        } else if (blockInfo.sIdx + blockInfo.validSeqCnt + constInfo_.cmpRatio > blockInfo.bSeqUsed) {
            uint32_t copySeqCnt = constInfo_.cmpRatio;
            if (copySeqCnt > blockInfo.validSeqCnt) {
                copySeqCnt = blockInfo.validSeqCnt;
            }
            uint64_t endSeqIdx = blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt;
            uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
            uint64_t srcBaseOffset = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt - copySeqCnt) * coff * dDealSize;
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset + dDealSize], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset + dDealSize], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
        }

        // 存左边
        if (blockInfo.dealTcSize == 1) {
            // 左边尾块和首块是同一块
            if (blockInfo.sIdx == 0) {
                SaveLeftFirst(kvLocal, scoreLocal, blockInfo, dStartIdx, dDealSize);
            } else {
                if (blockInfo.tailHolderSeqCnt > 0) {
                    // 左边为本batch数据
                    uint32_t copySeqCnt = constInfo_.cmpRatio;
                    if (blockInfo.sIdx < copySeqCnt) {
                        copySeqCnt = blockInfo.sIdx;
                    }
                    uint64_t endSeqIdx = blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt - (constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt);
                    uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
                    uint64_t srcBaseOffset = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt + blockInfo.tailHolderSeqCnt - copySeqCnt) * coff * dDealSize;
                    WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                    WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                }
            }
        } else {
            // 存左边第一块
            if (blockInfo.sIdx == 0) {
                SaveLeftFirst(kvLocal, scoreLocal, blockInfo, dStartIdx, dDealSize);
            }

            // 存左边最后一块
            if (blockInfo.tailHolderSeqCnt > 0) {
                // 左边为本batch数据
                uint32_t copySeqCnt = constInfo_.cmpRatio;
                if (blockInfo.validSeqCnt - (constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt) < copySeqCnt) {
                    copySeqCnt = blockInfo.validSeqCnt - (constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt);
                }
                uint64_t endSeqIdx = blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt - (constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt);
                uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
                uint64_t srcBaseOffset = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt + blockInfo.tailHolderSeqCnt - copySeqCnt) * coff * dDealSize;
                WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
            }
        }
    } else {
        if (blockInfo.tailHolderSeqCnt > 0) { // 仅尾块不满时需要存到state上
            uint32_t copySeqCnt = constInfo_.cmpRatio - blockInfo.tailHolderSeqCnt;
            if (copySeqCnt > blockInfo.validSeqCnt) {
                copySeqCnt = blockInfo.validSeqCnt;
            }
            uint64_t srcBaseOffset = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt - copySeqCnt) * dDealSize;
            uint64_t endSeqIdx = blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt;
            uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::ReadState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (blockInfo.compressTcSize == 0) {
        return;
    }

    if (COMP::coff == Compressor::COFF::OVERLAP) {
        float SOFTMAX_MIN_NUM = static_cast<float>(-1.0/0.0);
        // 填充右边
        if (blockInfo.headHolderSeqCnt > 0) {
            // 整个batch的第一块
            uint32_t copySeqCnt = blockInfo.bStartPos % constInfo_.cmpRatio;
            uint64_t endSeqIdx = blockInfo.bStartPos;
            uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
            uint64_t srcBaseOffset = 0;
            ReadFromCacheState(kvLocal[dDealSize], kvStateGm_, kvBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
            ReadFromCacheState(scoreLocal[dDealSize], scoreStateGm_, scoreBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + constInfo_.headDim, dDealSize);
        }

        // 填充左边
        if (blockInfo.compressTcSize == 1) {
            if (blockInfo.sIdx == 0) {
                // 右边为整个batch的第一块
                if (blockInfo.bStartPos < constInfo_.cmpRatio) {
                    // 无历史数据
                    // dDealSize必须为64
                    Duplicate(kvLocal, FLOAT_ZERO, dDealSize, constInfo_.cmpRatio, 1, 2 * dDealSize / 8);
                    Duplicate(scoreLocal, SOFTMAX_MIN_NUM, dDealSize, constInfo_.cmpRatio, 1, 2 * dDealSize / 8);
                } else {
                    // 存在前一个压缩组, 左侧数据全部在state中
                    uint32_t copySeqCnt = constInfo_.cmpRatio;
                    uint64_t endSeqIdx = blockInfo.bStartPos / constInfo_.cmpRatio * constInfo_.cmpRatio;
                    uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
                    uint64_t srcBaseOffset = 0;
                    ReadFromCacheState(kvLocal, kvStateGm_, kvBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                    ReadFromCacheState(scoreLocal, scoreStateGm_, scoreBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                }
            } else if (blockInfo.sIdx == (constInfo_.cmpRatio - (blockInfo.bStartPos % constInfo_.cmpRatio))) {
                // 右边为本次需要压缩的第二个压缩组
                if ((blockInfo.bStartPos % constInfo_.cmpRatio) > 0) {
                    uint32_t copySeqCnt = blockInfo.bStartPos % constInfo_.cmpRatio;
                    uint64_t endSeqIdx = blockInfo.bStartPos;
                    uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
                    uint64_t srcBaseOffset = 0;
                    ReadFromCacheState(kvLocal, kvStateGm_, kvBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                    ReadFromCacheState(scoreLocal, scoreStateGm_, scoreBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                }
            }
        } else {
            if (blockInfo.sIdx == 0) {
                if (blockInfo.bStartPos < constInfo_.cmpRatio) {
                    // 无历史数据
                    // dDealSize必须为64
                    Duplicate(kvLocal, FLOAT_ZERO, dDealSize, constInfo_.cmpRatio, 1, 2 * dDealSize / 8);
                    Duplicate(scoreLocal, SOFTMAX_MIN_NUM, dDealSize, constInfo_.cmpRatio, 1, 2 * dDealSize / 8);
                } else {
                    uint32_t copySeqCnt = constInfo_.cmpRatio + blockInfo.bStartPos % constInfo_.cmpRatio;
                    uint64_t endSeqIdx = blockInfo.bStartPos;
                    uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
                    uint64_t srcBaseOffset = 0;
                    ReadFromCacheState(kvLocal, kvStateGm_, kvBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                    ReadFromCacheState(scoreLocal, scoreStateGm_, scoreBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                }
            }
        }
    } else {
        // 需要压缩时, 首块如果头部有占位行, 必然需要从state拷贝数据
        if (blockInfo.headHolderSeqCnt > 0) {
            // 整个batch的第一块
            uint32_t copySeqCnt = blockInfo.bStartPos % constInfo_.cmpRatio;
            uint64_t endSeqIdx = blockInfo.bStartPos;
            uint64_t startSeqIdx = endSeqIdx - copySeqCnt;
            uint64_t srcBaseOffset = 0;
            ReadFromCacheState(kvLocal, kvStateGm_, kvBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
            ReadFromCacheState(scoreLocal, scoreStateGm_, scoreBlockTableGm_, blockInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::UpdateState(LocalTensor<T> kvLocal, LocalTensor<T> scoreLocal,
    uint32_t startTcIdx, const BlockInfo &blockInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventId_MTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventId_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
    SaveState(kvLocal, scoreLocal, blockInfo, dStartIdx, dDealSize);
    SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
    SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
    WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
    ReadState(kvLocal, scoreLocal, blockInfo, dStartIdx, dDealSize);
    SetFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::SoftmaxDN(
    LocalTensor<T> &scoreLocal, LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio :
        constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt],
                      tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::KvMulReduceScore(
    LocalTensor<T> &kvLocal, LocalTensor<T>& scoreLocal, LocalTensor<T> &dstLocal, LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio :
        constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    Mul(kvLocal, kvLocal, scoreLocal, tcDealSize * rCnt);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSum(dstLocal[r * dDealSize], kvLocal[r * rCnt], tmpUb[r  * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::CopyOutVec1Res(const RunInfo &info, LocalTensor<T> comporessedUb,
    uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint64_t outGmOffset = info.vec1ResOffset + compressedCnt_ * constInfo_.headDim + dStartIdx;
    DataCopyParams copyParams;
    copyParams.blockCount = compressTcSize;
    copyParams.blockLen = dDealSize / (32 / sizeof(T));
    copyParams.dstStride = (constInfo_.headDim - dDealSize) / (32 / sizeof(T));
    copyParams.srcStride = 0;
    DataCopy(vec1ResGm_[outGmOffset], comporessedUb, copyParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::UpdateBlockInfo(BlockInfo &blockInfo)
{
    if (!blockInfo.isFirst) {
        blockInfo.sIdx += blockInfo.validSeqCnt;
        if (blockInfo.sIdx == blockInfo.bSeqUsed) {
            blockInfo.sIdx = 0;
            do {
                if (blockInfo.bIdx > constInfo_.batchSize) {
                    break;
                }
                blockInfo.bIdx++;
                blockInfo.bSeqUsed = GetSeqUsed(blockInfo.bIdx);
                blockInfo.bStartPos = GetStartPos(blockInfo.bIdx);
            } while (blockInfo.bSeqUsed == 0);
        }
        if (blockInfo.dealSeqSize == 0) {
            return;
        }
    } else {
        if (blockInfo.dealSeqSize == 0) {
            return;
        }
        blockInfo.bSeqUsed = GetSeqUsed(blockInfo.bIdx);
        // 如果S=0，跳B
        while (blockInfo.bSeqUsed == 0) {
            if (blockInfo.bIdx > constInfo_.batchSize) {
                break;
            }
            blockInfo.bIdx++;
            blockInfo.bSeqUsed = GetSeqUsed(blockInfo.bIdx);
        }
        blockInfo.bStartPos = GetStartPos(blockInfo.bIdx);
        blockInfo.isFirst = false;
    }

    // 计算头部占位行数、有效数据行数、尾部占位行数
    blockInfo.headHolderSeqCnt = (blockInfo.bStartPos + blockInfo.sIdx) % constInfo_.cmpRatio;
    blockInfo.validSeqCnt = blockInfo.bSeqUsed - blockInfo.sIdx;
    if (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt > blockInfo.dealSeqSize) {
        blockInfo.validSeqCnt = blockInfo.dealSeqSize - blockInfo.headHolderSeqCnt;
    }
    blockInfo.tailHolderSeqCnt = constInfo_.cmpRatio - (blockInfo.bStartPos + blockInfo.sIdx + blockInfo.validSeqCnt) % constInfo_.cmpRatio;
    if (blockInfo.tailHolderSeqCnt == constInfo_.cmpRatio) {
        blockInfo.tailHolderSeqCnt = 0;
    }

    // 计算本次可以处理的Tc个数
    blockInfo.dealTcSize = (blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt + blockInfo.tailHolderSeqCnt) / constInfo_.cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时blockInfo.tailHolderSeqCnt > 0
    blockInfo.compressTcSize = blockInfo.dealTcSize;
    if (blockInfo.tailHolderSeqCnt > 0) {
        blockInfo.compressTcSize = blockInfo.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    // 更新剩余未处理的行数
    blockInfo.dealSeqSize -= blockInfo.headHolderSeqCnt + blockInfo.validSeqCnt + blockInfo.tailHolderSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVector<COMP>::DealVec1BaseBlock(const RunInfo &info, BlockInfo &blockInfo,
    uint32_t startTcIdx, uint32_t dStartIdx, uint32_t dDealSize)
{
    while (blockInfo.dealSeqSize > 0) {
        UpdateBlockInfo(blockInfo);
        LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
        OverLapScore(scoreUb, info, startTcIdx, blockInfo.dealTcSize, dStartIdx, dDealSize);
        inputQue1.EnQue(scoreUb);
        inputQue1.DeQue<T>();
        PipeBarrier<PIPE_V>();
        LocalTensor<T> scoreLocal = tmpBuff1.Get<T>();
        DataCopy(scoreLocal, scoreUb, blockInfo.dealTcSize * (uint32_t)COMP::coff * constInfo_.cmpRatio * dDealSize);
        inputQue1.FreeTensor(scoreUb);

        LocalTensor<T> apeUb = inputQue1.AllocTensor<T>();
        CopyInApe(apeUb, dStartIdx, dDealSize);
        inputQue1.EnQue(apeUb);
        inputQue1.DeQue<T>();
        PipeBarrier<PIPE_V>();
        AddApeToScore(scoreLocal, apeUb, blockInfo.dealTcSize, dDealSize);
        inputQue1.FreeTensor(apeUb);

        LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
        OverLapKv(kvUb, info, startTcIdx, blockInfo.dealTcSize, dStartIdx, dDealSize);
        inputQue1.EnQue(kvUb);
        inputQue1.DeQue<T>();
        PipeBarrier<PIPE_V>();
        LocalTensor<T> kvLocal = tmpBuff2.Get<T>();
        DataCopy(kvLocal, kvUb, blockInfo.dealTcSize * (uint32_t)COMP::coff * constInfo_.cmpRatio * dDealSize);
        inputQue1.FreeTensor(kvUb);

        UpdateState(kvLocal, scoreLocal, startTcIdx, blockInfo, dStartIdx, dDealSize);

        if (blockInfo.compressTcSize > 0) {
            LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
            PipeBarrier<PIPE_V>();
            SoftmaxDN(scoreLocal, tmpUb, blockInfo.compressTcSize, dDealSize);

            LocalTensor<T> comporessedUb = outputQue1.AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            KvMulReduceScore(kvLocal, scoreLocal, comporessedUb, tmpUb, blockInfo.compressTcSize, dDealSize);

            outputQue1.EnQue(comporessedUb);
            outputQue1.DeQue<T>();
            CopyOutVec1Res(info, comporessedUb, blockInfo.compressTcSize, dStartIdx, dDealSize);
            outputQue1.FreeTensor(comporessedUb);
        }

        compressedCnt_ += blockInfo.compressTcSize;
        startTcIdx += blockInfo.dealTcSize;
    }
}

template <typename COMP>
 __aicore__ inline void CompressorBlockVector<COMP>::ComputeVec1(const RunInfo &info)
{
    // 计算当前VecCore的任务量
    uint32_t dealTcStartIdx = 0;
    uint32_t dealTcSize = 128 / constInfo_.cmpRatio;
    if (dealTcSize > info.dealTcNum) {
        dealTcSize = info.dealTcNum;
    }
    if (GetBlockIdx() % 2 == 1) {
        dealTcStartIdx = dealTcSize;
        dealTcSize = info.dealTcNum - dealTcSize;
    }
    if (dealTcSize == 0) {
        return;
    }

    // 刷新当前VecCore的起始bIdx和sIdx
    uint32_t curBStart = info.bStart;
    uint32_t curSStart = info.sStart;
    uint32_t curCompressedCnt = 0;
    if (GetBlockIdx() % 2 == 1) {
        BlockInfo blockInfo(info.bStart, info.sStart, 0);
        blockInfo.dealSeqSize = dealTcStartIdx * constInfo_.cmpRatio;
        while (blockInfo.dealSeqSize > 0) {
            UpdateBlockInfo(blockInfo);
            curCompressedCnt += blockInfo.compressTcSize;
        }
        // 因为需要获取本次完成后的bIdx和sIdx, 所以这里再调用一次
        UpdateBlockInfo(blockInfo);
        curBStart = blockInfo.bIdx;
        curSStart = blockInfo.sIdx;
    }

    // 计算headDim和Tc方向切分大小
    constexpr uint32_t BASE_BLOCK_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    uint32_t maxDealColNum = BASE_BLOCK_ELEMENT_NUM / (constInfo_.cmpRatio * (uint32_t)COMP::coff);

    uint32_t tcSplitSize = 0;
    uint32_t dSplitSize = 0;
    uint32_t dLoopCount = 0;
    if (maxDealColNum < constInfo_.dBaseSize) {
        tcSplitSize = 1;
        dLoopCount = (constInfo_.dBaseSize + maxDealColNum - 1) / maxDealColNum;
        dSplitSize = constInfo_.dBaseSize / dLoopCount;
    } else {
        dSplitSize = constInfo_.dBaseSize;
        dLoopCount = 1;
        tcSplitSize = maxDealColNum / constInfo_.dBaseSize;
    }

    // 切块循环
    for (uint32_t dLoopIdx = 0; dLoopIdx < dLoopCount; dLoopIdx++) {
        BlockInfo blockInfo(curBStart, curSStart, 0);
        uint32_t actDealTcSize = tcSplitSize;
        compressedCnt_ = curCompressedCnt;
        for (uint32_t tcIdx = 0; tcIdx < dealTcSize; tcIdx += tcSplitSize) {
            if (tcIdx + tcSplitSize > dealTcSize) {
                actDealTcSize = dealTcSize - tcIdx;
            }
            // 处理单个切块
            blockInfo.dealSeqSize = actDealTcSize * constInfo_.cmpRatio;
            DealVec1BaseBlock(info, blockInfo, tcIdx, dLoopIdx * dSplitSize, dSplitSize);
        }
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::ComputeVec2(const Compressor::RunInfo &info)
{
    SplitCoreV2(info);
    uint32_t vec2DealM = v2TcEndIdx - v2TcStartIdx;
    uint32_t loopCount = (vec2DealM + v2MBaseSize - 1) / v2MBaseSize;
    for (uint32_t v2LoopIdx = 0, dealSize = v2MBaseSize; v2LoopIdx < loopCount; ++v2LoopIdx) {
        if (v2LoopIdx == loopCount - 1) {
            dealSize = vec2DealM - v2LoopIdx * v2MBaseSize;
        }
        DealVec2BaseBlock(info, v2TcStartIdx + v2LoopIdx * v2MBaseSize, dealSize);
    }
    v2TcStartIdx = 0;
    v2TcEndIdx = 0;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::DealVec2BaseBlock(const Compressor::RunInfo& info, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t computeSize = dealRowCount * constInfo_.headDim;
    int64_t inGmOffset = startRow * constInfo_.headDim;
    // CopyIn
    LocalTensor<T> vec1ResUb = inputQue1.AllocTensor<T>();
    DataCopy(vec1ResUb, vec2InputGm_[inGmOffset], computeSize);
    inputQue1.EnQue(vec1ResUb);
    inputQue1.DeQue<T>();

    // RmsNorm
    LocalTensor<T> normResUb = tmpBuff1.Get<T>();
    LocalTensor<T> tempLocal = tmpBuff2.Get<T>();
    PipeBarrier<PIPE_V>();
    MultRowRmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, info, dealRowCount);
    inputQue1.FreeTensor(vec1ResUb);

    // rope: 只对后RD进行rope; 将normResUb每行前headDim - ropeHeadDim个元素cast到X_T，然后再与rope后的结果组合存到outputUb
    LocalTensor<X_T> outputUb = outputQue1.AllocTensor<X_T>();
    PipeBarrier<PIPE_V>();
    CalRope(info, outputUb, normResUb, startRow - v2TcStartIdx, dealRowCount);
    PipeBarrier<PIPE_V>();
    // CopyOut
    outputQue1.EnQue(outputUb);
    outputQue1.DeQue<X_T>();
    CopyFinalResultOut(info, outputUb, startRow - v2TcStartIdx, dealRowCount);
    outputQue1.FreeTensor(outputUb);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::MultRowRmsNorm(LocalTensor<T> &normResUb, LocalTensor<T> &vec1ResUb, LocalTensor<T> &normWeightUb,
    LocalTensor<T> &tempLocal, const Compressor::RunInfo& info, uint32_t dealRowCount)
{
    RmsNormParam rmsNormParams = {
        constInfo_.reciprocalD,
        constInfo_.normEps,
        dealRowCount,
        constInfo_.headDim
    };
    RmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, rmsNormParams);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CalRope(const Compressor::RunInfo& info, LocalTensor<X_T> &outputUb,
    LocalTensor<T> &normResUb, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t bStartIdx = OutputBStartIdx;
    uint32_t sStartIdx = OutputSStartIdx;
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, bStartIdx, sStartIdx, globalScStart);
    uint32_t totalSize = dealRowCount * constInfo_.headDim;
    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        curDealScSize = dealRowCount;
        uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
        uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
        // sin与cos各占一半, 实际分别最多只会用8K,总占用16K
        LocalTensor<X_T> cosUb = inputQue1.AllocTensor<X_T>();
        LocalTensor<X_T> sinUb = cosUb[BUFFER_SIZE_BYTE_8K / sizeof(X_T)];
        DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
        DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
        inputQue1.EnQue(sinUb);
        inputQue1.DeQue<X_T>();

        LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
        LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
        LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
        Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
        Cast(ropeSinFp32Local, sinUb, RoundMode::CAST_NONE, computeSize);
        PipeBarrier<PIPE_V>();
        inputQue1.FreeTensor(sinUb);
        RotaryPosEmb<COMP::rotaryMode>(normResUb, normResUb, ropeCosFp32Local, ropeSinFp32Local, tempLocal, gatherOffsetCastUb, curDealScSize, 
                                        constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
        PipeBarrier<PIPE_V>();
        while (dealScSize > 0) {
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // 处理BSH有效数据在内存上不连续 （可能存在pad）
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // 逐batch计算写出索引
            preOutputBStartIdx = bStartIdx;
            preOutputSStartIdx = sStartIdx;
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);
            if (curDealScSize) {
                uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
                uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
                // sin与cos各占一半, 实际分别最多只会用8K,总占用16K
                LocalTensor<X_T> cosUb = inputQue1.AllocTensor<X_T>();
                LocalTensor<X_T> sinUb = cosUb[BUFFER_SIZE_BYTE_8K / sizeof(X_T)];
                DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
                DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
                inputQue1.EnQue(sinUb);
                inputQue1.DeQue<X_T>();

                LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
                LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
                LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();

                PipeBarrier<PIPE_V>();
                Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
                Cast(ropeSinFp32Local, sinUb, RoundMode::CAST_NONE, computeSize);
                PipeBarrier<PIPE_V>();
                inputQue1.FreeTensor(sinUb);

                RotaryPosEmb<COMP::rotaryMode>(normResUb[(dealRowCount - dealScSize - curDealScSize) * constInfo_.headDim],
                                    normResUb[(dealRowCount - dealScSize - curDealScSize) * constInfo_.headDim],
                                    ropeCosFp32Local, ropeSinFp32Local, tempLocal, gatherOffsetCastUb, curDealScSize, 
                                    constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
                PipeBarrier<PIPE_V>();
            }
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, bStartIdx, sStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
    Cast(outputUb, normResUb, RoundMode::CAST_RINT, totalSize);
    PipeBarrier<PIPE_V>();
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::SplitCoreV2(const Compressor::RunInfo& info)
{   
    // 累积N个基本块数据后做vec2，N=2，传入的RunInfo包含该组核处理的数据块的bStart、bEnd、sStart、sEnd以及dealTcCount；
    // 每组核切M方向将C1/V1后的数据分8 * 2个vec核上进行V2计算
    // 每次进行v2计算都会根据当前情况将workspace中的每组核处理的数据重新分到当前组的vec核

    // Input: syncAll前每组cube核处理的实际数据块在batch及s方向的起止idx及实际数据量(m方向)
    // Output: 每个vec核的处理数据块在m方向的起止位置及输出到Gm上的起始位置
    uint32_t coreNum = constInfo_.dBasicBlockNum * 2; // 组中有多少个vec核:16
    usedCoreNum = coreNum;
    uint32_t currCoreIdx = GetBlockIdx(); // 当前vec核ID
    uint32_t curVecCoreGroupIdx = currCoreIdx / coreNum; // 当前vec核所在组ID
    vec1ResGmStart = curVecCoreGroupIdx * constInfo_.nSize * constInfo_.tcBaseSize * constInfo_.headDim;
    // 1.计算总vec2基本块数量
    uint64_t totalBaseNum = info.dealScSize; // 当前组核累积的实际数据量
    // 2.每个vec核上分到的数据量
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (currCoreIdx % coreNum >= usedCoreNum) {
        return;
    }
    // 3.计算每个vec核的起始结束位置
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = (currCoreIdx % coreNum + 1) * avgBaseNum; // 当前vec核目标要达到的基本块数量
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum; // 分当前vec核时前面已经完成分核的基本块数量
    bool setStart = false;
    for (uint32_t i = 0; i < totalBaseNum; ++i) {
        if (accumBaseNum >= totalBaseNum) {
            return;
        }
        accumBaseNum += 1;
        if (!setStart && (accumBaseNum > targetStartBaseNum)) {
            v2TcStartIdx = i;
            setStart = true;
        } 
        if (setStart && (accumBaseNum >= targetBaseNum || i == (totalBaseNum - 1))) {
            // 更新当前核的End分核信息
            v2TcEndIdx = i + 1;
            GetScIdxInfo(info.bStart, info.scStart, info.dealScSize, v2TcStartIdx, v2TcEndIdx,
                OutputBStartIdx, OutputSStartIdx, OutputSize);
            return;
        }
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd,
                                                                            uint32_t scEnd, uint64_t &globalScStart)
{
    for (uint32_t bIdx = bStart; bIdx < bEnd; ++bIdx) {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            globalScStart += GetScSize();
        } else {
            curActSeqLength_ = constInfo_.sSize;
            curStartPos_ = GetStartPos(bIdx);
            globalScStart += (curActSeqLength_ + constInfo_.cmpRatio - 1) / constInfo_.cmpRatio;
        }
    }
    globalScStart -= scStart;
    globalScStart += scEnd;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart,
                                                                        uint32_t &dealScSize, uint32_t &curDealScSize) 
{
    curActSeqLength_ = GetSeqLength(outputBStart);
    curStartPos_ = GetStartPos(outputBStart);
    uint32_t curBatchScSize = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    uint32_t curBatchRemainScSize = curBatchScSize - outputSStart;
    curDealScSize = curBatchRemainScSize > dealScSize ? dealScSize : curBatchRemainScSize;
    dealScSize -= curDealScSize;
    outputSStart += curDealScSize;
    if (outputSStart == curBatchScSize) {
        outputBStart++;
        outputSStart = 0;
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVector<COMP>::CopyFinalResultOut(const Compressor::RunInfo& info, const LocalTensor<X_T> &cmpKvOutUb,
                                                                        uint32_t startRow, uint32_t dealRowCount)
{   
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, OutputBStartIdx, OutputSStartIdx, globalScStart);
    uint64_t outOffset = globalScStart * constInfo_.headDim;
    uint32_t copySize = dealRowCount * constInfo_.headDim;

    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        DataCopy(cmpKvOutGm_[outOffset], cmpKvOutUb, copySize);
        while (dealScSize > 0) {
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // 处理BSH有效数据在内存上不连续（可能存在pad）
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // 逐batch计算写出索引
            preOutputBStartIdx = OutputBStartIdx;
            preOutputSStartIdx = OutputSStartIdx;
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
            DataCopy(cmpKvOutGm_[globalScStart * constInfo_.headDim], cmpKvOutUb[ubProcessedCount * constInfo_.headDim], curDealScSize * constInfo_.headDim);
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, OutputBStartIdx, OutputSStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
} 
} // namespace Compressor
#endif // COMPRESSOR_BLOCK_VECTOR_H