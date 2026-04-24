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
 * \file compressor_block_cube.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_CUBE_H
#define COMPRESSOR_BLOCK_CUBE_H

#include "../compressor_comm.h"

using namespace AscendC;

namespace Compressor {

template<typename COMP> class CompressorBlockCube {
using MM1_OUT_T = float;
public:
    __aicore__ inline CompressorBlockCube(){};
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
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGlobalBuffers(const GlobalTensor<MM1_OUT_T>& preMm1ResGm, const GlobalTensor<MM1_OUT_T>& curMm1ResGm);
    __aicore__ inline void AllocEventID(TPipe *pipe);
    __aicore__ inline void FreeEventID(TPipe *pipe);
    __aicore__ inline void ComputeMm1(const RunInfo &info);

private:
    using T = float;
    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;

    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetTIdxByBatch(uint32_t bIdx);
    __aicore__ inline void CopyWeightGmToL1(const RunInfo &info, LocalTensor<X_T> wL1Tensor,
        uint32_t hIdx, uint32_t kBase, uint32_t nLoopIdx);
    __aicore__ inline void CopyXGmToL1(const RunInfo &info, LocalTensor<X_T> xL1Tensor,
        uint32_t hIdx, uint32_t kBase, uint32_t mStart, uint32_t mDealSize, bool isLastM);
    __aicore__ inline void LoadAToL0(LocalTensor<X_T> aL0Tensor, LocalTensor<X_T> xL1Tensor,
        uint32_t mStart, uint32_t mDealSize, uint32_t kBase, uint32_t mSize, bool isRightLeft);
    __aicore__ inline void LoadBToL0(LocalTensor<X_T> bL0Tensor, LocalTensor<X_T> wL1Tensor,
        uint32_t nStart, uint32_t nDealSize, uint32_t nBase, uint32_t kBase);
    __aicore__ inline void MatrixMmad(LocalTensor<T> cL0Tensor, LocalTensor<X_T> aL0Tensor,
        LocalTensor<X_T> bL0Tensor, uint32_t mActSize, uint32_t nDealSize, uint32_t kActSize, bool isInitL0C);
    __aicore__ inline void CopyL0CDataToUb(LocalTensor<T> ubTensor, LocalTensor<T> cL0Tensor,
        uint32_t vecCoreIdx, uint32_t mSizeAlign, uint32_t nSizeAlign, uint32_t nIdx);

    ConstInfo constInfo_ = {};

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<MM1_OUT_T>preMm1ResGm;
    GlobalTensor<MM1_OUT_T>curMm1ResGm;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
    bool isExistSeqUsed = false;
    bool isExistStartPos = false;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_X_SIZE = 128 * 1024;
    static constexpr uint32_t L1_W_SIZE = 64 * 1024;
    // L1 Buffer
    TBuf<TPosition::A1> xBufL1;
    TBuf<TPosition::A1> wBufL1;
    // =================================L0 Buffer=================================
    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0B_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0C_PP_SIZE = 64 * 1024;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    // =================================Event&Buffer ID===========================
    // mte2 <> mte1 EventID
    static constexpr uint32_t X_EVENT0 = EVENT_ID0;
    static constexpr uint32_t X_EVENT1 = EVENT_ID1;
    uint32_t xBufId = 0;
    static constexpr uint32_t W_EVENT0 = EVENT_ID2;
    static constexpr uint32_t W_EVENT1 = EVENT_ID3;
    uint32_t wBufId = 0;
    static constexpr uint32_t M_LOCK_EVENT0 = EVENT_ID4;
    static constexpr uint32_t M_LOCK_EVENT1 = EVENT_ID5;
    uint32_t mLockId = 0;
    static constexpr uint32_t N_LOCK_EVENT0 = EVENT_ID6;
    static constexpr uint32_t N_LOCK_EVENT1 = EVENT_ID7;
    uint32_t nLockId = 0;
    // mte1 <> mmad EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    uint32_t l0abBufId = 0;
    // mmad <> fixpipe EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID0;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID1;
    uint32_t l0cBufId = 0;

    // =================================Loop======================================
    uint32_t curBIdx_ = 0;
    uint32_t curSIdx_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo_ = constInfo;
}

template <typename COMP> __aicore__ inline void CompressorBlockCube<COMP>::Init(
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
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitBuffers(TPipe *pipe)
{
    // L1
    pipe->InitBuffer(xBufL1, L1_X_SIZE * 2);
    pipe->InitBuffer(wBufL1, L1_W_SIZE * 2);

    // L0
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::InitGlobalBuffers(const GlobalTensor<MM1_OUT_T>& preMm1ResGm, const GlobalTensor<MM1_OUT_T>& curMm1ResGm)
{
    this->preMm1ResGm = preMm1ResGm;
    this->curMm1ResGm = curMm1ResGm;
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::AllocEventID(TPipe *pipe)
{
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT1);

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::FreeEventID(TPipe *pipe)
{
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT1);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyWeightGmToL1(const RunInfo &info, LocalTensor<X_T> wL1Tensor,
    uint32_t hIdx, uint32_t kBase, uint32_t nLoopIdx)
{
    // hIdx: hidden_size轴的索引ID
    // kBase: hidden_size轴单次往L1搬运的长度, 128
    // nLoopIdx: D方向L1搬运的循环ID, 0/1
    //      coff=1时, nLoopIdx=0, 搬运wkv nBase, nLoopIdx=1, 搬运wgate nBase
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // coff=2时, constInfo_.dBaseSize = 64, wkv和wgate在D轴各占一半
        // nLoopIdx=0, 搬运coffIdx=0的数据; nLoopIdx=1, 搬运coffIdx=1的数据
        uint64_t coffIdx = nLoopIdx;
        uint64_t dIdx = constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum * constInfo_.dBaseSize;
        uint64_t gmOffset = coffIdx * constInfo_.headDim * constInfo_.hSize + dIdx * constInfo_.hSize + hIdx;
        uint32_t wkvUbOffset = nLoopIdx * (kBase * 2 * constInfo_.dBaseSize);
        uint32_t wgateUbOffset = wkvUbOffset + constInfo_.dBaseSize * (32 / sizeof(X_T));
        CopySingleMatrixNDToNZ(wL1Tensor[wkvUbOffset], wkvGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, 2 * constInfo_.dBaseSize);
        CopySingleMatrixNDToNZ(wL1Tensor[wgateUbOffset], wgateGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, 2 * constInfo_.dBaseSize);
    } else {
        // coff=1时, constInfo_.dBaseSize = 128, nLoopIdx=0, 搬运wkv nBase, nLoopIdx=1, 搬运wgate nBase
        uint64_t dIdx = constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum * constInfo_.dBaseSize;
        uint64_t gmOffset = dIdx * constInfo_.hSize + hIdx;
        uint32_t ubOffset = nLoopIdx * kBase * constInfo_.dBaseSize;
        if (nLoopIdx == 0) {
            CopySingleMatrixNDToNZ(wL1Tensor[ubOffset], wkvGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, constInfo_.dBaseSize);
        } else {
            CopySingleMatrixNDToNZ(wL1Tensor[ubOffset], wgateGm_[gmOffset], constInfo_.dBaseSize, kBase, constInfo_.hSize, constInfo_.dBaseSize);
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockCube<COMP>::GetSeqUsed(uint32_t bIdx)
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
__aicore__ inline uint32_t CompressorBlockCube<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos) {
        return (uint32_t)startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockCube<COMP>::GetTIdxByBatch(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return (uint32_t)(cuSeqlensGm_.GetValue(bIdx));
    } else {
        return constInfo_.sSize * bIdx;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyXGmToL1(const RunInfo &info, LocalTensor<X_T> xL1Tensor,
    uint32_t hIdx, uint32_t kBase, uint32_t mStart, uint32_t mDealSize, bool isLastM)
{
    uint32_t bStartPos;
    uint32_t bSeqUsed;
    bool copyLastCmpBlock = false;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (mStart == 0) {
            curBIdx_ = info.bStart;
            curSIdx_ = info.sStart;
            // update cur idx
            if (curSIdx_ == 0) {
                if (curBIdx_ == 0) {
                    copyLastCmpBlock = true;
                } else {
                    // 取上一个batch的尾
                    // S=0时向前取B
                    do {
                        curBIdx_ = (curBIdx_ - 1 + constInfo_.batchSize) % constInfo_.batchSize;
                        bStartPos = GetStartPos(curBIdx_);
                        bSeqUsed = GetSeqUsed(curBIdx_);
                    } while (bSeqUsed == 0);

                    curSIdx_ = bStartPos + bSeqUsed == 0 ? 0 : max(Trunc(bStartPos + bSeqUsed - 1, constInfo_.cmpRatio), bStartPos) - bStartPos;
                }
            } else {
                curSIdx_ = curSIdx_ < constInfo_.cmpRatio ? 0 : curSIdx_ - constInfo_.cmpRatio;
            }
        }
        if (isLastM) {
            mDealSize = mDealSize + constInfo_.cmpRatio;
        }
    } else {
        if (mStart == 0) {
            curBIdx_ = info.bStart;
            curSIdx_ = info.sStart;
        }
    }

    uint32_t ubOffset = mStart * (32 / sizeof(X_T));
    uint32_t mSizeFinish = 0;
    if (copyLastCmpBlock) {
        uint32_t lastBIdx = constInfo_.batchSize;
        // S=0时向前取B
        do {
            lastBIdx = (lastBIdx - 1 + constInfo_.batchSize) % constInfo_.batchSize;
            bStartPos = GetStartPos(lastBIdx);
            bSeqUsed = GetSeqUsed(lastBIdx);
        } while (bSeqUsed == 0);

        uint32_t copySeqCnt = (bStartPos + bSeqUsed) % constInfo_.cmpRatio;
        if (copySeqCnt == 0) {
            copySeqCnt = constInfo_.cmpRatio;
        }
        if (bSeqUsed < copySeqCnt) {
            copySeqCnt = bSeqUsed;
        }
        uint64_t tmpSeqId = bSeqUsed- copySeqCnt;
        uint32_t rOffset = (bStartPos + bSeqUsed - copySeqCnt) % constInfo_.cmpRatio * (32 / sizeof(X_T));
        uint64_t sIdx = GetTIdxByBatch(lastBIdx) + tmpSeqId;
        uint64_t gmOffset = sIdx * constInfo_.hSize + hIdx;
        uint32_t nValue = copySeqCnt;
        uint32_t dValue = kBase;
        uint32_t srcDValue = constInfo_.hSize;
        uint32_t dstNzC0Stride = (info.dealTcNum * constInfo_.cmpRatio + 15) / 16 * 16;
        if constexpr (COMP::coff == COFF::OVERLAP) {
            dstNzC0Stride = (info.dealTcNum * constInfo_.cmpRatio + constInfo_.cmpRatio + 15) / 16 * 16;
        }
        CopySingleMatrixNDToNZ(xL1Tensor[ubOffset + rOffset], xGm_[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);

        ubOffset += constInfo_.cmpRatio * (32 / sizeof(X_T));
        mSizeFinish += constInfo_.cmpRatio;
    }

    while (mSizeFinish < mDealSize) {
        if (curBIdx_ > info.bEnd) {
            break;
        }
        uint32_t bSeqUsed = GetSeqUsed(curBIdx_);
        uint32_t bStartPos = GetStartPos(curBIdx_);

        // 如果S=0，直接到下一个B
        if (bSeqUsed == 0) {
            curBIdx_++;
            curSIdx_ = 0;
            continue;
        }

        uint32_t headHolderCnt = (bStartPos + curSIdx_) % constInfo_.cmpRatio;
        uint32_t canCopyCnt = bSeqUsed - curSIdx_;
        if (mSizeFinish + headHolderCnt + canCopyCnt > mDealSize) {
            canCopyCnt = mDealSize - (mSizeFinish + headHolderCnt);
        }
        uint32_t tailHolderCnt = constInfo_.cmpRatio - (bStartPos + curSIdx_ + canCopyCnt) % constInfo_.cmpRatio;
        if (tailHolderCnt == constInfo_.cmpRatio) {
            tailHolderCnt = 0;
        }
        if (canCopyCnt == 0) {
            continue;
        }

        // 跳过batch头部的预留行
        ubOffset += headHolderCnt * (32 / sizeof(X_T));

        uint64_t sIdx = GetTIdxByBatch(curBIdx_) + curSIdx_;
        uint64_t gmOffset = sIdx * constInfo_.hSize + hIdx;
        uint32_t nValue = canCopyCnt;
        uint32_t dValue = kBase;
        uint32_t srcDValue = constInfo_.hSize;
        uint32_t dstNzC0Stride = (info.dealTcNum * constInfo_.cmpRatio + 15) / 16 * 16;
        if constexpr (COMP::coff == COFF::OVERLAP) {
            dstNzC0Stride = (info.dealTcNum * constInfo_.cmpRatio + constInfo_.cmpRatio + 15) / 16 * 16;
        }
        CopySingleMatrixNDToNZ(xL1Tensor[ubOffset], xGm_[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
        ubOffset += (canCopyCnt + tailHolderCnt) * (32 / sizeof(X_T));

        curSIdx_ += canCopyCnt;
        if (curSIdx_ == bSeqUsed) {
            curBIdx_++;
            curSIdx_ = 0;
        }

        mSizeFinish += headHolderCnt + canCopyCnt + tailHolderCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadAToL0(LocalTensor<X_T> aL0Tensor, LocalTensor<X_T> xL1Tensor,
    uint32_t mStart, uint32_t mDealSize, uint32_t kBase, uint32_t mSize, bool isRightLeft)
{
    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (isRightLeft) {
            mStart += constInfo_.cmpRatio;
        }
        mSize += constInfo_.cmpRatio;
    }
    uint32_t xTensorOffset = mStart * (32 / sizeof(X_T));
    uint32_t mLoop = (mDealSize + 15) / 16;
    for (uint32_t i = 0; i < mLoop; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = i;
        loadData2DParams.repeatTimes = kBase / 16;
        loadData2DParams.srcStride = (mSize + 15) / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(aL0Tensor[16 * i * kBase], xL1Tensor[xTensorOffset], loadData2DParams);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::LoadBToL0(LocalTensor<X_T> bL0Tensor, LocalTensor<X_T> wL1Tensor,
    uint32_t nStart, uint32_t nDealSize, uint32_t nBase, uint32_t kBase)
{
    uint64_t wTensorOffset = (nStart / nBase) * (nBase * kBase);
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nBase / 16) * (kBase / 16);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(bL0Tensor, wL1Tensor[wTensorOffset], loadData2DParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::MatrixMmad(LocalTensor<T> cL0Tensor, LocalTensor<X_T> aL0Tensor,
    LocalTensor<X_T> bL0Tensor, uint32_t mActSize, uint32_t nDealSize, uint32_t kActSize, bool isInitL0C)
{
    MmadParams mmadParams;
    mmadParams.m = (mActSize + 15) / 16 * 16;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }
    mmadParams.n = nDealSize;
    mmadParams.k = kActSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    AscendC::PipeBarrier<PIPE_M>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::CopyL0CDataToUb(LocalTensor<T> ubTensor, LocalTensor<T> cL0Tensor,
    uint32_t vecCoreIdx, uint32_t mSizeAlign, uint32_t nSizeAlign, uint32_t nIdx)
{
}

template <typename COMP>
__aicore__ inline void CompressorBlockCube<COMP>::ComputeMm1(const RunInfo &info)
{
    static constexpr uint32_t K_SIZE = 256;
    static constexpr uint32_t K_L1_BASE = 128;
    static constexpr uint32_t N_L1_BASE = 128;
    static constexpr uint32_t M_L1_BASE = 128;
    static constexpr uint32_t K_L1_LOOP = K_SIZE / K_L1_BASE;

    uint32_t nSize = constInfo_.dBaseSize * 2 * ((uint32_t)COMP::coff);
    uint32_t mSize = info.dealTcNum * constInfo_.cmpRatio;
    bool needMLock = (mSize > M_L1_BASE);

    uint32_t hSize = constInfo_.hSize;
    for (uint32_t h = 0; h < hSize; h += K_SIZE) {
        for (uint32_t kL1Idx = 0; kL1Idx < K_L1_LOOP; kL1Idx++) {
            WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);

            LocalTensor<X_T> xL1Tensor = xBufL1.GetWithOffset<X_T>(L1_X_SIZE / sizeof(X_T), xBufId * L1_X_SIZE);
            LocalTensor<X_T> wL1Tensor = wBufL1.GetWithOffset<X_T>(L1_W_SIZE / sizeof(X_T), wBufId * L1_W_SIZE);
            for (uint32_t nL1 = 0; nL1 < nSize; nL1 += N_L1_BASE) {
                // nLockId = nL1 / N_L1_BASE
                WaitFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0 + nLockId);
                CopyWeightGmToL1(info, wL1Tensor, h + kL1Idx * K_L1_BASE, K_L1_BASE, nL1 / N_L1_BASE);
                SetFlag<HardEvent::MTE2_MTE1>(N_LOCK_EVENT0 + nLockId);
                WaitFlag<HardEvent::MTE2_MTE1>(N_LOCK_EVENT0 + nLockId);
                for (uint32_t mL1 = 0; mL1 < mSize; mL1 += M_L1_BASE) {
                    uint32_t mDealSize = M_L1_BASE;
                    bool isLastM = (mL1 + M_L1_BASE >= mSize);
                    if (isLastM) {
                        mDealSize = mSize - mL1;
                    }

                    if (nL1 == 0) {
                        mLockId = mL1 / M_L1_BASE;
                        if (needMLock) {
                            WaitFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0 + mLockId);
                        }
                        // 当Coff=2时，mL1为最后一块时，需要多拷贝一个cmpRatio
                        CopyXGmToL1(info, xL1Tensor, h + kL1Idx * K_L1_BASE, K_L1_BASE, mL1, mDealSize, isLastM);

                        // m轴可以一次拷贝时, 不需要m轴的拷贝锁；同步替换成大锁
                        if (needMLock) {
                            SetFlag<HardEvent::MTE2_MTE1>(M_LOCK_EVENT0 + mLockId);
                            WaitFlag<HardEvent::MTE2_MTE1>(M_LOCK_EVENT0 + mLockId);
                        } else {
                            SetFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
                            WaitFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
                        }
                    }

                    {
                        // 获取L0C
                        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                        LocalTensor<T> cL0Tensor = tmpBufL0C.GetWithOffset<T>((L0C_PP_SIZE / sizeof(T)), l0cBufId * L0C_PP_SIZE);
                        {
                            WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                            LocalTensor<X_T> aL0Tensor = tmpBufL0A.GetWithOffset<X_T>(L0A_PP_SIZE / sizeof(X_T), l0abBufId * L0A_PP_SIZE);
                            LocalTensor<X_T> bL0Tensor = tmpBufL0B.GetWithOffset<X_T>(L0B_PP_SIZE / sizeof(X_T), l0abBufId * L0B_PP_SIZE);
                            // 当Coff=2时，nL1=0时计算的是pre数据，nL1=N_L1_BASE时计算的是cur数据
                            uint32_t K_L0_BASE = K_L1_BASE;
                            LoadAToL0(aL0Tensor, xL1Tensor, mL1, mDealSize, K_L0_BASE, mSize, (nL1 > 0));
                            uint32_t N_L0_BASE = N_L1_BASE;

                            LoadBToL0(bL0Tensor, wL1Tensor, nL1, N_L0_BASE, N_L0_BASE, K_L0_BASE);
                            SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                            WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                            MatrixMmad(cL0Tensor, aL0Tensor, bL0Tensor, mDealSize, N_L0_BASE, K_L0_BASE, (h == 0) && (kL1Idx == 0));

                            SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                            l0abBufId = (l0abBufId + 1) % 2;
                        }
                        {
                            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
                            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);

                            if (kL1Idx != 0 || h != 0) {
                                SetAtomicAdd<MM1_OUT_T>();
                            }
                            if constexpr (COMP::coff == COFF::OVERLAP) {
                                FixpipeParamsV220 fixParams;
                                fixParams.mSize = (mDealSize + 15) / 16 * 16;
                                fixParams.nSize = N_L1_BASE;
                                fixParams.srcStride = (mDealSize + 15) / 16 * 16;   // 需要16对齐
                                fixParams.dstStride = constInfo_.dBaseSize * 2;
                                fixParams.ndNum = 1;
                                if (nL1 < nSize / 2) {
                                    Fixpipe(preMm1ResGm[mL1 * constInfo_.dBaseSize * 2], cL0Tensor, fixParams);
                                } else {
                                    Fixpipe(curMm1ResGm[mL1 * constInfo_.dBaseSize * 2], cL0Tensor, fixParams);
                                }
                            } else {
                                FixpipeParamsV220 fixParams;
                                fixParams.mSize = (mDealSize + 15) / 16 * 16;
                                fixParams.nSize = N_L1_BASE;
                                fixParams.srcStride = (mDealSize + 15) / 16 * 16;   // 需要16对齐
                                fixParams.dstStride = constInfo_.dBaseSize * 2;
                                fixParams.ndNum = 1;
                                Fixpipe(curMm1ResGm[mL1*nSize+nL1], cL0Tensor, fixParams);
                            }
                            
                            if (kL1Idx != 0 || h != 0) {
                                SetAtomicNone();
                            }
                        }
                        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                        l0cBufId = (l0cBufId + 1) % 2;
                    }

                    if (nL1 + N_L1_BASE >= nSize) {
                        if (needMLock) {
                            mLockId = mL1 / M_L1_BASE;
                            SetFlag<HardEvent::MTE1_MTE2>(M_LOCK_EVENT0 + mLockId);
                        }
                    }
                }
                SetFlag<HardEvent::MTE1_MTE2>(N_LOCK_EVENT0 + nLockId);
                nLockId = (nLockId + 1) % 2;
            }

            SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            xBufId = (xBufId + 1) % 2;
            SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);
            wBufId = (wBufId + 1) % 2;
        }
    }
}

} // namespace Compressor

#endif // COMPRESSOR_BLOCK_CUBE_H