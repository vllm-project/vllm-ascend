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
 * \file compressor_block_cube_perf.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_CUBE_PERF_H
#define COMPRESSOR_BLOCK_CUBE_PERF_H

#include "compressor_comm.h"
#include "compressor_tools.h"

using namespace AscendC;

namespace Compressor {

template<typename COMP> class CompressorBlockCubePerf {
using MM1_OUT_T = float;
public:
    __aicore__ inline CompressorBlockCubePerf(){};
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
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGlobalBuffers(const GlobalTensor<MM1_OUT_T>& kvMm1ResGm, const GlobalTensor<MM1_OUT_T>& scoreMm1ResGm);
    __aicore__ inline void AllocEventID(TPipe *pipe);
    __aicore__ inline void FreeEventID(TPipe *pipe);
    __aicore__ inline void ComputeMm1(const RunInfo &info);

private:
    using T = float;
    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;

    __aicore__ inline uint32_t GetMSize(const RunInfo &info, uint32_t coffId);
    __aicore__ inline void CopyXGmToL1(const RunInfo &info, LocalTensor<X_T> xL1Tensor, uint32_t hIdx, uint32_t kBase);
    __aicore__ inline void CopyWeightGmToL1(LocalTensor<X_T> wL1Tensor,
        uint32_t hIdx, uint32_t kBase, uint32_t coffId);
    __aicore__ inline void LoadAToL0(const RunInfo &info, LocalTensor<X_T> aL0Tensor, LocalTensor<X_T> xL1Tensor,
        uint32_t kStart, uint32_t kBase, uint32_t mStart, uint32_t mDealSize);
    __aicore__ inline void LoadBToL0(LocalTensor<X_T> bL0Tensor, LocalTensor<X_T> wL1Tensor,
        uint32_t kStart, uint32_t kBase);
    __aicore__ inline void MatrixMmad(LocalTensor<T> cL0Tensor, LocalTensor<X_T> aL0Tensor,
        LocalTensor<X_T> bL0Tensor, uint32_t mActSize, uint32_t nDealSize, uint32_t kActSize, bool isInitL0C);
    __aicore__ inline void CopyOutMm1Res(const RunInfo &info, LocalTensor<T> cL0Tensor,
        uint32_t coffId, uint32_t mStart, uint32_t mDealSize);

    ConstInfo constInfo_ = {};
    CompressorTools<COMP> tools_;

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<MM1_OUT_T>kvMm1ResGm;
    GlobalTensor<MM1_OUT_T>scoreMm1ResGm;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
    bool isExistSeqUsed = false;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_X_SIZE = 128 * 1024;
    static constexpr uint32_t L1_W_SIZE = 64 * 1024;
    // L1 Buffer
    TBuf<TPosition::A1> xBufL1;
    TBuf<TPosition::A1> wBufL1;
    // =================================L0 Buffer=================================
    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = 32 * 1024;      // 128 * 128 * 2 = 32k
    static constexpr uint32_t L0B_PP_SIZE = 32 * 1024;      // 128 * 128 * 2 = 32k
    static constexpr uint32_t L0C_PP_SIZE = 64 * 1024;      // (128 * 2) * 64 * 4 = 64k
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
    uint32_t xBufId = 0;    // 用于DB计数
    static constexpr uint32_t W_EVENT0 = EVENT_ID4;
    static constexpr uint32_t W_EVENT1 = EVENT_ID5;
    static constexpr uint32_t W_EVENT2 = EVENT_ID6;
    static constexpr uint32_t W_EVENT3 = EVENT_ID7;
    uint32_t wBufId = 0;    // 用于DB计数
    // mte1 <> mmad EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    uint32_t l0abBufId = 0;
    // mmad <> fixpipe EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID0;   // 每块L0C单独分配EVENT_ID
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID1;
    uint32_t l0cBufId = 0;

    // =================================Loop======================================
    uint32_t curBIdx_ = 0;
    uint32_t curSIdx_ = 0;

};

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::InitParams(const ConstInfo &constInfo, const CompressorTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
}

template <typename COMP> __aicore__ inline void CompressorBlockCubePerf<COMP>::Init(
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
        __gm__ uint8_t *cmpKvOut)
{
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    isExistSeqUsed = (seqUsed != nullptr);
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::InitBuffers(TPipe *pipe)
{
    // L1
    // 1. coff=1时, mBase=256, kL1=256, X单次拷贝到L1的数据量最大为mBase*kL1*sizeof(BF16/FP16)=256*256*2=128K
    // 2. coff=2时, mBase=128, kL1=256, r最大为128, X单次拷贝到L1的最大数据量为(128+r)*kL1*sizeof(BF16/FP16)<=128K
    pipe->InitBuffer(xBufL1, L1_X_SIZE * 2);
    // dBaseSize<=64, wkv和wgate各一份, kL1=256, 右矩阵为dBaseSize*2*sizeof(BF16/FP16)<=64K
    // cur和pre循环使用, 2份buffer就足够
    pipe->InitBuffer(wBufL1, L1_W_SIZE * 4);

    // L0
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::InitGlobalBuffers(const GlobalTensor<MM1_OUT_T>& kvMm1ResGm, const GlobalTensor<MM1_OUT_T>& scoreMm1ResGm)
{
    this->kvMm1ResGm = kvMm1ResGm;
    this->scoreMm1ResGm = scoreMm1ResGm;
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::AllocEventID(TPipe *pipe)
{
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(X_EVENT1);

    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(W_EVENT3);

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::FreeEventID(TPipe *pipe)
{
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT1);

    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT3);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::CopyXGmToL1(const RunInfo &info, LocalTensor<X_T> xL1Tensor,
    uint32_t hIdx, uint32_t kBase)
{
    uint32_t tStart = tools_.GetTIdxByBatch(info.bStart) + info.sStart; // 此基本块在整个序列中的位置
    uint32_t copySeqCnt = info.dealSeqCnt; // 此基本块处理的长度

    uint32_t xL1Offset = 0 * (32 / sizeof(X_T));
    uint64_t sIdx = tStart;    // 起始s在整个T的起始点
    uint64_t gmOffset = sIdx * constInfo_.hSize + hIdx;
    uint32_t nValue = copySeqCnt;
    uint32_t dValue = kBase;  // 拷贝的列数kBase
    uint32_t srcDValue = constInfo_.hSize;
    uint32_t dstNzC0Stride = (copySeqCnt + 15) / 16 * 16;    // 1行变2行的行方向的偏移，需要16对齐
    CopySingleMatrixNDToNZ(xL1Tensor[xL1Offset], xGm_[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::CopyWeightGmToL1(LocalTensor<X_T> wL1Tensor,
    uint32_t hIdx, uint32_t kBase, uint32_t coffId)
{
    // coffId=0, 搬运左矩阵的数据; coffId=1, 搬运右矩阵的数据
    uint64_t gmOffset = coffId * constInfo_.headDim * constInfo_.hSize + constInfo_.dIdx * constInfo_.hSize + hIdx;
    uint32_t wkvL1Offset = 0;
    uint32_t wgateL1Offset = constInfo_.dBaseSize * (32 / sizeof(X_T)); // wgate与wkv的起始点相隔dBaseSize个32B
    uint32_t nValue = constInfo_.dBaseSize;
    uint32_t dValue = kBase;
    uint32_t srcDValue = constInfo_.hSize;
    uint32_t dstNzC0Stride = 2 * constInfo_.dBaseSize; // 2: wkv和wgate各搬运dBaseSize行, dBaseSize需保证8的倍数
    CopySingleMatrixNDToNZ(wL1Tensor[wkvL1Offset], wkvGm_[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
    CopySingleMatrixNDToNZ(wL1Tensor[wgateL1Offset], wgateGm_[gmOffset],  nValue, dValue, srcDValue, dstNzC0Stride);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::LoadAToL0(const RunInfo &info, LocalTensor<X_T> aL0Tensor,
    LocalTensor<X_T> xL1Tensor, uint32_t kStart, uint32_t kBase, uint32_t mStart, uint32_t mDealSize)
{
    uint32_t mSize = info.dealSeqCnt;

    uint32_t mSizeAlign = Align(mSize, 16U);
    uint32_t xTensorOffset = kStart * mSizeAlign + mStart * (32 / sizeof(X_T));
    uint32_t mLoop = Align(mDealSize, 16U) / 16;

    for (uint32_t i = 0; i < mLoop; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = i;
        loadData2DParams.repeatTimes = kBase / (32 / sizeof(X_T));
        loadData2DParams.srcStride = mSizeAlign / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(aL0Tensor[i * 16 * kBase], xL1Tensor[xTensorOffset], loadData2DParams); // 16: 一个分型的行数
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::LoadBToL0(LocalTensor<X_T> bL0Tensor, LocalTensor<X_T> wL1Tensor,
    uint32_t kStart, uint32_t kBase)
{
    uint32_t rowCnt = 2 * constInfo_.dBaseSize; // 2: wkv和wgate各搬运dBaseSize行, dBaseSize需保证8的倍数
    uint64_t wTensorOffset = rowCnt * kStart;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (rowCnt / 16) * (kBase / (32 / sizeof(X_T)));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(bL0Tensor, wL1Tensor[wTensorOffset], loadData2DParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::MatrixMmad(LocalTensor<T> cL0Tensor, LocalTensor<X_T> aL0Tensor,
    LocalTensor<X_T> bL0Tensor, uint32_t mActSize, uint32_t nDealSize, uint32_t kActSize, bool isInitL0C)
{
    MmadParams mmadParams;
    mmadParams.m = (mActSize + 15) / 16 * 16;
    mmadParams.n = nDealSize;
    mmadParams.k = kActSize;
    mmadParams.cmatrixInitVal = isInitL0C;
    mmadParams.cmatrixSource = false;
    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    PipeBarrier<PIPE_M>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::CopyOutMm1Res(const RunInfo &info, LocalTensor<T> cL0Tensor,
    uint32_t coffId, uint32_t mStart, uint32_t mDealSize)
{
    // coffId=0, 存左矩阵的数据; coffId=1, 存右矩阵的数据
    FixpipeParamsV220 fixParams;
    fixParams.mSize = mDealSize;
    fixParams.nSize = constInfo_.dBaseSize;
    fixParams.srcStride = (mDealSize + 15) / 16 * 16;   // 需要16对齐
    fixParams.dstStride = (uint32_t)COMP::coff * constInfo_.headDim;
    fixParams.ndNum = 1;

    uint64_t dbOffset = info.cubeDbIdx * constInfo_.dbSize;
    uint64_t gmOffset = coffId * constInfo_.headDim + constInfo_.dIdx + mStart * fixParams.dstStride + dbOffset;
    uint32_t kvOffset = 0;
    uint32_t scoreOffset = (mDealSize + 15) / 16 * 16 * constInfo_.dBaseSize;

    Fixpipe(kvMm1ResGm[gmOffset], cL0Tensor[kvOffset], fixParams);
    Fixpipe(scoreMm1ResGm[gmOffset], cL0Tensor[scoreOffset], fixParams);

}


template <typename COMP>
__aicore__ inline uint32_t CompressorBlockCubePerf<COMP>::GetMSize(const RunInfo &info, uint32_t coffId)
{
    return info.dealSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockCubePerf<COMP>::ComputeMm1(const RunInfo &info)
{
    static constexpr uint32_t K_SIZE = 512;
    static constexpr uint32_t K_L1_BASE = 256;
    static constexpr uint32_t M_L0_BASE = 128;
    static constexpr uint32_t K_L0_BASE = 128;
    uint32_t nCoff =  (uint32_t)COMP::coff;

    // hSize为K_SIZE=512的倍数
    uint32_t hSize = constInfo_.hSize;
    uint32_t hIdxStart = (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * K_L1_BASE;  // 每组核内的h循环起始不同
    for (uint32_t h = 0; h < hSize; h += K_SIZE) {
        for (uint32_t k = 0; k < K_SIZE; k += K_L1_BASE) {
            bool isFirst = (h == 0 && k == 0);
            bool isLast = ((h + K_SIZE >= hSize) && (k + K_L1_BASE >= K_SIZE));
            uint32_t hIdx = (h + k + hIdxStart) % hSize;    // h方向错位搬运
            WaitFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            LocalTensor<X_T> xL1Tensor = xBufL1.GetWithOffset<X_T>(L1_X_SIZE / sizeof(X_T), xBufId * L1_X_SIZE);
            CopyXGmToL1(info, xL1Tensor, hIdx, K_L1_BASE);
            SetFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
            WaitFlag<HardEvent::MTE2_MTE1>(X_EVENT0 + xBufId);
            for (uint32_t i = nCoff; i > 0; i--) {
                // coffId=0, 计算pre数据; coffId=1, 计算cur数据
                uint32_t coffId = i - 1;
                WaitFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);
                LocalTensor<X_T> wL1Tensor = wBufL1.GetWithOffset<X_T>(L1_W_SIZE / sizeof(X_T), wBufId * L1_W_SIZE);
                CopyWeightGmToL1(wL1Tensor, hIdx, K_L1_BASE, coffId);
                SetFlag<HardEvent::MTE2_MTE1>(W_EVENT0 + wBufId);
                WaitFlag<HardEvent::MTE2_MTE1>(W_EVENT0 + wBufId);

                uint32_t mSize = GetMSize(info, coffId);
                uint32_t actMDealSize = M_L0_BASE;
                for (uint32_t mL0 = 0; mL0 < mSize; mL0 += M_L0_BASE) {
                    if (mL0 + M_L0_BASE > mSize) {
                        actMDealSize = mSize - mL0;
                    }

                    l0cBufId = coffId + (mL0 / M_L0_BASE);
                    LocalTensor<T> cL0Tensor = tmpBufL0C.GetWithOffset<T>((L0C_PP_SIZE / sizeof(T)), l0cBufId * L0C_PP_SIZE);
                    if (isFirst) {
                        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                    }
                    uint32_t nDealSize = 2 * constInfo_.dBaseSize; // 2: wkv和wgate各搬运dBaseSize行, dBaseSize需保证8的倍数
                    for (uint32_t kL0 = 0; kL0 < K_L1_BASE; kL0 += K_L0_BASE) {
                        WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                        LocalTensor<X_T> aL0Tensor = tmpBufL0A.GetWithOffset<X_T>(L0A_PP_SIZE / sizeof(X_T), l0abBufId * L0A_PP_SIZE);
                        LocalTensor<X_T> bL0Tensor = tmpBufL0B.GetWithOffset<X_T>(L0B_PP_SIZE / sizeof(X_T), l0abBufId * L0B_PP_SIZE);
                        LoadAToL0(info, aL0Tensor, xL1Tensor, kL0, K_L0_BASE, mL0, actMDealSize);
                        LoadBToL0(bL0Tensor, wL1Tensor, kL0, K_L0_BASE);
                        SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                        WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId);
                        bool isInitL0C = isFirst && (kL0 == 0);
                        MatrixMmad(cL0Tensor, aL0Tensor, bL0Tensor, actMDealSize, nDealSize, K_L0_BASE, isInitL0C);
                        SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId);
                        l0abBufId = (l0abBufId + 1) % 2;
                    }
                    if (isLast) {
                        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
                        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId);
                        CopyOutMm1Res(info, cL0Tensor, coffId, mL0, actMDealSize);
                        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId);
                    }
                }

                SetFlag<HardEvent::MTE1_MTE2>(W_EVENT0 + wBufId);
                wBufId = (wBufId + 1) % 4;
            }
            SetFlag<HardEvent::MTE1_MTE2>(X_EVENT0 + xBufId);
            xBufId = (xBufId + 1) % 2;
        }
    }

}

} // namespace Compressor

#endif // COMPRESSOR_BLOCK_CUBE_PERF_H