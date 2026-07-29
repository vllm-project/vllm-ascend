/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_decode_update_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef lightning_indexer_decode_update_SERVICE_CUBE_H
#define lightning_indexer_decode_update_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_decode_update_common.h"

namespace LIKernel {
using namespace LICommon;
template <typename LIT>
class LIMatmul {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;

    __aicore__ inline LIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t QUERY_L0_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t KEY_L0_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s2GmOffset, uint64_t s2L0RealSize,
                                const LICommon::RunInfo &runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s2L0RealSize);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a();
    __aicore__ inline void QueryNd2Nz(const LICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUFFER_OFFSET * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * QUERY_L0_BUFFER_OFFSET * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * KEY_L0_BUFFER_OFFSET * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void
LIMatmul<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                   const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComputeMm1(const LICommon::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
            QueryNd2Nz(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        }
        for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
            uint64_t s2L0RealSize =
                s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
            LoadQueryToL0a();
            LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);
            SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
            WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
            ComuteL0c(s2L0RealSize);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
            Fixp(s2GmOffset + s2L1Offset, s2L0RealSize, runInfo);
            l0BufIdx_++;
        }
        if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
            SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const LICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                   constInfo_.kCacheBlockSize * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * BLOCK_CUBE :
                             s2L1Offset * BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::QueryNd2Nz(const LICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = constInfo_.qHeadNum;
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_, queryGm_[static_cast<uint64_t>(runInfo.bIdx) * constInfo_.qHeadNum * constInfo_.headDim],
             nd2nzPara);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadQueryToL0a()
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(constInfo_.qHeadNum, static_cast<uint64_t>(BLOCK_CUBE));
    loadData3DParams.l1W = BLOCK_CUBE;
    loadData3DParams.channelSize = constInfo_.headDim;

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;

    loadData3DParams.mExtension = constInfo_.qHeadNum;
    loadData3DParams.kExtension = constInfo_.headDim;
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * QUERY_L0_BUFFER_OFFSET], queryL1_,
                                   loadData3DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                                   const LICommon::RunInfo &runInfo)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * KEY_L0_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::ComuteL0c(uint64_t s2L0RealSize)
{
    MmadParams mmadParams;
    mmadParams.m = constInfo_.qHeadNum;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * QUERY_L0_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * KEY_L0_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::Fixp(uint64_t s2GmOffset, uint64_t s2L0RealSize,
                                           const LICommon::RunInfo &runInfo)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = constInfo_.qHeadNum;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;
    intriParams.srcStride = constInfo_.qHeadNum;
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11; // 3 unitflag
    intriParams.reluPre = 1;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize * constInfo_.s2BaseSize + s2GmOffset],
                      cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LIMatmul<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace LIKernel
#endif
