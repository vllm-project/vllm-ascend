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
 * \file quant_lightning_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef quant_lightning_indexer_SERVICE_CUBE_H
#define quant_lightning_indexer_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"

namespace QLIKernel {
using namespace QLICommon;
struct MmInfo {
    int64_t s2L0LoopId;
    int64_t s1gL0LoopId;
    int64_t s2L0RealSize;
    int64_t s2GmOffset;
};

template <typename QLIT>
class QLIMatmul {
public:
    using Q_T = typename QLIT::queryType;
    using K_T = typename QLIT::keyType;

    __aicore__ inline QLIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<float> &mm1ResGm,
                                               const GlobalTensor<half> &weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};  // isSetFMatrix isSetPadding;
    static constexpr uint64_t DOUBLE_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QW_MTE1_MTE2_EVENT = EVENT_ID5;  // KEY_MTE1_MTE2_EVENT + DOUBLE_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S1G_BASIC_BLOCK_L1 = 256;

    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK;
    static constexpr uint64_t SL1_BUFFER_OFFSET = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t WEIGHT_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * BLOCK_CUBE;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    static constexpr uint64_t L0C_BUFFER_OFFSET = 64 * 256;

private:
    __aicore__ inline void WeightDmaCopy(uint64_t s1gL1RealSize, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                      int64_t mStartPt);
    __aicore__ inline void LoadWeightToL0a(uint64_t s1gL1Offset);
    __aicore__ inline void ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset);
    __aicore__ inline void FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                       uint64_t s2GmOffset, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                     const MmInfo &mmInfo, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                     const MmInfo &mmInfo, const QLICommon::RunInfo &runInfo);
    __aicore__ inline void CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt, const MmInfo &lastMmInfo,
                                      const QLICommon::RunInfo &runInfo);
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<half> weightGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufWeightL1_;
    LocalTensor<half> weightL1_;
    TBuf<TPosition::B1> bufSL1_;
    LocalTensor<half> sL1_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t qwL1Mte2BufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufWeightL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * S1G_BASIC_BLOCK_L0 * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                            const GlobalTensor<K_T> &keyGm,
                                                            const GlobalTensor<Q_T> &queryGm,
                                                            const GlobalTensor<float> &mm1ResGm,
                                                            const GlobalTensor<half> &weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1BufIdx,
                                                  const MmInfo &mmInfo, const QLICommon::RunInfo &runInfo)
{
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    for (int64_t s1gOffset = 0; s1gOffset < s1gL0RealSize; s1gOffset += constInfo_.gSize) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        LoadSToL0b(s1gL0RealSize, mmInfo.s2L0RealSize, sL1BufIdx, s1gOffset);
        LoadWeightToL0a(s1gOffset + s1gL1Offset);

        ComputeWs(s1gL0RealSize, mmInfo.s2L0RealSize, s1gOffset);

        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        l0BufIdx_++;
    }

    FixpResToGm(s1gL0RealSize / constInfo_.gSize, mmInfo.s2L0RealSize, s1gL1Offset / constInfo_.gSize,
                mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    l0cBufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                                  const MmInfo &mmInfo, const QLICommon::RunInfo &runInfo)
{
    if (mmInfo.s1gL0LoopId == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BSND) {
            KeyNd2NzForPA(mmInfo.s2L0RealSize, runInfo.s2Idx * constInfo_.s2BaseSize + mmInfo.s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(mmInfo.s2L0RealSize, mmInfo, runInfo);
        }
        
        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    LoadQueryToL0a(s1gL1Offset, runInfo.actMBaseSize, s1gL0RealSize);
    LoadKeyToL0b(mmInfo.s2L0RealSize);

    if (mmInfo.s1gL0LoopId + 1 >= s1L0LoopCnt) {
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        keyL1BufIdx_++;
    }

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    ComputeQk(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);

    FixpSToL1(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt,
                                                   const MmInfo &lastMmInfo, const QLICommon::RunInfo &runInfo)
{
    mmInfo.s2L0LoopId = loopIdx / s1L0LoopCnt;
    mmInfo.s1gL0LoopId = loopIdx % s1L0LoopCnt;

    if (mmInfo.s1gL0LoopId == 0) {
        mmInfo.s2GmOffset = mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0;
        mmInfo.s2L0RealSize = mmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize
                                  ? runInfo.actualSingleProcessSInnerSize - mmInfo.s2GmOffset
                                  : S2_BASIC_BLOCK_L0;
    } else {
        mmInfo.s2L0RealSize = lastMmInfo.s2L0RealSize;
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeMm1(const QLICommon::RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        QueryNd2Nz(runInfo.actMBaseSize, runInfo);  // 256 * 128 // L1BasicBlock
        WeightDmaCopy(runInfo.actMBaseSize, runInfo);
    }
    int64_t loopIdx = 0;
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0);  // 2048取128
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize, S1G_BASIC_BLOCK_L0);                  // 256取128
    int64_t s1gL1Offset[2] = {0, static_cast<int64_t>(S1G_BASIC_BLOCK_L0)};
    int64_t s1gL0RealSize[2] = {s1L0LoopCnt > 1 ? static_cast<int64_t>(S1G_BASIC_BLOCK_L0) : runInfo.actMBaseSize,
                                runInfo.actMBaseSize - s1gL1Offset[1]};
    MmInfo mmInfo[2];
    CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);

    ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt],
                s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt], s1L0LoopCnt, mmInfo[loopIdx & 1],
                runInfo);

    SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
    sL1BufIdx_++;
    loopIdx++;

    while (loopIdx < s2L0LoopCnt * s1L0LoopCnt) {
        CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);

        ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt],
                  s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId % s1L0LoopCnt], s1L0LoopCnt, mmInfo[loopIdx & 1],
                  runInfo);

        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);
        sL1BufIdx_++;

        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % DOUBLE_BUF_NUM);

        ProcessWs(s1gL0RealSize[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt],
                    s1gL1Offset[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt], sL1BufIdx_,
                    mmInfo[(loopIdx + 1) & 1], runInfo);
        loopIdx++;
    }

    WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + (sL1BufIdx_ + 1) % DOUBLE_BUF_NUM);

    ProcessWs(s1gL0RealSize[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt],
              s1gL1Offset[mmInfo[(loopIdx + 1) & 1].s1gL0LoopId % s1L0LoopCnt], sL1BufIdx_ - 1,
              mmInfo[(loopIdx + 1) & 1], runInfo);

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        qwL1Mte2BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                      const QLICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                   constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset
                                                                            : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;  // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * S8_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo,
                                                 const QLICommon::RunInfo &runInfo)
{
    uint64_t dStride = constInfo_.headDim;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::BSND || K_LAYOUT_T == LI_LAYOUT::TND) {
        dStride = constInfo_.headDim * constInfo_.kHeadNum; // constInfo_.kHeadNum
    }
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = dStride;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + mmInfo.s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// batch, s1, g, 1
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::WeightDmaCopy(uint64_t s1gL1RealSize, const QLICommon::RunInfo &runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = s1gL1RealSize;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET],
             weightGm_[runInfo.loop % DOUBLE_BUF_NUM * BLOCK_CUBE * constInfo_.mBaseSize], copyInParams);
}

// batch, s1, n2, g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::QueryNd2Nz(uint64_t s1gL1RealSize, const QLICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET], queryGm_[runInfo.tensorQueryOffset],
             nd2nzPara);
}

// s1g, d
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                       uint64_t s1gL0RealSize)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE);  // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                          // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;          // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE);  // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                    // K   width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
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

    LoadData<Q_T, LOAD3DV2_CONFIG>(l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
                                   queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

// s1, g, s2  -->  2 * 64* 128
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                                   int64_t mStartPt)
{
    LoadData3DParamsV2<half> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = S1G_BASIC_BLOCK_L0 / BLOCK_CUBE;              // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                                   // Win=M0
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = constInfo_.gSize;                     // M height维度目的
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // K   width维度目的
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = mStartPt;
    LoadData<half, LOAD3DV2_CONFIG>(
        l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
        sL1_[(sL1BufIdx % DOUBLE_BUF_NUM) * SL1_BUFFER_OFFSET], loadData3DParams);
}

// s1,g,1(16), 2,64,16
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadWeightToL0a(uint64_t s1gL1Offset)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.gSize, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
             weightL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * WEIGHT_BUFFER_OFFSET + s1gL1Offset* BLOCK_CUBE],
             loadData2DParams);
}

// s2, d -> 128,128
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadKeyToL0b(uint64_t s2L0RealSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

// A: s1,g,1(16) B: s1,g,s2  C: s1, 1(16), s2
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.gSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET +
                                                s1gOffset * S2_BASIC_BLOCK_L0],
            l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
            l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
            mmadParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    DataCopy(sL1_[(sL1BufIdx_ % DOUBLE_BUF_NUM) * SL1_BUFFER_OFFSET],
             cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET], params);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                                    uint64_t s2GmOffset, const QLICommon::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = 16;
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(s1L0RealCount, CeilDiv(constInfo_.gSize, BLOCK_CUBE) * S2_BASIC_BLOCK_L0 / BLOCK_CUBE,
                                 2048);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize / constInfo_.gSize * constInfo_.s2BaseSize +
                                s1GmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET],
                      intriParams);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
}  // namespace QLIKernel
#endif