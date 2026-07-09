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
 * \file hc_pre_cube_compute.h
 * \brief
 */
#ifndef HC_PRE_CUBE_COMPUTE_H
#define HC_PRE_CUBE_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "hc_pre_base.h"

using AscendC::BLOCK_CUBE;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
using namespace AscendC;

namespace HcPre {
struct MmParams {
    uint64_t curML1;
    uint64_t curKL1;
    uint64_t curNL1;
    uint64_t singleCoreK;
    uint64_t kGmBaseOffset;
    uint64_t nGmSize;
    uint64_t kGmSize;
    uint64_t nOutSize;
    uint64_t xWsKSize;
    bool isLastK;
    bool isFirstK;
};
#define HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM template <bool enableSquareSum>

#define HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS HcCubeCompute<enableSquareSum>

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
class HcCubeCompute {
public:
    __aicore__ inline HcCubeCompute(){};

    __aicore__ inline void Init(const GlobalTensor<float>& xGm, const GlobalTensor<float>& fnGm, TPipe *tpipe);
    __aicore__ inline void ComputeDecode(const AscendC::GlobalTensor<float> &xGm, const AscendC::GlobalTensor<float> &workspaceGlobalA2,
        const AscendC::GlobalTensor<float> &workspaceGlobalAB, const MmParams &mmParams);
    __aicore__ inline void CopyInB1(
    uint64_t mGmOffset, uint64_t kGmOffset, uint64_t kL1Size, const MmParams &mmParams);
    __aicore__ inline void SetBL1Mte1ToMte2Flag();
    __aicore__ inline void WaitBL1Mte1ToMte2Flag();
    __aicore__ inline void End();

private:
    __aicore__ inline void CopyInA1(
    uint64_t kL1Size,
    const GlobalTensor<float> &aGlobal, const LocalTensor<float> &al1Local, const MmParams &mmParams);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float> &workspaceGlobal,
        const AscendC::LocalTensor<float> &c1Local, uint64_t baseM, uint64_t baseN, bool enableNz2Nd, uint64_t N);
    __aicore__ inline void Fixp(const AscendC::GlobalTensor<float> &workspaceGlobalA2,
        const AscendC::GlobalTensor<float> &workspaceGlobalAB, const MmParams &mmParams);
    __aicore__ inline void LoadAToL0A(
        uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams);
    __aicore__ inline void LoadAToL0B(
        uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams);
    __aicore__ inline void MmadA2(uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams &mmParams);
    __aicore__ inline void LoadBToL0B(
        uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams);
    __aicore__ inline void MmadAB(
    uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams &mmParams);

    TPipe *pipe_;
    const HcPreTilingData *tiling_;

    int32_t blkIdx_ = -1;
    int64_t batch_ = 0;
    int64_t hcParam_ = 0;
    int64_t dParam_ = 0;

    static constexpr int32_t ONE_BLOCK_SIZE = 32;
    int32_t perBlock32 = ONE_BLOCK_SIZE / sizeof(float);

    GlobalTensor<float> fnGm_;
    GlobalTensor<float> yGm_;

    static constexpr uint64_t MM1_MTE2_MTE1_EVENT = 2;
    static constexpr uint64_t X_MTE1_MTE2_EVENT = 2;
    static constexpr uint64_t B_MTE1_MTE2_EVENT = 4;
    static constexpr uint64_t M_MTE1_EVENT_L0A = 3;
    static constexpr uint64_t M_MTE1_EVENT_L0B = 5;
    static constexpr uint64_t MTE1_M_EVENT = 2;

    static constexpr uint64_t L1_BUF_NUM = 2;
    static constexpr uint64_t L0A_BUF_NUM = 2;
    static constexpr uint64_t L0B_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 2;
    static constexpr uint64_t L0C_BUF_NUM = 2;

    static constexpr uint64_t L1_BUF_OFFSET = 128 * 256;
    static constexpr uint64_t L0AB_BUF_OFFSET = 32 * 256;
    static constexpr uint64_t L0C_BUF_OFFSET = 64 * 256;
    static constexpr uint64_t L0C_A2_BUF_OFFSET = 256 * 16;

    static constexpr uint16_t UNIT_FLAG_ENABLE = 2;
    static constexpr uint16_t UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;

    static constexpr uint64_t K_L0_SIZE = 32UL;
    static constexpr uint64_t FLOAT_C0_SIZE = 8UL;

    uint64_t l1aLoopIdx_ = 0;
    uint64_t l1bLoopIdx_ = 0;
    uint64_t l0aLoopIdx_ = 0;
    uint64_t l0bLoopIdx_ = 0;
    uint64_t l0cLoopIdx_ = 0;

    LocalTensor<float> l1a_;
    LocalTensor<float> l1b_;

    LocalTensor<float> l0a_;
    LocalTensor<float> l0b_;
    LocalTensor<float> l0c_;

    uint64_t k_ = 0;
    uint64_t n_ = 0;
};

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::Init(const GlobalTensor<float>& xGm, const GlobalTensor<float>& fnGm, TPipe *tpipe)
{
    fnGm_ = fnGm;

    TBuf<TPosition::A1> l1aBuffer;
    tpipe->InitBuffer(l1aBuffer, 256 * 1024);
    l1a_ = l1aBuffer.Get<float>();

    TBuf<TPosition::B1> l1bBuffer;
    tpipe->InitBuffer(l1bBuffer, 256 * 1024);
    l1b_ = l1bBuffer.Get<float>();

    TBuf<TPosition::A2> l0aBuffer;
    tpipe->InitBuffer(l0aBuffer, 64 * 1024);
    l0a_ = l0aBuffer.Get<float>();

    TBuf<TPosition::B2> l0bBuffer;
    tpipe->InitBuffer(l0bBuffer, 64 * 1024);
    l0b_ = l0bBuffer.Get<float>();

    // loc
    TBuf<TPosition::CO1> l0cBuffer;
    tpipe->InitBuffer(l0cBuffer, 64 * 1024);
    l0c_ = l0cBuffer.Get<float>();

    for (int i = 0; i < L0A_BUF_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
    }
    for (int i = 0; i < L1_BUF_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + i);
        SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + i);
    }
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::CopyInA1(
    uint64_t kL1Size,
    const GlobalTensor<float> &aGlobal, const LocalTensor<float> &al1Local, const MmParams &mmParams)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = mmParams.curML1;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = mmParams.xWsKSize;  // vec处理的singleK
    nd2nzParams.dstNzC0Stride = (mmParams.curML1 + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    DataCopy(al1Local, aGlobal, nd2nzParams);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::CopyInB1(
    uint64_t mGmOffset, uint64_t kGmOffset, uint64_t kL1Size, const MmParams &mmParams)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = mmParams.curNL1;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = mmParams.kGmSize;  // 原始k
    nd2nzParams.dstNzC0Stride = (mmParams.curNL1 + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    DataCopy(l1b_[(l1bLoopIdx_ % L1_BUF_NUM) * L1_BUF_OFFSET], fnGm_[kGmOffset], nd2nzParams);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::CopyOut(const AscendC::GlobalTensor<float> &workspaceGlobal,
    const AscendC::LocalTensor<float> &c1Local, uint64_t baseM, uint64_t baseN, bool enableNz2Nd, uint64_t N)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = baseN;
    intriParams.mSize = baseM;
    // set mode to float32, then cast in ub
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = enableNz2Nd;
    if (enableNz2Nd) {
        intriParams.dstStride = N;  // NZ -> ND
        intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    } else {
        intriParams.dstStride = CeilAlign(intriParams.nSize, AscendC::BLOCK_CUBE);  // NZ -> NZ
        intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
    }
    intriParams.unitFlag = UNIT_FLAG_ENABLE_AUTO_CLOSE;  // 3

    AscendC::DataCopy(workspaceGlobal, c1Local, intriParams);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::Fixp(const AscendC::GlobalTensor<float> &workspaceGlobalA2,
    const AscendC::GlobalTensor<float> &workspaceGlobalAB, const MmParams &mmParams)
{
    // Copy MmadA2
    CopyOut(workspaceGlobalA2,
        l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * L0C_BUF_OFFSET],
        mmParams.curML1,
        BLOCK_CUBE,
        false,
        BLOCK_CUBE);  // nz m,16
    // Copy MmadAB
    CopyOut(workspaceGlobalAB,
        l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * L0C_BUF_OFFSET + L0C_A2_BUF_OFFSET],
        mmParams.curML1,
        mmParams.curNL1,
        true,
        mmParams.nOutSize);  // ND M/N are 512B aligned.
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::SetBL1Mte1ToMte2Flag()
{
    SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + l1bLoopIdx_ % L1_BUF_NUM);
    l1bLoopIdx_++;
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::WaitBL1Mte1ToMte2Flag()
{
    WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + l1bLoopIdx_ % L1_BUF_NUM);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::ComputeDecode(
    const AscendC::GlobalTensor<float> &xGm,
    const AscendC::GlobalTensor<float> &workspaceGlobalA2, const AscendC::GlobalTensor<float> &workspaceGlobalAB,
    const MmParams &mmParams)
{
    WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + l1aLoopIdx_ % L1_BUF_NUM);
    uint64_t kGmOffset = 0;
    uint64_t curKL1Size = (kGmOffset + mmParams.curKL1 >= mmParams.singleCoreK) ? (mmParams.singleCoreK - kGmOffset) : mmParams.curKL1;
    CopyInA1(curKL1Size, xGm[kGmOffset], l1a_[(l1aLoopIdx_ % DOUBLE_BUFFER) * L1_BUF_OFFSET], mmParams);

    SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT + l1aLoopIdx_ % L1_BUF_NUM);
    kGmOffset += mmParams.curKL1;
    while (kGmOffset < mmParams.singleCoreK) {
        WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + (l1aLoopIdx_ + 1) % L1_BUF_NUM);
        uint64_t nextKL1Size = (kGmOffset + mmParams.curKL1 >= mmParams.singleCoreK) ? (mmParams.singleCoreK - kGmOffset) : mmParams.curKL1;
        CopyInA1(nextKL1Size, xGm[kGmOffset], l1a_[((l1aLoopIdx_ + 1) % DOUBLE_BUFFER) * L1_BUF_OFFSET], mmParams);

        SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT + (l1aLoopIdx_ + 1) % L1_BUF_NUM);
        WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT + l1aLoopIdx_ % L1_BUF_NUM);

        for (uint64_t kL1Offset = 0; kL1Offset < curKL1Size; kL1Offset += K_L0_SIZE) {  // K_L0_SIZE 32
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            LoadAToL0A(kL1Offset, K_L0_SIZE, l1aLoopIdx_, mmParams);

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadAToL0B(kL1Offset, K_L0_SIZE, l1aLoopIdx_, mmParams);
            MmadA2(kGmOffset + kL1Offset - mmParams.curKL1, K_L0_SIZE,
            false,
            mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadBToL0B((kGmOffset + kL1Offset - mmParams.curKL1) % mmParams.xWsKSize, K_L0_SIZE, l1bLoopIdx_, mmParams);
            MmadAB(kGmOffset + kL1Offset - mmParams.curKL1, K_L0_SIZE,
                false,
                mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            l0aLoopIdx_++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + l1aLoopIdx_ % L1_BUF_NUM);
        l1aLoopIdx_++;
        kGmOffset += mmParams.curKL1;
        curKL1Size = nextKL1Size;
    }

    WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT + l1aLoopIdx_ % L1_BUF_NUM);
    for (uint64_t kL1Offset = 0; kL1Offset < curKL1Size; kL1Offset += K_L0_SIZE) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
        LoadAToL0A(kL1Offset, K_L0_SIZE, l1aLoopIdx_, mmParams);  // to l0a

        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
        LoadAToL0B(kL1Offset, K_L0_SIZE, l1aLoopIdx_, mmParams);  // to l0b
        MmadA2(kGmOffset + kL1Offset - mmParams.curKL1, K_L0_SIZE,
            mmParams.isLastK && kL1Offset + K_L0_SIZE >= curKL1Size, mmParams);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
        l0bLoopIdx_++;

        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
        LoadBToL0B((kGmOffset + kL1Offset - mmParams.curKL1)  % mmParams.xWsKSize, K_L0_SIZE, l1bLoopIdx_, mmParams);  // to l0b
        MmadAB(kGmOffset + kL1Offset - mmParams.curKL1, K_L0_SIZE,
            mmParams.isLastK && kL1Offset + K_L0_SIZE >= curKL1Size,
            mmParams);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
        l0bLoopIdx_++;

        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
        l0aLoopIdx_++;
    }
    SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + l1aLoopIdx_ % L1_BUF_NUM);
    l1aLoopIdx_++;
    if (mmParams.isLastK) {
        Fixp(workspaceGlobalA2, workspaceGlobalAB, mmParams);  // l0cLoopIdx_++;
    }
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::End()
{
    for (int i = 0; i < L0A_BUF_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
    }
    for (int i = 0; i < L1_BUF_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT + i);
        WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT + i);
    }
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::LoadAToL0A(
    uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams)
{
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
    LoadData3DParamsV2<float> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(mmParams.curML1, BLOCK_CUBE);  // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                          // Win=M0
    loadData3DParams.channelSize = kL0Size;          // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(mmParams.curML1, BLOCK_CUBE);  // M height维度目的
    loadData3DParams.kExtension = kL0Size;                    // K   width维度目的
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

    LoadData<float, LOAD3DV2_CONFIG>(l0a_[(l0aLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET],
                                   l1a_[(l1LoopIdx % L1_BUF_NUM) * L1_BUF_OFFSET +
             CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset],
                                   loadData3DParams);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::LoadAToL0B(
    uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams)
{
    // mk nz -> m,k zz
    for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
        LoadData2DParams l1ToL0bParams;
        l1ToL0bParams.startIndex = 0;
        l1ToL0bParams.repeatTimes = CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
        l1ToL0bParams.srcStride = CeilDiv(mmParams.curML1, (uint64_t)BLOCK_CUBE);
        l1ToL0bParams.dstGap = 0;
        LoadData(l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET +
                      mL0Offset * CeilAlign(kL0Size, (uint64_t)BLOCK_CUBE >> 1)],
            l1a_[(l1LoopIdx % L1_BUF_NUM) * L1_BUF_OFFSET +
                 CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset +
                 mL0Offset * (uint64_t)(BLOCK_CUBE >> 1)],
            l1ToL0bParams);
    }
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::MmadA2(
    uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams &mmParams)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
        MmadParams mmadParams;
        mmadParams.m = BLOCK_CUBE;
        mmadParams.n = BLOCK_CUBE;
        mmadParams.k = kL0Size;
        mmadParams.cmatrixInitVal = mmParams.isFirstK && kGmOffset == 0;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = isLastK ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;
        // mk zz @ mk zz
        Mmad(l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * L0C_BUF_OFFSET + mL0Offset * BLOCK_CUBE],
            l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
            l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
            mmadParams);
    }
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::LoadBToL0B(
    uint64_t kL1Offset, uint64_t kL0Size, uint64_t l1LoopIdx, const MmParams &mmParams)
{
    LoadData2DParams l1ToL0bParams;
    l1ToL0bParams.startIndex = 0;
    l1ToL0bParams.repeatTimes =
        CeilDiv(mmParams.curNL1, (uint64_t)BLOCK_CUBE) * CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
    l1ToL0bParams.srcStride = 1;
    l1ToL0bParams.dstGap = 0;
    // n,k nz -> n,k nz
    LoadData(l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET],
        l1b_[(l1LoopIdx % L1_BUF_NUM) * L1_BUF_OFFSET +
             CeilAlign(mmParams.curNL1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset],
        l1ToL0bParams);
}

HC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void HC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::MmadAB(
    uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams &mmParams)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    AscendC::SetHF32Mode(1);
    AscendC::SetHF32TransMode(1);
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(mmParams.curML1, BLOCK_CUBE);
    mmadParams.n = mmParams.curNL1;
    mmadParams.k = kL0Size;  // kl0Size
    mmadParams.cmatrixInitVal = mmParams.isFirstK && kGmOffset == 0;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = isLastK ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;
    // mk zz @ nk nz
    Mmad(l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * L0C_BUF_OFFSET + L0C_A2_BUF_OFFSET],
        l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * L0AB_BUF_OFFSET],
        l0b_[(l0bLoopIdx_ % L0AB_BUF_NUM) * L0AB_BUF_OFFSET],
        mmadParams);
    AscendC::SetHF32Mode(0);
}

}  // namespace HcPre

#endif  // HC_PRE_CUBE_COMPUTE_H
