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
 * \file gmmsq_weight_quant_cube_compute.h
 * \brief GMMSQ MxA8W4 CubeCompute class for AIC-side MMA matmul computation
 */
#ifndef GMMSQ_WEIGHT_QUANT_CUBE_COMPUTE_H
#define GMMSQ_WEIGHT_QUANT_CUBE_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "weight_quant_tool.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::Dn2NzParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using namespace WeightQuantBatchMatmulV2::Arch35;



namespace GMMSQWeightQuant {

#define GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM                                                                           \
    template <typename xType, typename weightScaleType, typename xScaleType, typename yType,                           \
              const WqmmConfig &wqmmConfig>

#define GMMSQ_WQ_CUBE_COMPUTE_CLASS GMMSQWeightQuantCubeCompute<xType, weightScaleType, xScaleType, yType, wqmmConfig>

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
class GMMSQWeightQuantCubeCompute {
public:
    __aicore__ inline GMMSQWeightQuantCubeCompute(){};
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y, __gm__ weightScaleType *weightScale,
                                            __gm__ xScaleType *xScale);
    __aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize);
    __aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1, int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t kMutilLoadL1Size, const BasicBlockOffsetParam &param);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam,
                                           uint64_t kbL1RealSize = 0);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BasicBlockOffsetParam &offsetParam,
                                          uint64_t kbL1RealSize = 0);
    __aicore__ inline void SetMTE1ToMTE2(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void SetWeightMTE1ToMTE2(const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset);
    __aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset);
    __aicore__ inline void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param);
    __aicore__ inline void PadL1IfNotAlign(const LocalTensor<xType> &l1Buf, uint64_t dbOffset, uint64_t dbIdx,
                                           uint64_t realSize, uint64_t dimSize);
    __aicore__ inline void EndSync();

private:
    __aicore__ inline void DoMmadCompute(uint64_t loopId, const L0CopyAndCalcParams &l0CopyAndCalcParams);
    __aicore__ inline void FillL1WithZero(const LocalTensor<uint32_t> &l1Buf, uint64_t blockCount);
    __aicore__ inline void ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize, uint64_t scaleKL1RealSize,
                                                  Dn2NzParams &dn2NzParams);
    using L0DataType = typename AscendC::GetL0DataType<xType, true>::Type;

    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
    static constexpr uint64_t L0C_BUF_SIZE = 256 * 256;

    static constexpr uint64_t MX_SCALE_L1_SIZE = 32 * GetKBUnit<xType>() * sizeof(xType); // scaleA/B单块分配空间

    int8_t aL1DbNum_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<xType>();
    static constexpr uint32_t EVENT_ID_MTE1_TO_MTE2 = 3;
    static constexpr uint32_t EVENT_ID_WEIGHT_MTE1_TO_MTE2 = 1;
    static constexpr uint32_t EVENT_ID_SCALE_MTE1_TO_MTE2 = 5;
    static constexpr uint32_t EVENT_ID_M_TO_MTE1 = 3;
    static constexpr uint32_t EVENT_ID_MTE1_TO_M = 3;
    static constexpr uint32_t EVENT_ID_MTE2_TO_MTE1 = 3;
    uint64_t aL1Count_ = 0;
    uint64_t aL1MaxHalfCount_ = 0;
    uint64_t scaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;

    AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    GlobalTensor<xType> xGlobal_;
    GlobalTensor<fp8_e8m0_t> xScaleGlobal_;
    GlobalTensor<fp8_e8m0_t> weightScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    uint64_t l0LoopIdx_ = 0;

    LocalTensor<L0DataType> l0a_{TPosition::A2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<L0DataType> l0b_{TPosition::B2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<float> l0c_{TPosition::CO1, 0, 256 * GetKBUnit<int8_t>() / sizeof(float)};

    LocalTensor<xType> aL1_;
    uint64_t aL1DbOffset_;

    LocalTensor<fp8_e8m0_t> xScaleAL1_;
    uint64_t xScaleAL1DbOffset_;

    LocalTensor<fp8_e8m0_t> weightScaleBL1_;
    uint64_t weightScaleBL1DbOffset_;

    uint64_t l0cOffset_ = 0;
};

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize)
{
    aL1Count_ = 0;
    aL1MaxHalfCount_ = 0;
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    xScaleAL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(fp8_e8m0_t));
    xScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    weightScaleBL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize, l1RemainSize / sizeof(fp8_e8m0_t));
    weightScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    aL1_ = LocalTensor<xType>(TPosition::TSCM, l1StartSize, l1RemainSize);
    aL1DbOffset_ = l1RemainSize >> 1;

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y,
                                                                     __gm__ weightScaleType *weightScale,
                                                                     __gm__ xScaleType *xScale)
{
    xGlobal_.SetGlobalBuffer(x);
    yGlobal_.SetGlobalBuffer(y);
    xScaleGlobal_.SetGlobalBuffer(xScale);
    weightScaleGlobal_.SetGlobalBuffer(weightScale);
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2(uint64_t kbGmOffset)
{
    if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (scaleBufIdx_ & 1));
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2(uint64_t kbGmOffset,
                                                                       const BasicBlockOffsetParam &offsetParam)
{
    if ((kbGmOffset + offsetParam.kbL1Size) % MX_SCALE_K_L1_SIZE == 0 ||
        kbGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + (scaleBufIdx_ & 1));
        scaleBufIdx_++;
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t kaGmOffset,
                                                                   const BasicBlockOffsetParam &offsetParam,
                                                                   uint64_t kbL1RealSize)
{
    if (kbL1RealSize > 0) {
        // Wait only at the last K iteration when the tail block is not 64-aligned
        if (kaGmOffset + kbL1RealSize >= offsetParam.kSize && kbL1RealSize % K_ALIGNMENT64 != 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2);
        }
    } else {
        if (aL1DbNum_ > SINGLE_BUFFER_NUM && kaGmOffset % offsetParam.kaL1Size == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1));
        }
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::SetWeightMTE1ToMTE2(const BasicBlockOffsetParam &offsetParam)
{
    if (offsetParam.kbL1Size >= offsetParam.kSize && offsetParam.kSize % K_ALIGNMENT64 != 0) {
        // 首轮weight清零
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2);
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t kaGmOffset,
                                                                  const BasicBlockOffsetParam &offsetParam,
                                                                  uint64_t kbL1RealSize)
{
    if (kbL1RealSize > 0) {
        // Set at the second-to-last K iteration, only when the upcoming last block will not be 64-aligned.
        // Condition: current is NOT the last iteration, but the next iteration IS the last,
        // and the last block's K size is not 64-aligned.
        if (kaGmOffset + kbL1RealSize < offsetParam.kSize &&
            kaGmOffset + kbL1RealSize + kbL1RealSize >= offsetParam.kSize &&
            offsetParam.kSize % K_ALIGNMENT64 != 0) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_WEIGHT_MTE1_TO_MTE2);
        }
    } else {
        if ((kaGmOffset + offsetParam.kbL1Size) % offsetParam.kaL1Size == 0 ||
            kaGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + (aL1BufIdx_ & 1));
            aL1BufIdx_++;
        }
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(const BasicBlockOffsetParam &param,
                                                                      uint64_t kbL1Offset)
{
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / 2;  // SwiGLU splits N into swish/gate halves
    uint64_t kSizeAligned = CeilAlign(param.kSize, K_ALIGNMENT64);
    uint64_t scaleKGmSize = kSizeAligned / MX_GROUPSIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > param.kSize ?
                                    CeilAlign(param.kSize - kbL1Offset, K_ALIGNMENT64) / MX_GROUPSIZE :
                                    MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;

    // Copy xScale
    Dn2NzParams scaleAdn2NzParams;
    ConfigScaleDn2NzParams(param.mL1Size, scaleKGmSize, scaleKL1RealSize, scaleAdn2NzParams);
    int64_t scaleAGmOffset = param.mOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;
    GlobalTensor<half> f16ScaleAGlobal;
    f16ScaleAGlobal.SetGlobalBuffer((__gm__ half *)xScaleGlobal_[scaleAGmOffset].GetPhyAddr(),
                                    (param.mL1Size * scaleKL1RealSize) >> 1);
    auto f16ScaleALocal = xScaleAL1_[(scaleBufIdx_ & 1) * xScaleAL1DbOffset_].template ReinterpretCast<half>();
    DataCopy(f16ScaleALocal, f16ScaleAGlobal, scaleAdn2NzParams);

    // Copy weightScale part1
    uint64_t swishNOffset = param.nOffset * scaleKGmSize;
    uint64_t gateNOffset = (param.nOffset + param.nSize / 2) * scaleKGmSize;
    uint64_t kScaleOffset = kbL1Offset / MX_GROUPSIZE;

    Dn2NzParams scaleBdn2NzParamsSwish;
    ConfigScaleDn2NzParams(nHalfSize, scaleKGmSize, scaleKL1RealSize, scaleBdn2NzParamsSwish);
    int64_t scaleBGmOffsetSwish = swishNOffset + kScaleOffset;
    GlobalTensor<half> f16ScaleBSwishGlobal;
    f16ScaleBSwishGlobal.SetGlobalBuffer((__gm__ half *)weightScaleGlobal_[scaleBGmOffsetSwish].GetPhyAddr(),
                                         (nHalfSize * scaleKL1RealSize) >> 1);
    auto f16ScaleBSwishLocal =
        weightScaleBL1_[(scaleBufIdx_ & 1) * weightScaleBL1DbOffset_].template ReinterpretCast<half>();
    DataCopy(f16ScaleBSwishLocal, f16ScaleBSwishGlobal, scaleBdn2NzParamsSwish);

    // Copy weightScale part2
    Dn2NzParams scaleBdn2NzParamsGate;
    ConfigScaleDn2NzParams(nHalfSize, scaleKGmSize, scaleKL1RealSize, scaleBdn2NzParamsGate);
    int64_t scaleBGmOffsetGate = gateNOffset + kScaleOffset;
    GlobalTensor<half> f16ScaleBGateGlobal;
    f16ScaleBGateGlobal.SetGlobalBuffer((__gm__ half *)weightScaleGlobal_[scaleBGmOffsetGate].GetPhyAddr(),
                                        (nHalfSize * scaleKL1RealSize) >> 1);
    auto f16ScaleBGateLocal =
        weightScaleBL1_[(scaleBufIdx_ & 1) * weightScaleBL1DbOffset_ + (nHalfSize * scaleKL1RealSize)]
            .template ReinterpretCast<half>();
    DataCopy(f16ScaleBGateLocal, f16ScaleBGateGlobal, scaleBdn2NzParamsGate);
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::CopyAGmToL1(const BasicBlockOffsetParam &param, int64_t kaGmOffset)
{
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;

    nd2nzParams.nValue = param.mL1Size;
    nd2nzParams.dValue = kaL1RealSize;
    nd2nzParams.srcDValue = param.kSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_], xGlobal_[kaGmOffset + param.mOffset * param.kSize], nd2nzParams);

    PadL1IfNotAlign(aL1_, aL1DbOffset_, aL1BufIdx_, kaL1RealSize, param.mL1Size);
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::LaunchMatmul(const LocalTensor<xType> &weightL1, int64_t kbOffset,
                                                                 uint64_t kbL1RealSize, uint64_t kMutilLoadL1Size,
                                                                 const BasicBlockOffsetParam &param)
{
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
    uint64_t aL1Offset = 0;

    aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;

    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = param.mL1Size;
    l0CopyAndCalcParams.mL1Size = param.mL1Size;
    uint64_t kL0Size = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
    l0CopyAndCalcParams.kL1Size = param.kaL1Size;
    l0CopyAndCalcParams.nL0Size = param.nL1Size;
    l0CopyAndCalcParams.nL1Size = param.nL1Size;
    l0CopyAndCalcParams.scaleKL1Size = kMutilLoadL1Size;
    uint64_t mL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t nL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (kbOffset == 0) {
        uint64_t nextL0cBufSize = CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                  CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
        if (l0cOffset_ + nextL0cBufSize > L0C_BUF_SIZE) {
            l0cOffset_ = 0;
        }
    }
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += kL0Size) {
        bool isLastL1K = l1KOffset + kL0Size >= kbL1RealSize;
        l0CopyAndCalcParams.isFirstKLoop = (l1KOffset + kbOffset) == 0;
        l0CopyAndCalcParams.isLastKLoop = l1KOffset + kbOffset + kL0Size >= param.kSize;

        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : kL0Size;
        l0CopyAndCalcParams.kL0Size = realL0k;
        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM;
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
        LoadAAndScaleL1ToL0(l0a_[loopId * L0_BUF_OFFSET_B8],
                            aL1_[aL1Offset + mL1AlignSize * ((l1KOffset + kbOffset) % param.kaL1Size)],
                            xScaleAL1_[(scaleBufIdx_ & 1) * xScaleAL1DbOffset_ +
                                       BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],
                            l0CopyAndCalcParams);
        LoadBAndScaleL1ToL0(l0b_[loopId * L0_BUF_OFFSET_B8], weightL1[nL1AlignSize * l1KOffset],
                            weightScaleBL1_[(scaleBufIdx_ & 1) * weightScaleBL1DbOffset_ +
                                            BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],
                            l0CopyAndCalcParams);

        DoMmadCompute(loopId, l0CopyAndCalcParams);
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::DoMmadCompute(uint64_t loopId,
                                                                  const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    MmadCompute(l0c_[l0cOffset_], l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8],
                l0CopyAndCalcParams);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
    l0LoopIdx_++;
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::ConfigScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmSize,
                                                                           uint64_t scaleKL1RealSize,
                                                                           Dn2NzParams &dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    // 矩阵行数，即要搬运数据M/N轴大小
    dn2NzParams.dValue = rowNum;
    // 矩阵列数，K轴大小，B8按B16搬运，大小除以2
    dn2NzParams.nValue = CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE);
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    // 原始scale一行大小
    dn2NzParams.srcDValue = CeilDivide(scaleKGmSize, SCALE_COPY_GROUP_SIZE);
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzC0Stride = CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE);
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_N_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::EndSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::PadL1IfNotAlign(const LocalTensor<xType> &l1Buf, uint64_t dbOffset,
                                                                    uint64_t dbIdx, uint64_t realSize, uint64_t dimSize)
{
    if (realSize % K_ALIGNMENT64 != 0) {
        uint64_t alignDimSize = CeilAlign(dimSize, static_cast<uint64_t>(BLOCK_CUBE));
        LocalTensor<uint32_t> fillTensor =
            l1Buf[(dbIdx & 1) * dbOffset + alignDimSize * realSize].template ReinterpretCast<uint32_t>();
        FillL1WithZero(fillTensor, alignDimSize);
    }
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::FillL1WithZero(const LocalTensor<uint32_t> &l1Buf,
                                                                   uint64_t blockCount)
{
    AscendC::InitConstValueParams<uint32_t> initParams;
    initParams.repeatTimes = 1;
    initParams.dstGap = 0;
    initParams.initValue = 0;
    initParams.blockNum = blockCount;
    AscendC::Fill(l1Buf, initParams);
}

GMMSQ_WQ_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_CUBE_COMPUTE_CLASS::GetTensorC(LocalTensor<yType> &yUb,
                                                               const BasicBlockOffsetParam &param)
{
    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = param.mL1Size;
    fixL0CToDstParams.nL0Size = param.nL1Size;
    fixL0CToDstParams.outNSize = param.nL1Size;
    fixL0CToDstParams.splitNSize = 0;
    fixL0CToDstParams.dstNdStride = 0;
    FixL0CToDst(yUb, l0c_[l0cOffset_], fixL0CToDstParams);
    l0cOffset_ += Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                  CeilAlign(fixL0CToDstParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
}

} // namespace GMMSQWeightQuant
#endif
