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
 * \file gmmsq_weight_quant_vec_compute.h
 * \brief
 */
#ifndef GMMSQ_WEIGHT_QUANT_VEC_COMPUTE_H
#define GMMSQ_WEIGHT_QUANT_VEC_COMPUTE_H

#include "kernel_cube_intf.h"
#include "gmmsq_math_common.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "weight_quant_tool.h"
#include "gmmsq_quant_mx_vf.h"
#include "gmmsq_weight_quant_cube_compute_tools.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
using namespace WeightQuantBatchMatmulV2::Arch35;

namespace GMMSQWeightQuant {

struct GmmsqUbBufferInfo {
    uint64_t weightHighBitBufferNum;
    uint64_t weightHighBitTotalSize;
    uint64_t weightHighBitSingleBufferSize;
    uint64_t weightLowbitTotalSize;
    uint64_t weightLowBitSingleBufferSize;
    uint64_t maxB16TotalSize;
    uint64_t scaleE8M0TotalSize;
    uint64_t yFp8TotalSize;
    uint64_t gluResTotalSize;
    uint64_t mxScaleOutputTotalSize;
    uint64_t halfScaleTotalSize;
};

template <const VecAntiQuantConfig& vecConfig>
__aicore__ constexpr GmmsqUbBufferInfo GetGMMSQMxA8W4BufferInfo()
{
    return {
        // 环降为 2 深：反量化输出只占该区低 64KB，高 64KB 让给 yF32 独立使用（见 UpdateYF32Buffer）
        .weightHighBitBufferNum = DOUBLE_BUFFER_NUM,
        .weightHighBitTotalSize = 128 * GetKBUnit<int8_t>(),
        .weightHighBitSingleBufferSize = 128 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM,
        .weightLowbitTotalSize = 64 * GetKBUnit<int8_t>(),
        .weightLowBitSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
        .maxB16TotalSize = 2 * GetKBUnit<half>(),
        .scaleE8M0TotalSize = 16 * GetKBUnit<fp8_e8m0_t>(),
        .yFp8TotalSize = 32 * GetKBUnit<fp8_e4m3fn_t>(),
        .gluResTotalSize = 32 * GetKBUnit<half>(),
        .mxScaleOutputTotalSize = 1 * GetKBUnit<int8_t>(),
        .halfScaleTotalSize = 1 * GetKBUnit<half>(),
    };
}

#define GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename yType, typename yScaleType, const WqmmConfig& wqmmConfig, \
              const VecAntiQuantConfig& vecConfig, typename actType>

// 类声明处带默认实参（默认 SwiGLU）；成员类外定义只能使用不含默认值的版本
#define GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM_DEFAULT \
    template <typename xType, typename wType, typename yType, typename yScaleType, const WqmmConfig& wqmmConfig, \
              const VecAntiQuantConfig& vecConfig, typename actType = GmmVecActivationSwiGLU>

#define GMMSQ_WQ_VEC_COMPUTE_CLASS \
    GMMSQWeightQuantVecCompute<xType, wType, yType, yScaleType, wqmmConfig, vecConfig, actType>

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM_DEFAULT
class GMMSQWeightQuantVecCompute {
public:
    __aicore__ inline GMMSQWeightQuantVecCompute(){};

    __aicore__ inline void Init();
    __aicore__ inline void UpdateGlobalAddr(__gm__ wType* weight, const bool weightL2Cacheable);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset, uint64_t kL1Offset,
                                      const BasicBlockOffsetParam& offsetParam);
    __aicore__ inline void WeightAntiQuantComputeNzNk(uint64_t kRealSize, const LocalTensor<xType>& weightHighBitL1,
                                                      const BasicBlockOffsetParam& offsetParam);
    __aicore__ inline void Activation(const LocalTensor<float>& yF32, const BasicBlockOffsetParam& param);
    __aicore__ inline void SetActParams(float beta, float invBeta, float linearBeta, float invLinearBeta);
    __aicore__ inline void ComputeMaxExp(const LocalTensor<float>& yF32, const BasicBlockOffsetParam& param);
    __aicore__ inline void ComputeScale(const BasicBlockOffsetParam& param);
    __aicore__ inline void ComputeDataForQuantTargetFp8(const LocalTensor<float>& yF32,
                                                        const BasicBlockOffsetParam& param);
    __aicore__ inline void CopyOutputFromUb2Gm(const BasicBlockOffsetParam& param, __gm__ yType* yFp32Addr);
    __aicore__ inline void CopyScaleFromUb2Gm(const BasicBlockOffsetParam& param, __gm__ yScaleType* yScaleAddr);
    // 输出落盘排空（y 与 yScale 的 GM 拷贝共用 yFp8Buffer_/scaleE8M0Buffer_，两块
    // buffer 都不与 weightHighBit 区别名）。排空点由「本块 epilogue 末尾」推迟到
    // 「下一块 epilogue 开头」：期间 V 可以先做反量化展开/激活，与这次落盘重叠。
    __aicore__ inline void WaitQuantOutputMte3();
    // 排空 weightHighBit 环上在途的 MTE3（UB→L1）权重拷贝。
    // epilogue 的 gluRes_/quantScaleOutput_/halfScale_ 与该环内存别名，而块首的预取
    // 已经发出本块前几个 K 块的拷贝；不排空就写这些 buffer 会把尚未读完的权重写脏，
    // 表现为该块前 2 个分形（32 列）× 全行的输出偏离（极端时权重为垃圾 → max 为 +Inf
    // → scale 0xFF、y 全 0）。
    __aicore__ inline void WaitWeightHighBitMte3();
    __aicore__ inline void ZeroTailOutput(__gm__ yType* y, __gm__ yScaleType* yScale, uint64_t rowStart,
                                          uint64_t rowCount, uint64_t nOut, uint64_t nScalePerRow);
    __aicore__ inline void TransMxScaleLayout(const BasicBlockOffsetParam& param);
    __aicore__ inline void End();

private:
    __aicore__ inline void AntiQuantProcessNzMxA8W4(uint64_t ubMte2KSize, const BasicBlockOffsetParam& offsetParam);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t antiQuantRealN, uint64_t antiQuantRealK,
                                                       const LocalTensor<xType>& weightHighBitL1);

    constexpr static GmmsqUbBufferInfo UB_BUFFER_INFO = GetGMMSQMxA8W4BufferInfo<vecConfig>();

    uint64_t weightMte2LoopIdx_ = 0;
    uint64_t ubComputeLoopIdx_ = 0;

    // 专用排空 event id：weight 环用 0/1、quant 用 2，这里取 3，不动环上的信用计数。
    static constexpr uint32_t EVENT_ID_WEIGHT_DRAIN = 3;
    static constexpr uint32_t EVENT_ID_V_TO_MTE2 = 0;
    static constexpr uint32_t EVENT_ID_MTE2_TO_V = 0;
    static constexpr uint32_t EVENT_ID_WEIGHT_MTE3_TO_V = 0;
    static constexpr uint32_t EVENT_ID_QUANT_MTE3_TO_V =
        EVENT_ID_WEIGHT_MTE3_TO_V + UB_BUFFER_INFO.weightHighBitBufferNum;
    static constexpr uint32_t EVENT_ID_V_TO_MTE3 = 0;

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<yType> yFp32Global_;
    GlobalTensor<yScaleType> yScaleGlobal_;

    actType act_;
    float beta_ = 1.0f;
    float invBeta_ = 1.0f;
    float linearBeta_ = 1.0f;
    float invLinearBeta_ = 1.0f;

    LocalTensor<int8_t> weightLowBit_;
    LocalTensor<xType> weightHighBit_;
    LocalTensor<half> maxB16Buffer_;
    LocalTensor<fp8_e8m0_t> scaleE8M0Buffer_;
    LocalTensor<fp8_e4m3fn_t> yFp8Buffer_;
    LocalTensor<bfloat16_t> gluRes_;
    LocalTensor<int8_t> quantScaleOutput_;
    LocalTensor<uint16_t> halfScale_;

    constexpr static uint32_t C0_SIZE = C0_SIZE_B8;
    constexpr static uint64_t VEC_REG_ELEM = VECTOR_REG_WIDTH;
    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();
    constexpr static uint64_t DIVISOR_FACTOR_TWO = 2;
};

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::Init()
{
    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightLowbitTotalSize);
    uint64_t ubOffset = UB_BUFFER_INFO.weightLowbitTotalSize;

    weightHighBit_ = LocalTensor<xType>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.weightHighBitTotalSize);
    // Reuse weightHighBit_ space for SwiGLU + Quant intermediate buffers
    uint64_t reuseOffset = ubOffset;
    gluRes_ = LocalTensor<bfloat16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.gluResTotalSize);
    reuseOffset += UB_BUFFER_INFO.gluResTotalSize * sizeof(bfloat16_t);
    quantScaleOutput_ = LocalTensor<int8_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.mxScaleOutputTotalSize);
    reuseOffset += UB_BUFFER_INFO.mxScaleOutputTotalSize * sizeof(int8_t);
    halfScale_ = LocalTensor<uint16_t>(TPosition::LCM, reuseOffset, UB_BUFFER_INFO.halfScaleTotalSize);

    ubOffset += UB_BUFFER_INFO.weightHighBitTotalSize;

    maxB16Buffer_ = LocalTensor<half>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.maxB16TotalSize);
    ubOffset += UB_BUFFER_INFO.maxB16TotalSize * sizeof(half);

    scaleE8M0Buffer_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.scaleE8M0TotalSize);

    ubOffset += UB_BUFFER_INFO.scaleE8M0TotalSize * sizeof(fp8_e8m0_t);
    yFp8Buffer_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, ubOffset, UB_BUFFER_INFO.yFp8TotalSize);

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::UpdateGlobalAddr(__gm__ wType* weight, const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WaitVToMTE2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::SetVToMTE2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)));
    weightMte2LoopIdx_++;
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyGmToUb(uint64_t ubMte2KSize, uint64_t kGmOffset,
                                                              uint64_t kL1Offset,
                                                              const BasicBlockOffsetParam& offsetParam)
{
    if (offsetParam.nL1Size == 0 || ubMte2KSize == 0) {
        return;
    }

    // swiglu和gate N轴对半切分
    uint64_t swishNOffset = offsetParam.nOffset * static_cast<uint64_t>(C0_SIZE);
    uint64_t gateNOffset = (offsetParam.nOffset + offsetParam.nSize / 2) * static_cast<uint64_t>(C0_SIZE);
    uint64_t kOffset = (kGmOffset + kL1Offset) * offsetParam.nAlign;

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + swishNOffset], CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        CeilAlign(offsetParam.nL1Size / 2, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, offsetParam.nAlign * C0_SIZE);

    DataCopyPad2D(
        weightLowBit_[(weightMte2LoopIdx_ % vecConfig.ubMte2BufferNum) * UB_BUFFER_INFO.weightLowBitSingleBufferSize +
                      (offsetParam.nL1Size / DIVISOR_FACTOR_TWO * C0_SIZE / 2)]
            .template ReinterpretCast<wType>(),
        wGlobal_[kOffset + gateNOffset], CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
        CeilAlign(offsetParam.nL1Size / 2, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
        CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, offsetParam.nAlign * C0_SIZE);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID_MTE2_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WeightAntiQuantComputeNzNk(uint64_t kRealSize,
                                                                              const LocalTensor<xType>& weightHighBitL1,
                                                                              const BasicBlockOffsetParam& offsetParam)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_MTE3_TO_V +
                                (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));

    AntiQuantProcessNzMxA8W4(kRealSize, offsetParam);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    if (likely(kRealSize > 0)) {
        CopyWeightHighBitForAligned(offsetParam.nL1Size, kRealSize, weightHighBitL1);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_MTE3_TO_V +
                               (ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)));
    ubComputeLoopIdx_++;
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::AntiQuantProcessNzMxA8W4(uint64_t ubMte2KSize,
                                                                            const BasicBlockOffsetParam& offsetParam)
{
    MxA8W4NzParams<xType, wType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = weightMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign = CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType*)weightLowBit_[ubMte2BufferIdx * UB_BUFFER_INFO.weightLowBitSingleBufferSize].GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType*)
            weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(CeilAlign(offsetParam.nL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                   static_cast<uint64_t>(VECTOR_REG_WIDTH));
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.weightHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;

    GmmsqAntiQuantMxA8W4NzNkVf<xType, wType>(mxA8W4NzParams);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyWeightHighBitForAligned(
    uint64_t antiQuantRealN, uint64_t antiQuantRealK, const LocalTensor<xType>& weightHighBitL1)
{
    DataCopyParams params;
    params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                        CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;

    params.blockLen = VEC_REG_ELEM / ONE_BLK_SIZE;
    params.srcStride = (UB_BUFFER_INFO.weightHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;
    DataCopy(weightHighBitL1,
             weightHighBit_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.weightHighBitBufferNum - 1)) * VEC_REG_ELEM], params);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::SetActParams(float beta, float invBeta, float linearBeta,
                                                                float invLinearBeta)
{
    beta_ = beta;
    invBeta_ = invBeta;
    linearBeta_ = linearBeta;
    invLinearBeta_ = invLinearBeta;
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::Activation(const LocalTensor<float>& yF32,
                                                              const BasicBlockOffsetParam& param)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / DIVISOR_FACTOR_TWO;

    __ubuf__ float* yAddr = (__ubuf__ float*)yF32.GetPhyAddr();
    __ubuf__ float* gateAddr = yAddr + nHalfSize;
    __ubuf__ bfloat16_t* gluResAddr = (__ubuf__ bfloat16_t*)gluRes_.GetPhyAddr();

    uint32_t nSrcUbAligned = CeilAlign(static_cast<uint32_t>(param.nL1Size), AscendC::ONE_BLK_SIZE / sizeof(float));
    uint32_t nDstUbAligned = CeilAlign(static_cast<uint32_t>(nHalfSize), AscendC::ONE_BLK_SIZE);
    constexpr uint16_t sizePerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t oneRowRepeatTimes = CeilDivide(static_cast<uint64_t>(nHalfSize), static_cast<uint64_t>(sizePerRepeat));

    act_.Compute(yAddr, gateAddr, gluResAddr, static_cast<uint16_t>(mRealSize), static_cast<uint16_t>(nHalfSize),
                 nSrcUbAligned, nDstUbAligned, oneRowRepeatTimes, sizePerRepeat, beta_, invBeta_, linearBeta_,
                 invLinearBeta_);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeMaxExp(const LocalTensor<float>& yF32,
                                                                 const BasicBlockOffsetParam& param)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / 2; // 2：N方向

    uint32_t nDstUbAligned = CeilAlign(static_cast<uint32_t>(nHalfSize), AscendC::ONE_BLK_SIZE);
    uint32_t totalDataInUb = static_cast<uint32_t>(mRealSize * nDstUbAligned);
    uint32_t vlForHalfNumber = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    uint16_t elementAfterReduce =
        AscendC::VECTOR_REG_WIDTH /
        static_cast<uint32_t>(32); // 32：一次载入两个VECTOR_REG_WIDTH，且每32个数取一个值，所以公式为
                                   // AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t) * 2 / 32 简化为 VECTOR_REG_WIDTH/32
    uint16_t loopDataNum = CeilDivide(static_cast<uint64_t>(totalDataInUb), static_cast<uint64_t>(vlForHalfNumber) * 2);

    __ubuf__ bfloat16_t* gluResAddr = (__ubuf__ bfloat16_t*)gluRes_.GetPhyAddr();
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxB16Buffer_.GetPhyAddr();

    ComputeMaxExpVf(gluResAddr, maxExpAddr, totalDataInUb, loopDataNum, vlForHalfNumber, elementAfterReduce);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeScale(const BasicBlockOffsetParam& param)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / DIVISOR_FACTOR_TWO;

    uint32_t nDstUbAligned = CeilAlign(static_cast<uint32_t>(nHalfSize), AscendC::ONE_BLK_SIZE);
    uint32_t totalDataInUb = static_cast<uint32_t>(mRealSize * nDstUbAligned);
    uint32_t totalScaleInUb = totalDataInUb / static_cast<uint32_t>(32); // 32：Mx量化的 groupSize 是32
    uint32_t vlForHalfNumber = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    uint16_t loopScaleNum = CeilDivide(static_cast<uint64_t>(totalScaleInUb), static_cast<uint64_t>(vlForHalfNumber));

    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxB16Buffer_.GetPhyAddr();
    __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)halfScale_.GetPhyAddr();

    uint16_t fpEmax = 0;
    if constexpr (IsSameType<yType, fp8_e4m3fn_t>::value) {
        fpEmax = static_cast<uint16_t>(0x0400);
    }

    ComputeScaleVf(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUb, loopScaleNum, vlForHalfNumber,
                   fpEmax);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ComputeDataForQuantTargetFp8(const LocalTensor<float>& yF32,
                                                                                const BasicBlockOffsetParam& param)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / DIVISOR_FACTOR_TWO;

    uint32_t nDstUbAligned = CeilAlign(static_cast<uint32_t>(nHalfSize), AscendC::ONE_BLK_SIZE);
    uint32_t totalDataInUb = static_cast<uint32_t>(mRealSize * nDstUbAligned);
    uint32_t vlForHalfNumber = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    uint16_t elementAfterReduce = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(32);
    uint16_t loopDataNum = CeilDivide(static_cast<uint64_t>(totalDataInUb), static_cast<uint64_t>(vlForHalfNumber) * 2);

    __ubuf__ bfloat16_t* gluResAddr = (__ubuf__ bfloat16_t*)gluRes_.GetPhyAddr();
    __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)halfScale_.GetPhyAddr();
    __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)yFp8Buffer_.GetPhyAddr();

    if constexpr (IsSameType<yType, fp8_e4m3fn_t>::value) {
        ComputeDataForQuantTargetFp8Vf<yType>(gluResAddr, halfScaleLocalAddr, outLocalAddr, totalDataInUb, loopDataNum,
                                              vlForHalfNumber, elementAfterReduce);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyOutputFromUb2Gm(const BasicBlockOffsetParam& param,
                                                                       __gm__ yType* yFp32Addr)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    yFp32Global_.SetGlobalBuffer(yFp32Addr);
    uint64_t nHalfSize = param.nL1Size / DIVISOR_FACTOR_TWO;

    uint64_t yGmOffset = (param.mOffset + AscendC::GetSubBlockIdx() * CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) *
                             CeilDivide(param.nSize, DIVISOR_FACTOR_TWO) +
                         param.nOffset;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yFp32Global_[yGmOffset], yFp8Buffer_, mRealSize, nHalfSize, nHalfSize,
                  param.nSize / DIVISOR_FACTOR_TWO);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::CopyScaleFromUb2Gm(const BasicBlockOffsetParam& param,
                                                                      __gm__ yScaleType* yScaleAddr)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    yScaleGlobal_.SetGlobalBuffer(yScaleAddr);
    uint64_t nGroupCount = CeilDivide(param.nL1Size / DIVISOR_FACTOR_TWO, static_cast<uint64_t>(MX_GROUPSIZE));
    uint64_t nTotalGroupCount = CeilDivide(param.nSize / DIVISOR_FACTOR_TWO, static_cast<uint64_t>(MX_GROUPSIZE));

    uint64_t scaleGmOffset =
        (param.mOffset + AscendC::GetSubBlockIdx() * CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) * nTotalGroupCount +
        param.nOffset / MX_GROUPSIZE;

    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    DataCopyPad2D(yScaleGlobal_[scaleGmOffset], scaleE8M0Buffer_, mRealSize, nGroupCount, nGroupCount,
                  nTotalGroupCount);

    // 这里不再排空：y/yScale 的 GM 拷贝是异步 MTE3，源 buffer 要到下一块 epilogue
    // 才会被覆写，排空改由 WaitQuantOutputMte3() 在下一块 epilogue 开头完成，
    // 让本次落盘与中间的反量化展开重叠。kernel 出口在 End() 里统一排空。
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WaitWeightHighBitMte3()
{
    // MTE3 队列屏障：Set 落在 MTE3 流里（排在已发出的 UB→L1 权重拷贝之后），
    // Wait 让 V 等它们全部完成。与 CopyGmToUb 里的 MTE2_V 屏障同款，不改变环的信用计数。
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_DRAIN);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_DRAIN);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::WaitQuantOutputMte3()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_QUANT_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_QUANT_MTE3_TO_V);
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::ZeroTailOutput(__gm__ yType* y, __gm__ yScaleType* yScale,
                                                                 uint64_t rowStart, uint64_t rowCount, uint64_t nOut,
                                                                 uint64_t nScalePerRow)
{
    if (rowCount == 0) {
        return;
    }
    // 复用 fp8 输出 buffer 作为清零源：本函数只在全部 group 处理完之后调用，该 buffer 已空闲。
    // yType/yScaleType 均为 1 字节，故按字节搬运。
    constexpr uint64_t ZERO_SRC_BYTES = 32 * 1024;
    constexpr uint64_t DUP_CHUNK = 4096;
    LocalTensor<int8_t> zeroSrc = yFp8Buffer_.template ReinterpretCast<int8_t>();
    // 复用该 buffer 前先等上一轮 MTE3 搬出完成，否则会把仍在读取的源数据覆盖掉
    SetFlag<HardEvent::MTE3_V>(EVENT_ID_QUANT_MTE3_TO_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID_QUANT_MTE3_TO_V);
    for (uint64_t i = 0; i < ZERO_SRC_BYTES; i += DUP_CHUNK) {
        Duplicate<int8_t>(zeroSrc[i], static_cast<int8_t>(0), static_cast<int32_t>(DUP_CHUNK));
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

    GlobalTensor<int8_t> yGm;
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y) + rowStart * nOut);
    uint64_t totalBytes = rowCount * nOut;
    for (uint64_t off = 0; off < totalBytes; off += ZERO_SRC_BYTES) {
        uint64_t remain = totalBytes - off;
        uint32_t chunk = static_cast<uint32_t>(remain < ZERO_SRC_BYTES ? remain : ZERO_SRC_BYTES);
        DataCopyExtParams copyParams{1, chunk, 0, 0, 0};
        DataCopyPad(yGm[off], zeroSrc, copyParams);
    }

    GlobalTensor<int8_t> yScaleGm;
    yScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(yScale) + rowStart * nScalePerRow);
    uint64_t totalScaleBytes = rowCount * nScalePerRow;
    for (uint64_t off = 0; off < totalScaleBytes; off += ZERO_SRC_BYTES) {
        uint64_t remain = totalScaleBytes - off;
        uint32_t chunk = static_cast<uint32_t>(remain < ZERO_SRC_BYTES ? remain : ZERO_SRC_BYTES);
        DataCopyExtParams copyParams{1, chunk, 0, 0, 0};
        DataCopyPad(yScaleGm[off], zeroSrc, copyParams);
    }
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::TransMxScaleLayout(const BasicBlockOffsetParam& param)
{
    uint64_t mRealSize = AscendC::GetSubBlockIdx() ? (param.mL1Size - CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO)) :
                                                     CeilDivide(param.mL1Size, DIVISOR_FACTOR_TWO);
    if (mRealSize == 0 || param.mL1Size == 0) {
        return;
    }
    uint64_t nHalfSize = param.nL1Size / 2;
    uint64_t scaleBlockN = CeilDivide(nHalfSize, static_cast<uint64_t>(64)) * DIVISOR_FACTOR_TWO;

    __ubuf__ int8_t* quantScaleOutputInUbAddr = (__ubuf__ int8_t*)quantScaleOutput_.GetPhyAddr();
    __ubuf__ int8_t* quantScaleBlockOutputInUbAddr = (__ubuf__ int8_t*)scaleE8M0Buffer_.GetPhyAddr();

    TransMxScaleLayoutVf(quantScaleOutputInUbAddr, quantScaleBlockOutputInUbAddr, static_cast<uint16_t>(mRealSize),
                         static_cast<uint16_t>(scaleBlockN));
}

GMMSQ_WQ_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void GMMSQ_WQ_VEC_COMPUTE_CLASS::End()
{
    // 最后一块的 y/yScale 落盘在 CopyScaleFromUb2Gm 里已不再排空，kernel 出口补一次，
    // 保证 GM 写完再退出。
    WaitQuantOutputMte3();

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.weightHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_WEIGHT_MTE3_TO_V + idx);
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + idx);
    }
}

} // namespace GMMSQWeightQuant

#endif
