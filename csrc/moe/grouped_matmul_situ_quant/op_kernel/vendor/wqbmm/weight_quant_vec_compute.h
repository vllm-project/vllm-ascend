/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file weight_quant_vec_compute.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
#define GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H

#include "anti_quant_y_vf.h"
#include "basic_block_config.h"
#include "basic_block_vf_mx.h"
#include "basic_block_vf_nd.h"
#include "basic_block_vf_nz.h"
#include "mx_a8w4_scale_trans_id_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "weight_quant_tool.h"

using AscendC::BLOCK_CUBE;
using AscendC::CacheMode;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::GetRound;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::TypeGet;

namespace WeightQuantBatchMatmulV2::Arch35 {

#define GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM                                                        \
    template <typename xType, typename wType, typename antiQuantScaleType, typename biasType, typename yType,          \
              const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS                                                                 \
    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType, wqmmConfig, vecConfig>

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
class BasicBlockLibVectorAntiQuantCompute {
public:
    __aicore__ inline BasicBlockLibVectorAntiQuantCompute(){};

    __aicore__ inline void UpdateGlobalAddr(__gm__ wType *weight, __gm__ antiQuantScaleType *antiQuantScale,
                                            __gm__ xType *antiQuantOffset, __gm__ float *perTokenScale,
                                            __gm__ float *perChannelScale, __gm__ biasType *bias,
                                            const bool weightL2Cacheable);
    __aicore__ inline void Init(TPipe *tPipe, bool hasBias);
    __aicore__ inline void InitKCG(uint32_t antiQuantGroupSize, bool hasBias,
                                   const LocalTensor<xType> &ubHighBitTotalBuffer, uint64_t highBitUbOffset);
    __aicore__ inline void WaitVToMTE2();
    __aicore__ inline void SetVToMTE2();
    __aicore__ inline void CopyWeightMxScaleGmToUb(uint64_t nOffset, uint64_t nL1Size, uint64_t kOffset,
                                                   uint64_t mxGroupNum, uint64_t realGroupNum);
    __aicore__ inline void TransWeightMxScale(uint64_t nSize, uint64_t kSize);
    __aicore__ inline void CopyWeightMxScaleUbToL1(LocalTensor<uint16_t> &scaleL1, uint64_t l1NOffset,
                                                   uint64_t nSize);
    __aicore__ inline void CopyGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                      uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyMxBiasGmToUb(uint64_t ubMte2MxBiasNSize, uint64_t ubMte2MxBiasNOffset);
    __aicore__ inline void WeightAntiQuantCompute(const UbConsumeConfig &ubConsumeConfig,
                                                  const LocalTensor<xType> &weightHighBitL1,
                                                  const L1ConsumeConfig &l1ConsumeConfig,
                                                  const LocalTensor<biasType> *biasL1Ptr);
    __aicore__ inline void WeightAntiQuantComputeNzNk(const UbConsumeConfig &ubConsumeConfig,
                                                      const LocalTensor<xType> &weightHighBitL1,
                                                      const L1ConsumeConfig &l1ConsumeConfig,
                                                      const LocalTensor<biasType> &biasL1);
    __aicore__ inline void CopyKcScaleBiasGmToUb(uint64_t nRealL0Size, uint64_t mRealL0Size, uint64_t nOffset,
                                                 uint64_t mOffset);
    __aicore__ inline void AntiQuantYWithKc(uint64_t nRealL0Size, uint64_t mRealL0Size);
    __aicore__ inline void CopyYUbToGm(uint64_t nRealL0Size, uint64_t mRealL0Size, __gm__ half *yGm,
                                       const BasicBlockOffsetParam &offsetParam, uint64_t aivMOffset);
    __aicore__ inline void End();

private:
    __aicore__ inline void InitMx();
    __aicore__ inline void CopyWeightGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                            uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void CopyAntiQuantParamsGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset,
                                                     uint64_t ubMte2KOffset, const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void WeightAntiQuantProcess(uint64_t nRealLen, uint64_t kRealLen, uint64_t antiQuantNOffset,
                                                  uint64_t antiQuantKOffset, const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void AntiQuantProcess(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                            uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
                                            const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void AntiQuantProcessNd(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                              uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantProcessNz(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                              uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void CalLocalAddrForVf(uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
                                             LocalAddressParam<xType, wType> &localAddressParam);
    __aicore__ inline void CalInt4NzKnLocalAddrForVf(uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
                                                     Int4NzParams<xType, wType, antiQuantScaleType> &int4NzParams);
    __aicore__ inline void WeightHighBitUbToL1(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                               uint64_t antiQuantRealK, const LocalTensor<xType> &weightHighBitL1,
                                               uint64_t l1RealExternalLen);
    __aicore__ inline uint64_t ComputeWeightHighBitL1Offset(uint64_t antiQuantNOffset, uint64_t antiQuantKOffset,
                                                            uint64_t nRealLen, uint64_t kRealLen,
                                                            const L1ConsumeConfig &l1ConsumeConfig);
    __aicore__ inline void CopyMxAntiQuantParamsGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize,
                                                       uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
                                                       const BasicBlockOffsetParam &offsetParam);
    __aicore__ inline void MxScaleProcess(uint64_t ubMte2NSize, uint64_t ubMte2KSize);
    __aicore__ inline void AntiQuantProcessNdMx(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                                uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantProcessNzMx(uint64_t vfExternalRealLen, uint64_t vfInnerRealLen,
                                                uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset);
    __aicore__ inline void AntiQuantProcessNzMxA8W4(const UbConsumeConfig &ubConsumeConfig);
    __aicore__ inline void CalLocalAddrForYVf(LocalAddressYParam<yType> &localAddressParam);
    __aicore__ inline void CopyWeightHighBitForAligned(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                                       const uint64_t antiQuantRealK,
                                                       const LocalTensor<xType> &weightHighBitL1);
    __aicore__ inline void CopyWeightHighBitForUnaligned(uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN,
                                                         uint64_t antiQuantRealK,
                                                         const LocalTensor<xType> &weightHighBitL1);

    // mte2搬运计数，用于控制weight输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2LoopIdx_ = 0;
    // mte2搬运计数，用于控制antiquantY输入的buffer和 mte2&&V间同步控制
    uint64_t ubMte2AntiquantYLoopIdx_ = 0;
    // vf中标准计算单元(vfNStandardLen, vfKStandardLen)的计数，用于控制weight反量化后输出的buffer和V&&mte3间同步控制
    uint64_t ubComputeLoopIdx_ = 0;
    uint64_t ubAntiquantYLoopIdx_ = 0;

    TEventID vecEventIdVToMte2_[QUADRUPLE_BUFFER_NUM];
    TEventID vecEventIdMte3ToV_[QUADRUPLE_BUFFER_NUM];
    TEventID vecEventIdTransScaleMte3ToV_[DOUBLE_BUFFER_NUM];
    uint64_t ubTransScaleLoopIdx_ = 0;

    xType scaleValue_;
    xType offsetValue_;

    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> antiQuantOffsetGlobal_;
    GlobalTensor<antiQuantScaleType> antiQuantScaleGlobal_;
    GlobalTensor<float> antiQuantYPerTokenScaleGlobal_;
    GlobalTensor<float> antiQuantYPerChannelScaleGlobal_;
    GlobalTensor<float> antiQuantYBiasGlobal_;
    GlobalTensor<half> antiQuantYF16Global_;
    GlobalTensor<biasType> biasGlobal_;

    LocalTensor<int8_t> ubWeightInputLowBitTotalBuffer_;
    LocalTensor<xType> ubHighBitTotalBuffer_;
    LocalTensor<antiQuantScaleType> ubAntiQuantScaleTotalBuffer_;
    LocalTensor<xType> ubAntiQuantScaleAfterCastTotalBuffer_;
    LocalTensor<xType> ubAntiQuantOffsetTotalBuffer_;
    LocalTensor<float> ubAntiQuantYPerTokenScaleTotalBuffer_;
    LocalTensor<float> ubAntiQuantYPerChannelScaleTotalBuffer_;
    LocalTensor<float> ubAntiQuantYBiasTotalBuffer_;
    LocalTensor<biasType> ubBiasTotalBuffer_;
    LocalTensor<biasType> ubBiasOutTotalBuffer_;
    LocalTensor<uint16_t> transIdBuffer_;
    LocalTensor<uint16_t> transScaleBuffer_;

    LocalTensor<uint64_t> ubAntiQuantScaleMaskBuffer_;

    uint64_t antiQuantGroupSize_;
    bool hasBias_;

    constexpr static uint32_t C0_SIZE =
        (IsSameType<xType, int8_t>::value || IsSameType<xType, fp8_e4m3fn_t>::value) ? C0_SIZE_B8 : BLOCK_CUBE;
    constexpr static uint64_t VEC_REG_ELEM =
        IsMxA8W4<xType, wqmmConfig.antiQuantType>() ? VECTOR_REG_WIDTH : VEC_MAX_ELEM_B16;

    constexpr static uint64_t UB_AVAILABLE_SIZE = 248 * GetKBUnit<int8_t>();

    constexpr static uint64_t ANTI_QUANT_Y_PER_TOKEN_SCALE_TOTAL_BUFFER_SIZE = 2 * GetKBUnit<float>();
    constexpr static uint64_t ANTI_QUANT_Y_PER_CHANNEL_SCALE_TOTAL_BUFFER_SIZE = 2 * GetKBUnit<float>();
    constexpr static uint64_t ANTI_QUANT_Y_BIAS_TOTAL_BUFFER_SIZE = 2 * GetKBUnit<float>();

    constexpr static uint64_t UB_ANTI_QUANT_Y_BUFFER_NUM = DOUBLE_BUFFER_NUM;

    constexpr static uint64_t ANTI_QUANT_Y_PER_TOKEN_SCALE_SINGLE_BUFFER_SIZE =
        ANTI_QUANT_Y_PER_TOKEN_SCALE_TOTAL_BUFFER_SIZE / UB_ANTI_QUANT_Y_BUFFER_NUM;
    constexpr static uint64_t ANTI_QUANT_Y_PER_CHANNEL_SCALE_SINGLE_BUFFER_SIZE =
        ANTI_QUANT_Y_PER_CHANNEL_SCALE_TOTAL_BUFFER_SIZE / UB_ANTI_QUANT_Y_BUFFER_NUM;
    constexpr static uint64_t ANTI_QUANT_Y_BIAS_SINGLE_BUFFER_SIZE =
        ANTI_QUANT_Y_BIAS_TOTAL_BUFFER_SIZE / UB_ANTI_QUANT_Y_BUFFER_NUM;

    TEventID vecEventIdAntiQuantYVToMte2_[UB_ANTI_QUANT_Y_BUFFER_NUM];

    constexpr static UbBufferInfo UB_BUFFER_INFO = GetBufferConfig<xType, wqmmConfig, vecConfig>();
    constexpr static VfConfig VF_CONFIG = GetVfConfig<xType, wqmmConfig, vecConfig>();

    constexpr static uint64_t ANTIQUANT_Y_STANDARD_N_SIZE = VECTOR_REG_WIDTH / sizeof(int32_t);

    constexpr static uint64_t TRANS_SCALE_SINGLE_BUF_SIZE = 16 * GetKBUnit<uint16_t>();
};

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ wType *weight, __gm__ antiQuantScaleType *antiQuantScale, __gm__ xType *antiQuantOffset,
    __gm__ float *perTokenScale, __gm__ float *perChannelScale, __gm__ biasType *bias, const bool weightL2Cacheable)
{
    wGlobal_.SetGlobalBuffer(weight);
    antiQuantScaleGlobal_.SetGlobalBuffer(antiQuantScale);

    if constexpr (IsSameType<xType, int8_t>::value) {
        antiQuantYPerTokenScaleGlobal_.SetGlobalBuffer(perTokenScale);
        antiQuantYPerChannelScaleGlobal_.SetGlobalBuffer(perChannelScale);
        antiQuantYBiasGlobal_.SetGlobalBuffer(bias);
    } else {
        antiQuantOffsetGlobal_.SetGlobalBuffer(antiQuantOffset);
    }

    if (!weightL2Cacheable) {
        wGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    if (hasBias_) {
        biasGlobal_.SetGlobalBuffer(bias);
    }
}

/*
 * 初始化buffer和同步所需的EventID
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::Init(TPipe *tPipe, bool hasBias)
{
    hasBias_ = hasBias;
    if constexpr (wqmmConfig.antiQuantType == QuantType::MX) {
        InitMx();
    } else {
        ubWeightInputLowBitTotalBuffer_ =
            LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightInputLowbitUbTotalSize);
        uint64_t highBitOffset = UB_BUFFER_INFO.weightInputLowbitUbTotalSize;
        ubHighBitTotalBuffer_ =
            LocalTensor<xType>(TPosition::LCM, highBitOffset, UB_BUFFER_INFO.highBitDataUbTotalSize);
        uint64_t scaleOffset = highBitOffset + UB_BUFFER_INFO.highBitDataUbTotalSize * sizeof(xType);
        ubAntiQuantScaleTotalBuffer_ =
            LocalTensor<antiQuantScaleType>(TPosition::LCM, scaleOffset, UB_BUFFER_INFO.antiQuantScaleUbTotalSize);
        uint64_t offsetOffset = scaleOffset + UB_BUFFER_INFO.antiQuantScaleUbTotalSize * sizeof(antiQuantScaleType);
        ubAntiQuantOffsetTotalBuffer_ =
            LocalTensor<xType>(TPosition::LCM, offsetOffset, UB_BUFFER_INFO.antiQuantOffsetUbTotalSize);
    }

    for (uint16_t i = 0; i < vecConfig.ubMte2BufferNum; ++i) {
        vecEventIdVToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    }

    for (uint16_t i = 0; i < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; ++i) {
        vecEventIdMte3ToV_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }
}

/*
 * 初始化mx场景buffer 封装防止Init超长
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::InitMx()
{
    if constexpr (IsSameType<xType, fp8_e4m3fn_t>::value) {
        // MxA8W4
        ubWeightInputLowBitTotalBuffer_ =
            LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightInputLowbitUbTotalSize);
        uint64_t highBitOffset = UB_BUFFER_INFO.weightInputLowbitUbTotalSize;
        ubHighBitTotalBuffer_ =
            LocalTensor<xType>(TPosition::LCM, highBitOffset, UB_BUFFER_INFO.highBitDataUbTotalSize);
        uint64_t currentOffset = highBitOffset + UB_BUFFER_INFO.highBitDataUbTotalSize;
        if (hasBias_) {
            ubBiasTotalBuffer_ = LocalTensor<biasType>(TPosition::LCM, currentOffset, UB_BUFFER_INFO.biasUbTotalSize);
            currentOffset += UB_BUFFER_INFO.biasUbTotalSize * sizeof(biasType);
            ubBiasOutTotalBuffer_ =
                LocalTensor<biasType>(TPosition::LCM, currentOffset, UB_BUFFER_INFO.biasReducedUbTotalSize);
            currentOffset += UB_BUFFER_INFO.biasReducedUbTotalSize * sizeof(biasType);
        }
        constexpr uint64_t transIdBufferSize = GetKBUnit<uint16_t>();
        transIdBuffer_ = LocalTensor<uint16_t>(TPosition::LCM, currentOffset, transIdBufferSize);
        currentOffset += transIdBufferSize * sizeof(uint16_t);
        GlobalTensor<uint16_t> transIdGm;
        transIdGm.SetGlobalBuffer((__gm__ uint16_t *)(MX_TRANS_ID));
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(MX_TRANS_ID), 256);
#endif
        DataCopyParams dmaParams;
        dmaParams.blockCount = 1;
        dmaParams.blockLen = 8;
        DataCopy(transIdBuffer_, transIdGm, dmaParams);
        transScaleBuffer_ =
            LocalTensor<uint16_t>(TPosition::LCM, currentOffset, DOUBLE_BUFFER_NUM * TRANS_SCALE_SINGLE_BUF_SIZE);
        for (uint16_t i = 0; i < DOUBLE_BUFFER_NUM; ++i) {
            vecEventIdTransScaleMte3ToV_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::MTE3_V>(vecEventIdTransScaleMte3ToV_[i]);
        }
    } else {
        ubWeightInputLowBitTotalBuffer_ =
            LocalTensor<int8_t>(TPosition::LCM, 0, UB_BUFFER_INFO.weightInputLowbitUbTotalSize);
        uint64_t highBitOffset = UB_BUFFER_INFO.weightInputLowbitUbTotalSize;
        ubHighBitTotalBuffer_ =
            LocalTensor<xType>(TPosition::LCM, highBitOffset, UB_BUFFER_INFO.highBitDataUbTotalSize);
        uint64_t scaleOffset = highBitOffset + UB_BUFFER_INFO.highBitDataUbTotalSize * sizeof(xType);
        ubAntiQuantScaleTotalBuffer_ =
            LocalTensor<antiQuantScaleType>(TPosition::LCM, scaleOffset, UB_BUFFER_INFO.antiQuantScaleUbTotalSize);
        uint64_t scaleAfterCastOffset =
            scaleOffset + UB_BUFFER_INFO.antiQuantScaleUbTotalSize * sizeof(antiQuantScaleType);
        ubAntiQuantScaleAfterCastTotalBuffer_ =
            LocalTensor<xType>(TPosition::LCM, scaleAfterCastOffset, UB_BUFFER_INFO.antiQuantScaleAfterCastUbTotalSize);
    }
}

/**
 * @brief KCG 初始化buffer和同步所需的EventID
 * @param antiQuantgroupSize per_group伪量化的groupSize
 * @param hasBias Y反量化是否存在bias
 * @param ubHighBitTotalBuffer weightS8/cS32/cF16 复用的UB
 * @param highBitUbOffset weightS8/cS32/cF16 复用UB的偏移长度
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::InitKCG(uint32_t antiQuantGroupSize, bool hasBias,
                                                        const LocalTensor<xType> &ubHighBitTotalBuffer,
                                                        uint64_t highBitUbOffset)
{
    antiQuantGroupSize_ = antiQuantGroupSize;
    hasBias_ = hasBias;

    ubHighBitTotalBuffer_ = ubHighBitTotalBuffer;
    ubWeightInputLowBitTotalBuffer_ =
        LocalTensor<int8_t>(TPosition::LCM, highBitUbOffset, UB_BUFFER_INFO.weightInputLowbitUbTotalSize);
    uint64_t scaleOffset = highBitUbOffset + UB_BUFFER_INFO.weightInputLowbitUbTotalSize;
    ubAntiQuantScaleTotalBuffer_ =
        LocalTensor<antiQuantScaleType>(TPosition::LCM, scaleOffset, UB_BUFFER_INFO.antiQuantScaleUbTotalSize);
    uint64_t perTokenScaleOffset = scaleOffset + UB_BUFFER_INFO.antiQuantScaleUbTotalSize * sizeof(antiQuantScaleType);
    ubAntiQuantYPerTokenScaleTotalBuffer_ =
        LocalTensor<float>(TPosition::LCM, perTokenScaleOffset, ANTI_QUANT_Y_PER_TOKEN_SCALE_TOTAL_BUFFER_SIZE);
    uint64_t perChannelScaleOffset =
        perTokenScaleOffset + ANTI_QUANT_Y_PER_TOKEN_SCALE_TOTAL_BUFFER_SIZE * sizeof(float);
    ubAntiQuantYPerChannelScaleTotalBuffer_ =
        LocalTensor<float>(TPosition::LCM, perChannelScaleOffset, ANTI_QUANT_Y_PER_CHANNEL_SCALE_TOTAL_BUFFER_SIZE);
    uint64_t yBiasOffset = perChannelScaleOffset + ANTI_QUANT_Y_PER_CHANNEL_SCALE_TOTAL_BUFFER_SIZE * sizeof(float);
    ubAntiQuantYBiasTotalBuffer_ = LocalTensor<float>(TPosition::LCM, yBiasOffset, ANTI_QUANT_Y_BIAS_TOTAL_BUFFER_SIZE);
    uint64_t maskOffset = yBiasOffset + ANTI_QUANT_Y_BIAS_TOTAL_BUFFER_SIZE * sizeof(float);
    ubAntiQuantScaleMaskBuffer_ =
        LocalTensor<uint64_t>(TPosition::LCM, maskOffset, UB_BUFFER_INFO.antiQuantScaleMaskBufferSize);
    ubAntiQuantScaleMaskBuffer_.SetValue(0, 0x00000000ffffffff);
    ubAntiQuantScaleMaskBuffer_.SetValue(1, 0x00000000ffffffff);
    ubAntiQuantScaleMaskBuffer_.SetValue(2, 0x00000000ffffffff);
    ubAntiQuantScaleMaskBuffer_.SetValue(3, 0x00000000ffffffff);

    for (uint16_t i = 0; i < vecConfig.ubMte2BufferNum; ++i) {
        vecEventIdVToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    }

    for (uint16_t i = 0; i < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; ++i) {
        vecEventIdMte3ToV_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }

    for (uint16_t i = 0; i < UB_ANTI_QUANT_Y_BUFFER_NUM; ++i) {
        vecEventIdAntiQuantYVToMte2_[i] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WaitVToMTE2()
{
    // 用临时变量接一下，优化编译的作用
    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    if (likely(ubMte2LoopIdx_ > vecConfig.ubMte2BufferNum - 1)) {
        if constexpr (vecConfig.ubMte2BufferNum == 2 || vecConfig.ubMte2BufferNum == 4) {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[ubMte2LoopIdx_ & (vecConfig.ubMte2BufferNum - 1)]);
        } else {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum]);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::SetVToMTE2()
{
    // 用临时变量接一下，优化编译的作用
    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    if constexpr (vecConfig.ubMte2BufferNum == 2 || vecConfig.ubMte2BufferNum == 4) {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[(ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)]);
    } else {
        SetFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[(ubMte2LoopIdx_ - 1) % vecConfig.ubMte2BufferNum]);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam)
{
    // ubMte2NSize和ubMte2KSize为实际MTE2搬运到UB的有效数据，
    // 其按照ubMte2InnerSize进行跳写，垃圾数据无需操作，搬出的时搬运有效数据即可。
    if (ubMte2NSize == 0 || ubMte2KSize == 0) {
        ubMte2LoopIdx_++;  // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
        return;
    }

    if constexpr (wqmmConfig.antiQuantType == QuantType::MX) {
        if constexpr (IsSameType<xType, fp8_e4m3fn_t>::value) {
            DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                          UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                              .template ReinterpretCast<wType>(),
                          wGlobal_[ubMte2KOffset * offsetParam.nAlign + ubMte2NOffset * static_cast<uint64_t>(C0_SIZE)],
                          CeilDivide(ubMte2KSize, static_cast<uint64_t>(C0_SIZE)),
                          CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                          CeilAlign(ubMte2NSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                          offsetParam.nAlign * C0_SIZE);
        } else {
            CopyAntiQuantParamsGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
            CopyWeightGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
        }
    } else {
        CopyWeightGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
        CopyAntiQuantParamsGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
    }

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    ubMte2LoopIdx_++;
}

/**
 * @brief 该函数搬运weight数据从GM进入UB上ubWeightInputLowBitTotalBuffer_中
 *        其分为vecConfig.ubMte2BufferNum块 每块大小为weightInputLowBitUbSingleBufferSize_
 * @param ubMte2NSize 从GM上搬运到UB的N方向大小
 * @param ubMte2KSize 从GM上搬运到UB的K方向大小
 * @param ubMte2NOffset 从GM上搬运到UB时, GM上N方向的偏移
 * @param ubMte2KOffset 从GM上搬运到UB时, GM上N方向的偏移
 * @param offsetParam 存储Weight矩阵的原始N,K,kAlign信息,用于搬运时GM上地址偏移的计算
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightGmToUb(uint64_t ubMte2NSize, uint64_t ubMte2KSize,
                                                                 uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
                                                                 const BasicBlockOffsetParam &offsetParam)
{
    if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
        if constexpr (!wqmmConfig.bTrans) {
            DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                          UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                              .template ReinterpretCast<wType>(),
                          wGlobal_[ubMte2KOffset * offsetParam.nSize + ubMte2NOffset], ubMte2KSize, ubMte2NSize,
                          vecConfig.ubMte2InnerSize, offsetParam.nSize);
        } else {
            DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                          UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                              .template ReinterpretCast<wType>(),
                          wGlobal_[ubMte2NOffset * offsetParam.kSize + ubMte2KOffset], ubMte2NSize, ubMte2KSize,
                          vecConfig.ubMte2InnerSize, offsetParam.kSize);
        }
    } else {
        if constexpr (!wqmmConfig.bTrans) {
            DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                          UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                              .template ReinterpretCast<wType>(),
                          wGlobal_[ubMte2NOffset * offsetParam.kAlign + ubMte2KOffset * static_cast<uint64_t>(C0_SIZE)],
                          CeilDivide(ubMte2NSize, static_cast<uint64_t>(C0_SIZE)),
                          CeilAlign(ubMte2KSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                          vecConfig.ubMte2InnerSize * C0_SIZE, offsetParam.kAlign * C0_SIZE);
        }
    }
}

/**
 * @brief 该函数搬运scale和offset数据从GM进入UB上ubAntiQuantScaleTotalBuffer_中
 *        pertensor场景读取单个数据即可，perchannel场景下搬运[1, ubMte2NSize]大小, 按照VECTOR_REG_WIDTH(256)对齐写入
 *        A8W4场景将perGroupScale搬入antiQuantPerGroupScaleGlobal_中，
 *        搬运[CeilDiv(ubMte2KOffset, antiQuantGroupSize), ubMte2NSize]大小，N方向按照128对齐写入
 * @param ubMte2NSize 从GM上搬运到UB的N方向大小
 * @param ubMte2KSize 从GM上搬运到UB的K方向大小
 * @param ubMte2NOffset 从GM上搬运到UB时, GM上N方向的偏移
 * @param ubMte2KOffset 从GM上搬运到UB时, GM上N方向的偏移
 * @param offsetParam 存储Weight矩阵的原始N,K,kAlign信息,用于搬运时GM上地址偏移的计算
 */
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyAntiQuantParamsGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam)
{
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_CHANNEL) {
        DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                   UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize],
                      antiQuantScaleGlobal_[ubMte2NOffset], 1, ubMte2NSize,
                      CeilAlign(ubMte2NSize, static_cast<uint64_t>(VECTOR_REG_WIDTH)), offsetParam.nSize);
        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            DataCopyPad2D(ubAntiQuantOffsetTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                        UB_BUFFER_INFO.antiQuantOffsetUbSingleBufferSize],
                          antiQuantOffsetGlobal_[ubMte2NOffset], 1, ubMte2NSize,
                          CeilAlign(ubMte2NSize, static_cast<uint64_t>(VECTOR_REG_WIDTH)), offsetParam.nSize);
        }
    } else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        scaleValue_ = antiQuantScaleGlobal_.GetValue(0);
        if constexpr (wqmmConfig.hasAntiQuantOffset) {
            offsetValue_ = antiQuantOffsetGlobal_.GetValue(0);
        }
    } else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
        DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                   UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize],
                      antiQuantScaleGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                      CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize, VEC_MAX_ELEM_B16, offsetParam.nSize);
    } else if constexpr (wqmmConfig.antiQuantType == QuantType::MX) {
        CopyMxAntiQuantParamsGmToUb(ubMte2NSize, ubMte2KSize, ubMte2NOffset, ubMte2KOffset, offsetParam);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyMxAntiQuantParamsGmToUb(
    uint64_t ubMte2NSize, uint64_t ubMte2KSize, uint64_t ubMte2NOffset, uint64_t ubMte2KOffset,
    const BasicBlockOffsetParam &offsetParam)
{
    uint64_t mxGroupNum = CeilDivide(offsetParam.kSize, MX_GROUPSIZE);
    LocalTensor<fp8_e8m0_t> ubAntiQuantScaleBuffer =
        ubAntiQuantScaleTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                     UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize]
            .template ReinterpretCast<fp8_e8m0_t>();
    if constexpr (wqmmConfig.bTrans) {
        DataCopyPad2D(ubAntiQuantScaleBuffer,
                      antiQuantScaleGlobal_[ubMte2NOffset * mxGroupNum + CeilDivide(ubMte2KOffset, MX_GROUPSIZE)],
                      ubMte2NSize, CeilDivide(ubMte2KSize, MX_GROUPSIZE),
                      CeilDivide(vecConfig.ubMte2InnerSize, MX_GROUPSIZE), mxGroupNum);
    } else {
        uint64_t scaleInnerSize = wqmmConfig.weightFormat != CubeFormat::NZ
                                      ? vecConfig.ubMte2InnerSize
                                      : 128UL;  // nz场景当前不会合并，固定对齐到128即可
        DataCopyPad2D(
            ubAntiQuantScaleBuffer,
            antiQuantScaleGlobal_[CeilDivide(ubMte2KOffset, MX_GROUPSIZE) * offsetParam.nSize + ubMte2NOffset],
            CeilDivide(ubMte2KSize, MX_GROUPSIZE), ubMte2NSize, scaleInnerSize, offsetParam.nSize);
    }

    event_t eventIdScaleMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdScaleMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdScaleMTE2ToV);
    MxScaleProcess(ubMte2NSize, ubMte2KSize);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::MxScaleProcess(uint64_t ubMte2NSize,
                                                                                      uint64_t ubMte2KSize)
{
    uint64_t ubMte2BufferIdx = ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum;
    MxFp4NdScaleParams<xType> mxFp4NdScaleParams;
    mxFp4NdScaleParams.antiQuantScaleBasePhyAddr =
        (__local_mem__ uint8_t *)
            ubAntiQuantScaleTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize]
                .GetPhyAddr();

    mxFp4NdScaleParams.antiQuantScaleF16PhyAddr0 =
        (__local_mem__ xType *)
            ubAntiQuantScaleAfterCastTotalBuffer_[ubMte2BufferIdx *
                                                  UB_BUFFER_INFO.antiQuantScaleAfterCastUbSingleBufferSize]
                .GetPhyAddr();

    mxFp4NdScaleParams.antiQuantScaleF16PhyAddr1 =
        mxFp4NdScaleParams.antiQuantScaleF16PhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    if constexpr (wqmmConfig.bTrans) {
        mxFp4NdScaleParams.ubLoopExternalAxis =
            CeilDivide(ubMte2NSize, static_cast<uint64_t>(4));  // 按照4行共128个数进行处理
    } else {
        if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
            // nd场景scale采取对齐到默认值策略
            mxFp4NdScaleParams.ubLoopExternalAxis = CeilDivide(ubMte2KSize, MX_GROUPSIZE);
        } else {
            // nz场景，scale采取内轴128 element对齐的策略，vf一次做256个数，此处直接除以2向上对齐
            mxFp4NdScaleParams.ubLoopExternalAxis = CeilDivide(CeilDivide(ubMte2KSize, MX_GROUPSIZE), 2UL);
        }
    }
    if constexpr (wqmmConfig.bTrans) {
        AscendC::VF_CALL<MxNkScaleVf<xType>>(mxFp4NdScaleParams);
    } else {
        AscendC::VF_CALL<MxKnScaleVf<xType>>(mxFp4NdScaleParams);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyMxBiasGmToUb(uint64_t ubMte2MxBiasNSize,
                                                                                        uint64_t ubMte2MxBiasNOffset)
{
    if (hasBias_) {
        DataCopyPad2D(
            ubBiasTotalBuffer_[(ubMte2LoopIdx_ % QUADRUPLE_BUFFER_NUM) * UB_BUFFER_INFO.biasUbSingleBufferSize],
            biasGlobal_[ubMte2MxBiasNOffset], 1, ubMte2MxBiasNSize, ubMte2MxBiasNSize, ubMte2MxBiasNSize);
        event_t eventIdBiasMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventIdBiasMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdBiasMTE2ToV);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyKcScaleBiasGmToUb(uint64_t nRealL0Size,
                                                                                             uint64_t mRealL0Size,
                                                                                             uint64_t nOffset,
                                                                                             uint64_t mOffset)
{
    if (ubMte2AntiquantYLoopIdx_ >= UB_ANTI_QUANT_Y_BUFFER_NUM) {
        WaitFlag<HardEvent::V_MTE2>(
            vecEventIdAntiQuantYVToMte2_[ubMte2AntiquantYLoopIdx_ & (UB_ANTI_QUANT_Y_BUFFER_NUM - 1)]);
    }
    // nRealL0Size和mRealL0Size为实际MTE2搬运到UB的有效数据，
    // 垃圾数据无需操作，搬出的时搬运有效数据即可。
    if (unlikely(mRealL0Size == 0)) {
        ubMte2AntiquantYLoopIdx_++; // 避免当前核无任务时，SetVToMTE2()对同一个flagID重复SetFlag的问题
        return;
    }
    // nRealL0Size 需要 对齐 32B / sizeof(T)   32B是UB最小对齐单元
    uint64_t nCopyLenAlign = CeilAlign(nRealL0Size, static_cast<uint64_t>(FP32_BLOCK_SIZE));

    DataCopyPad2D(ubAntiQuantYPerChannelScaleTotalBuffer_[(ubMte2AntiquantYLoopIdx_ % UB_ANTI_QUANT_Y_BUFFER_NUM) *
                                                          ANTI_QUANT_Y_PER_CHANNEL_SCALE_SINGLE_BUFFER_SIZE],
                  antiQuantYPerChannelScaleGlobal_[nOffset], 1, nRealL0Size, nCopyLenAlign, nRealL0Size);

    uint64_t mCopyLenAlign = CeilAlign(mRealL0Size, static_cast<uint64_t>(FP32_BLOCK_SIZE));
    DataCopyPad2D(ubAntiQuantYPerTokenScaleTotalBuffer_[(ubMte2AntiquantYLoopIdx_ % UB_ANTI_QUANT_Y_BUFFER_NUM) *
                                                        ANTI_QUANT_Y_PER_TOKEN_SCALE_SINGLE_BUFFER_SIZE],
                  antiQuantYPerTokenScaleGlobal_[mOffset], 1, mRealL0Size, mCopyLenAlign, mRealL0Size);

    if (hasBias_) {
        DataCopyPad2D(ubAntiQuantYBiasTotalBuffer_[(ubMte2AntiquantYLoopIdx_ % UB_ANTI_QUANT_Y_BUFFER_NUM) *
                                                   ANTI_QUANT_Y_BIAS_SINGLE_BUFFER_SIZE],
                      antiQuantYBiasGlobal_[nOffset], 1, nRealL0Size, nCopyLenAlign, nRealL0Size);
    }
    event_t eventIdAntiquantYMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdAntiquantYMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdAntiquantYMTE2ToV);
    ubMte2AntiquantYLoopIdx_++;
}

/**
* @brief 该函数作用为对于搬运到UB的weight,scale, offst,按照标准VF计算单元(64,256)进行多次循环计算,总计算量为L1所需的大小
         每一个VF计算单元的结果放置于ubHighBitTotalBuffer_中，其计算和MTE3搬运使用vecEventIdMte3ToV_控制同步
* @param ubConsumeConfig 其中l1RequireVfComputeRealN,l1RequireVfComputeRealK表示L1上需要VEC计算的实际数据量
                         nWeightLowBitUbOffset, kWeightLowBitUbOffset表示在搬运到UB上的weight上的偏移
* @param weightHighBitL1 L1上的weight地址，用于反量化结束后存放
* @param l1ConsumeConfig 其中l1RealExternalLen为L1上真实外轴长度, SplitTwoVecExternalOffset为L1上切分给两个VEC核的外轴偏
                         移大小，用于搬运到L1的dst地址偏移计算。
*/
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightAntiQuantCompute(
    const UbConsumeConfig &ubConsumeConfig, const LocalTensor<xType> &weightHighBitL1,
    const L1ConsumeConfig &l1ConsumeConfig, const LocalTensor<biasType> *biasL1Ptr)
{
    uint64_t weightHighBitL1Offset;
    uint64_t nRealLen;
    uint64_t kRealLen;
    TEventID vecEventIdMte3ToV[QUADRUPLE_BUFFER_NUM];

    // 用临时变量接一下，优化编译的作用
    vecEventIdMte3ToV[0] = vecEventIdMte3ToV_[0];
    vecEventIdMte3ToV[1] = vecEventIdMte3ToV_[1];
    if constexpr (wqmmConfig.weightFormat == CubeFormat::NZ) {
        vecEventIdMte3ToV[2] = vecEventIdMte3ToV_[2];
        vecEventIdMte3ToV[3] = vecEventIdMte3ToV_[3];
    }
    for (uint64_t antiQuantKOffset = 0; antiQuantKOffset < ubConsumeConfig.l1RequireVfComputeRealK;
         antiQuantKOffset += VF_CONFIG.vfKStandardLen) {
        for (uint64_t antiQuantNOffset = 0; antiQuantNOffset < ubConsumeConfig.l1RequireVfComputeRealN;
             antiQuantNOffset += VF_CONFIG.vfNStandardLen) {
            if (likely(ubComputeLoopIdx_ > UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) {
                WaitFlag<HardEvent::MTE3_V>(
                    vecEventIdMte3ToV[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
            }
            nRealLen = antiQuantNOffset + VF_CONFIG.vfNStandardLen >= ubConsumeConfig.l1RequireVfComputeRealN
                           ? ubConsumeConfig.l1RequireVfComputeRealN - antiQuantNOffset
                           : VF_CONFIG.vfNStandardLen;
            kRealLen = antiQuantKOffset + VF_CONFIG.vfKStandardLen >= ubConsumeConfig.l1RequireVfComputeRealK
                           ? ubConsumeConfig.l1RequireVfComputeRealK - antiQuantKOffset
                           : VF_CONFIG.vfKStandardLen;
            WeightAntiQuantProcess(nRealLen, kRealLen, antiQuantNOffset, antiQuantKOffset, ubConsumeConfig);

            weightHighBitL1Offset =
                ComputeWeightHighBitL1Offset(antiQuantNOffset, antiQuantKOffset, nRealLen, kRealLen, l1ConsumeConfig);
            event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WeightHighBitUbToL1(weightHighBitL1Offset, nRealLen, kRealLen, weightHighBitL1,
                                l1ConsumeConfig.l1RealExternalLen);
            if (biasL1Ptr != nullptr && ubConsumeConfig.calcMxBias) {
                auto biasL1 = *biasL1Ptr;
                DataCopy(
                    biasL1[l1ConsumeConfig.l1MxBiasSplitNOffset],
                    ubBiasOutTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                          UB_BUFFER_INFO.biasReducedSingleBufferSize],
                    ubConsumeConfig.ubMxBiasNsize);
            }
            SetFlag<HardEvent::MTE3_V>(
                vecEventIdMte3ToV[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
            ubComputeLoopIdx_++;
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightMxScaleGmToUb(
    uint64_t nOffset, uint64_t nL1Size, uint64_t kOffset, uint64_t mxGroupNum, uint64_t realGroupNum)
{
    if (nL1Size == 0) {
        return;
    }
    DataCopyPad2D(ubWeightInputLowBitTotalBuffer_[(ubMte2LoopIdx_ % vecConfig.ubMte2BufferNum) *
                                                  UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                      .template ReinterpretCast<antiQuantScaleType>(),
                  antiQuantScaleGlobal_[nOffset * mxGroupNum + CeilDivide<uint64_t>(kOffset, MX_GROUPSIZE)], nL1Size,
                  realGroupNum, MX_SCALE_GROUP_NUM_DEFAULT_LEN + MX_SCALE_BANK_CONFLICT_OFFSET, mxGroupNum);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    ubMte2LoopIdx_++;
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::TransWeightMxScale(uint64_t nSize,
                                                                                          uint64_t kSize)
{
    WaitFlag<HardEvent::MTE3_V>(vecEventIdTransScaleMte3ToV_[ubTransScaleLoopIdx_ & (DOUBLE_BUFFER_NUM - 1)]);
    AscendC::VF_CALL<TransMxScaleNdToDnVf<((MX_SCALE_GROUP_NUM_DEFAULT_LEN + MX_SCALE_BANK_CONFLICT_OFFSET) >> 1),
                                          VEC_MAX_ELEM_B16>>(
        (__ubuf__ uint16_t *)ubWeightInputLowBitTotalBuffer_.GetPhyAddr(
            ((ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1)) *
            UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize),
        (__ubuf__ uint16_t *)transScaleBuffer_.GetPhyAddr((ubTransScaleLoopIdx_ & (DOUBLE_BUFFER_NUM - 1)) *
                                                          TRANS_SCALE_SINGLE_BUF_SIZE),
        (__ubuf__ uint16_t *)transIdBuffer_.GetPhyAddr(), CeilDivide<uint64_t>(nSize, BLOCK_CUBE),
        CeilDivide<uint64_t>(CeilDivide<uint64_t>(kSize, 2 * MX_GROUPSIZE), 8),
        (MX_SCALE_GROUP_NUM_DEFAULT_LEN * BLOCK_CUBE) >> 1);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightMxScaleUbToL1(
    LocalTensor<uint16_t> &scaleL1, uint64_t l1NOffset, uint64_t nSize)
{
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopyParams params;
    params.blockCount = CeilDivide<uint64_t>(nSize, BLOCK_CUBE) * MX_SCALE_GROUP_NUM_DEFAULT_LEN / 16;
    params.blockLen = VEC_MAX_ELEM_B16 / BLOCK_CUBE;
    params.srcStride = 0;
    params.dstStride = 0;

    DataCopy(scaleL1[l1NOffset * MX_SCALE_GROUP_NUM_DEFAULT_LEN / 2],
             transScaleBuffer_[(ubTransScaleLoopIdx_ & (DOUBLE_BUFFER_NUM - 1)) * TRANS_SCALE_SINGLE_BUF_SIZE]
                 .template ReinterpretCast<uint16_t>(),
             params);

    SetFlag<HardEvent::MTE3_V>(vecEventIdTransScaleMte3ToV_[ubTransScaleLoopIdx_ & (DOUBLE_BUFFER_NUM - 1)]);
    ubTransScaleLoopIdx_++;
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightAntiQuantComputeNzNk(
    const UbConsumeConfig &ubConsumeConfig, const LocalTensor<xType> &weightHighBitL1,
    const L1ConsumeConfig &l1ConsumeConfig, const LocalTensor<biasType> &biasL1)
{
    TEventID vecEventIdMte3ToV[QUADRUPLE_BUFFER_NUM];

    // 用临时变量接一下，优化编译的作用
    vecEventIdMte3ToV[0] = vecEventIdMte3ToV_[0];
    vecEventIdMte3ToV[1] = vecEventIdMte3ToV_[1];
    if constexpr (wqmmConfig.weightFormat == CubeFormat::NZ) {
        vecEventIdMte3ToV[2] = vecEventIdMte3ToV_[2];
        vecEventIdMte3ToV[3] = vecEventIdMte3ToV_[3];
    }

    if (likely(ubComputeLoopIdx_ > UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) {
        WaitFlag<HardEvent::MTE3_V>(
            vecEventIdMte3ToV[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    }

    AntiQuantProcessNzMxA8W4(ubConsumeConfig);

    uint64_t weightHighBitL1Offset = ComputeWeightHighBitL1Offset(
        0, 0, ubConsumeConfig.l1RequireVfComputeRealN, ubConsumeConfig.l1RequireVfComputeRealK, l1ConsumeConfig);

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    if (likely(ubConsumeConfig.l1RequireVfComputeRealK > 0)) {
        CopyWeightHighBitForAligned(weightHighBitL1Offset, ubConsumeConfig.l1RequireVfComputeRealN,
                                    ubConsumeConfig.l1RequireVfComputeRealK, weightHighBitL1);
    }
    if (ubConsumeConfig.calcMxBias) {
        DataCopy(biasL1[l1ConsumeConfig.l1MxBiasSplitNOffset],
                 ubBiasOutTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                       UB_BUFFER_INFO.biasReducedSingleBufferSize],
                 ubConsumeConfig.ubMxBiasNsize);
    }
    SetFlag<HardEvent::MTE3_V>(
        vecEventIdMte3ToV[ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)]);
    ubComputeLoopIdx_++;
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::ComputeWeightHighBitL1Offset(
    uint64_t antiQuantNOffset, uint64_t antiQuantKOffset, uint64_t nRealLen, uint64_t kRealLen,
    const L1ConsumeConfig &l1ConsumeConfig)
{
    if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
        uint64_t l1RealExternalLenAlign =
            CeilAlign(l1ConsumeConfig.l1RealExternalLen, static_cast<uint64_t>(BLOCK_CUBE));
        if constexpr (!wqmmConfig.bTrans) {
            return l1RealExternalLenAlign * antiQuantNOffset +
                   (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * static_cast<uint64_t>(BLOCK_CUBE);
        } else {
            return l1RealExternalLenAlign * antiQuantKOffset +
                   (antiQuantNOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) * static_cast<uint64_t>(BLOCK_CUBE);
        }
    } else {
        if constexpr (!wqmmConfig.bTrans) {
            uint64_t kRealLenAlign = CeilAlign(kRealLen, static_cast<uint64_t>(BLOCK_CUBE));
            return kRealLenAlign * (antiQuantNOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) +
                   antiQuantKOffset * static_cast<uint64_t>(C0_SIZE);
        } else {
            uint64_t nRealLenAlign = CeilAlign(nRealLen, static_cast<uint64_t>(BLOCK_CUBE));
            return nRealLenAlign * (antiQuantKOffset + l1ConsumeConfig.l1SplitTwoVecExternalOffset) +
                   antiQuantNOffset * static_cast<uint64_t>(C0_SIZE);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightAntiQuantProcess(
    uint64_t nRealLen, uint64_t kRealLen, uint64_t antiQuantNOffset, uint64_t antiQuantKOffset,
    const UbConsumeConfig &ubConsumeConfig)
{
     if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) {
        AntiQuantProcess(kRealLen, nRealLen, ubConsumeConfig.nWeightLowBitUbOffset + antiQuantNOffset,
                         ubConsumeConfig.kWeightLowBitUbOffset + antiQuantKOffset, ubConsumeConfig);
    } else if constexpr ((wqmmConfig.weightFormat != CubeFormat::NZ && wqmmConfig.bTrans) ||
                         (wqmmConfig.weightFormat == CubeFormat::NZ && !wqmmConfig.bTrans)) {
        AntiQuantProcess(nRealLen, kRealLen, ubConsumeConfig.nWeightLowBitUbOffset + antiQuantNOffset,
                         ubConsumeConfig.kWeightLowBitUbOffset + antiQuantKOffset, ubConsumeConfig);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcess(
    uint64_t vfExternalRealLen, uint64_t vfInnerRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
    const UbConsumeConfig &ubConsumeConfig)
{
    if constexpr (wqmmConfig.antiQuantType == QuantType::MX) {
        if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
            AntiQuantProcessNdMx(vfExternalRealLen, vfInnerRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        } else {
            AntiQuantProcessNzMx(vfExternalRealLen, vfInnerRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        }
    } else {
        if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
            AntiQuantProcessNd(vfExternalRealLen, vfInnerRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        } else {
            AntiQuantProcessNz(vfExternalRealLen, vfInnerRealLen, nWeightLowBitUbOffset, kWeightLowBitUbOffset);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNdMx(
    uint64_t vfExternalRealLen, uint64_t vfInnerRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset)
{
    uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
    uint64_t weightF16BufIdx = ubComputeLoopIdx_ & 1;
    uint64_t weightLowBitBufOffset = mte2BufIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize;
    uint64_t antiQuantScaleAfterCastBufOffset = mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleAfterCastUbSingleBufferSize;
    uint64_t weightLowBitOffset;
    uint64_t antiQuantScaleAfterCastOffset;

    if constexpr (wqmmConfig.bTrans) {
        weightLowBitOffset = nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset;
        antiQuantScaleAfterCastOffset = (nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize / MX_GROUPSIZE +
                                         CeilDivide(kWeightLowBitUbOffset, MX_GROUPSIZE)) *
                                        2;  // 在经过e8m0-->f16后，偏移需要乘2
    } else {
        weightLowBitOffset = kWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset;
        antiQuantScaleAfterCastOffset =
            CeilDivide(kWeightLowBitUbOffset, MX_GROUPSIZE) * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset;
    }

    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        weightLowBitOffset = weightLowBitOffset >> 1;
    }

    MxFp4NdWeightParams<xType, wType> mxFp4NdWeightParams;
    mxFp4NdWeightParams.antiQuantScaleF16PhyAddr0 =
        (__local_mem__ xType *)ubAntiQuantScaleAfterCastTotalBuffer_.GetPhyAddr(antiQuantScaleAfterCastBufOffset) +
        antiQuantScaleAfterCastOffset;
    mxFp4NdWeightParams.weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_.GetPhyAddr(weightLowBitBufOffset) + weightLowBitOffset;
    mxFp4NdWeightParams.weightLowBitPhyAddr1 = mxFp4NdWeightParams.weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    mxFp4NdWeightParams.weightF16PhyAddr0 = (__local_mem__ xType *)ubHighBitTotalBuffer_.GetPhyAddr(
        weightF16BufIdx * UB_BUFFER_INFO.highBitDataUbSingleBufferSize);
    mxFp4NdWeightParams.weightF16PhyAddr1 =
        mxFp4NdWeightParams.weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    if constexpr (wqmmConfig.bTrans) {
        mxFp4NdWeightParams.ubLoopExternalAxis = vfExternalRealLen;
        // 8个scale对应8/2 * 32=128个数
        mxFp4NdWeightParams.antiQuantScaleF16PhyAddr1 = mxFp4NdWeightParams.antiQuantScaleF16PhyAddr0 + 8;
        AscendC::VF_CALL<MxNkWeightVf<xType, wType, vecConfig>>(mxFp4NdWeightParams);

    } else {
        mxFp4NdWeightParams.ubLoopExternalAxis = CeilDivide(vfExternalRealLen, MX_GROUPSIZE);
        mxFp4NdWeightParams.antiQuantScaleF16PhyAddr1 =
            mxFp4NdWeightParams.antiQuantScaleF16PhyAddr0 + (VECTOR_REG_WIDTH >> 1);
        AscendC::VF_CALL<MxKnWeightVf<xType, wType, vecConfig>>(mxFp4NdWeightParams);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNzMx(
    uint64_t vfExternalRealLen, uint64_t vfInnerRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset)
{
    if constexpr (!wqmmConfig.bTrans) {
        Fp4NzParams<xType, wType> fp4NzParams;
        uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);

        uint64_t antiQuantScaleAfterCastBufOffset =
            ubMte2BufferIdx * UB_BUFFER_INFO.antiQuantScaleAfterCastUbSingleBufferSize + nWeightLowBitUbOffset;
        fp4NzParams.antiQuantScaleBasePhyAddr =
            (__local_mem__ xType *)ubAntiQuantScaleAfterCastTotalBuffer_[antiQuantScaleAfterCastBufOffset].GetPhyAddr();
        fp4NzParams.weightLowBitPhyAddr =
            (__local_mem__ wType *)
                ubWeightInputLowBitTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                    .GetPhyAddr() +
            ((nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset * C0_SIZE) >> 1);

        fp4NzParams.weightHighBitPhyAddr =
            (__local_mem__ xType *)
                ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                      VEC_MAX_ELEM_B16]
                    .GetPhyAddr();

        fp4NzParams.loopN1 = CeilDivide(vfExternalRealLen, static_cast<uint64_t>(C0_SIZE));
        // 跳写UB避免bank冲突
        fp4NzParams.innerDstStride = VEC_MAX_ELEM_B16 * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;

        fp4NzParams.loopGroupNum = CeilDivide(vfInnerRealLen, MX_GROUPSIZE);
        // 小数据量vfInnerRealLen对齐到BLOCK_CUBE，再计算循环次数
        fp4NzParams.loopInnerNum =
            vfInnerRealLen < MX_GROUPSIZE
                ? CeilDivide(CeilAlign(vfInnerRealLen, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE, VEC_MAX_ELEM_B16)
                : CeilDivide(MX_GROUPSIZE * C0_SIZE, VEC_MAX_ELEM_B16);
        fp4NzParams.groupDstStride = fp4NzParams.loopInnerNum * fp4NzParams.innerDstStride;
        fp4NzParams.loopN1DstStride = fp4NzParams.loopGroupNum * fp4NzParams.groupDstStride;
        AscendC::VF_CALL<AntiQuantFp4NzKnVf<xType, wType, vecConfig.ubMte2InnerSize>>(fp4NzParams);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNzMxA8W4(
    const UbConsumeConfig &ubConsumeConfig)
{
    MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams;
    uint64_t ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
    mxA8W4NzParams.nRealSizeAlign =
        CeilAlign(ubConsumeConfig.l1RequireVfComputeRealN, static_cast<uint64_t>(BLOCK_CUBE));
    mxA8W4NzParams.weightLowBitPhyAddr =
        (__ubuf__ wType *)
            ubWeightInputLowBitTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                .GetPhyAddr();

    mxA8W4NzParams.weightHighBitPhyAddr =
        (__ubuf__ xType *)
            ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                  VECTOR_REG_WIDTH]
                .GetPhyAddr();
    mxA8W4NzParams.loopKNum = CeilDivide(ubConsumeConfig.l1RequireVfComputeRealK, static_cast<uint64_t>(C0_SIZE));
    mxA8W4NzParams.innerLoopNum =
        CeilDivide(mxA8W4NzParams.nRealSizeAlign * C0_SIZE, static_cast<uint64_t>(VECTOR_REG_WIDTH));
    // 跳写UB避免bank冲突
    mxA8W4NzParams.innerDstStride = VECTOR_REG_WIDTH * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;
    mxA8W4NzParams.loopKDstStride = mxA8W4NzParams.innerLoopNum * mxA8W4NzParams.innerDstStride;
    if (ubConsumeConfig.calcMxBias) {
        mxA8W4NzParams.biasInUbAddr =
            (__ubuf__ biasType *)ubBiasTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.biasUbSingleBufferSize]
                .GetPhyAddr();
        mxA8W4NzParams.biasOutUbAddr =
            (__ubuf__ biasType *)
                ubBiasOutTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                      UB_BUFFER_INFO.biasReducedSingleBufferSize]
                    .GetPhyAddr();
        if (ubConsumeConfig.isBiasSingleVector) {
            mxA8W4NzParams.biasLoopNum = CeilDivide(ubConsumeConfig.ubMxBiasNsize, VEC_MAX_ELEM_B16);
            AntiQuantMxA8W4NzNkVf<xType, wType, biasType, true, true>(mxA8W4NzParams);
        } else {
            AntiQuantMxA8W4NzNkVf<xType, wType, biasType, true, false>(mxA8W4NzParams);
        }
    } else {
        AntiQuantMxA8W4NzNkVf<xType, wType, biasType, false, false>(mxA8W4NzParams);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNd(
    uint64_t vfExternalRealLen, uint64_t vfInnerRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset)
{
    LocalAddressParam<xType, wType> addressParam;
    CalLocalAddrForVf(nWeightLowBitUbOffset, kWeightLowBitUbOffset, addressParam);
    if constexpr (IsSameType<wType, int8_t>::value || IsSameType<wType, hifloat8_t>::value) {
        if constexpr (wqmmConfig.bTrans) {
            AntiQuantB8CommonNdNk<xType, wType, wqmmConfig, vecConfig>(
                addressParam.antiQuantScaleBasePhyAddr, addressParam.antiQuantOffsetBasePhyAddr,
                addressParam.weightLowBitPhyAddr0, addressParam.weightF16PhyAddr0, scaleValue_, offsetValue_,
                vfExternalRealLen);
        } else {
            AntiQuantB8CommonNdKn<xType, wType, wqmmConfig, vecConfig>(
                addressParam.antiQuantScaleBasePhyAddr, addressParam.antiQuantOffsetBasePhyAddr,
                addressParam.weightLowBitPhyAddr0, addressParam.weightF16PhyAddr0, scaleValue_, offsetValue_,
                vfExternalRealLen);
        }
    } else if constexpr (IsSameType<wType, fp8_e5m2_t>::value || IsSameType<wType, fp8_e4m3fn_t>::value) {
        CalculateParam<xType> calculateParam;
        calculateParam.offsetValue = offsetValue_;
        calculateParam.scaleValue = scaleValue_;
        calculateParam.ubLoop = vfExternalRealLen;
        if constexpr (wqmmConfig.bTrans) {
            AscendC::VF_CALL<AntiQuantFP8NdNkVf<xType, wType, wqmmConfig.hasAntiQuantOffset, vecConfig.ubMte2InnerSize,
                                                wqmmConfig.antiQuantType>>(addressParam, calculateParam);
        } else {
            AscendC::VF_CALL<AntiQuantFP8NdKnVf<xType, wType, wqmmConfig.hasAntiQuantOffset, vecConfig.ubMte2InnerSize,
                                                wqmmConfig.antiQuantType>>(addressParam, calculateParam);
        }
    } else {
        if constexpr (wqmmConfig.bTrans) {
            AntiQuantInt4NdNk<xType, wType, wqmmConfig, vecConfig>(
                addressParam.antiQuantScaleBasePhyAddr, addressParam.antiQuantOffsetBasePhyAddr,
                addressParam.weightLowBitPhyAddr0, addressParam.weightF16PhyAddr0, scaleValue_, offsetValue_,
                vfExternalRealLen);
        } else {
            AntiQuantInt4NdKn<xType, wType, wqmmConfig, vecConfig>(
                addressParam.antiQuantScaleBasePhyAddr, addressParam.antiQuantOffsetBasePhyAddr,
                addressParam.weightLowBitPhyAddr0, addressParam.weightF16PhyAddr0, scaleValue_, offsetValue_,
                vfExternalRealLen);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantProcessNz(
    uint64_t vfExternalRealLen, uint64_t vfInnerRealLen, uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset)
{
    if constexpr (!wqmmConfig.bTrans) {
        Int4NzParams<xType, wType, antiQuantScaleType> int4NzParams;
        CalInt4NzKnLocalAddrForVf(nWeightLowBitUbOffset, kWeightLowBitUbOffset, int4NzParams);
        int4NzParams.loopN1 = CeilDivide(vfExternalRealLen, static_cast<uint64_t>(C0_SIZE));
        // 跳写UB避免bank冲突，A16跳1024B，A8跳512B; MTE3对应跳读
        int4NzParams.innerDstStride = VEC_MAX_ELEM_B16 * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;

        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
            int4NzParams.antiQuantGroupSize = antiQuantGroupSize_;
            int4NzParams.loopGroupNum = CeilDivide(vfInnerRealLen, antiQuantGroupSize_);
            int4NzParams.loopInnerNum =
                vfInnerRealLen < antiQuantGroupSize_ ?
                    CeilDivide(CeilAlign(vfInnerRealLen, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE,
                               VEC_MAX_ELEM_B16) :
                    CeilDivide(antiQuantGroupSize_ * C0_SIZE, VEC_MAX_ELEM_B16);
            int4NzParams.groupDstStride = int4NzParams.loopInnerNum * int4NzParams.innerDstStride;
            int4NzParams.loopN1DstStride = int4NzParams.loopGroupNum * int4NzParams.groupDstStride;
            AscendC::VF_CALL<AntiQuantS8S4NzKnGroupVf<xType, wType, antiQuantScaleType, wqmmConfig.hasAntiQuantOffset,
                                                      vecConfig.ubMte2InnerSize>>(int4NzParams);
        } else {
            int4NzParams.loopInnerNum = CeilDivide(
                static_cast<uint64_t>(CeilAlign(vfInnerRealLen, static_cast<uint64_t>(BLOCK_CUBE))) * C0_SIZE,
                VEC_MAX_ELEM_B16);
            int4NzParams.loopN1DstStride = int4NzParams.loopInnerNum * int4NzParams.innerDstStride;

            if (int4NzParams.loopN1 == 1) {
                AscendC::VF_CALL<AntiQuantInt4NzKnVf<xType, wType, antiQuantScaleType, wqmmConfig.hasAntiQuantOffset,
                                                     vecConfig.ubMte2InnerSize, false>>(int4NzParams);
            } else {
                AscendC::VF_CALL<AntiQuantInt4NzKnVf<xType, wType, antiQuantScaleType, wqmmConfig.hasAntiQuantOffset,
                                                     vecConfig.ubMte2InnerSize, true>>(int4NzParams);
            }
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CalInt4NzKnLocalAddrForVf(
    uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset,
    Int4NzParams<xType, wType, antiQuantScaleType> &int4NzParams)
{
    uint64_t ubMte2BufferIdx;
    if constexpr (IsSameType<xType, int8_t>::value) {
        ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) % vecConfig.ubMte2BufferNum;
        int4NzParams.antiQuantScaleBasePhyAddr =
            (__local_mem__ antiQuantScaleType *)
                ubAntiQuantScaleTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
                                             kWeightLowBitUbOffset / antiQuantGroupSize_ * VEC_MAX_ELEM_B16 +
                                             nWeightLowBitUbOffset]
                    .GetPhyAddr();
        int4NzParams.antiQuantScaleMaskPhyAddr = (__local_mem__ uint8_t *)ubAntiQuantScaleMaskBuffer_.GetPhyAddr();
    } else {
        ubMte2BufferIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
        int4NzParams.antiQuantScaleBasePhyAddr =
            (__local_mem__ antiQuantScaleType *)
                ubAntiQuantScaleTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
                                             nWeightLowBitUbOffset]
                    .GetPhyAddr();
        int4NzParams.antiQuantOffsetBasePhyAddr =
            (__local_mem__ xType *)
                ubAntiQuantOffsetTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
                                              nWeightLowBitUbOffset]
                    .GetPhyAddr();
    }

    int4NzParams.weightLowBitPhyAddr =
        (__local_mem__ wType *)
            ubWeightInputLowBitTotalBuffer_[ubMte2BufferIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize]
                .GetPhyAddr() +
        ((nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset * C0_SIZE) >> 1);

    int4NzParams.weightHighBitPhyAddr =
        (__local_mem__ xType *)
            ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                  VEC_MAX_ELEM_B16]
                .GetPhyAddr();
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CalLocalAddrForVf(
    uint64_t nWeightLowBitUbOffset, uint64_t kWeightLowBitUbOffset, LocalAddressParam<xType, wType> &localAddressParam)
{
    uint64_t mte2BufIdx = (ubMte2LoopIdx_ - 1) & (vecConfig.ubMte2BufferNum - 1);
    uint64_t antiquantParamOffset =
        mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
    localAddressParam.antiQuantScaleBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantScaleTotalBuffer_.GetPhyAddr(antiquantParamOffset);
    localAddressParam.antiQuantScaleBasePhyAddr1 = localAddressParam.antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    localAddressParam.antiQuantOffsetBasePhyAddr =
        (__local_mem__ xType *)ubAntiQuantOffsetTotalBuffer_.GetPhyAddr(antiquantParamOffset);
    localAddressParam.antiQuantOffsetBasePhyAddr1 = localAddressParam.antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;

    uint64_t weightLowBitOffset;
    if constexpr (wqmmConfig.bTrans) {
        weightLowBitOffset = nWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + kWeightLowBitUbOffset;
    } else {
        weightLowBitOffset = kWeightLowBitUbOffset * vecConfig.ubMte2InnerSize + nWeightLowBitUbOffset;
    }

    if constexpr (IsSameType<wType, int4b_t>::value) {
        weightLowBitOffset = weightLowBitOffset >> 1;
    }

    uint64_t weightLowBitBufOffset = mte2BufIdx * UB_BUFFER_INFO.weightInputLowBitUbSingleBufferSize;
    localAddressParam.weightLowBitPhyAddr0 =
        (__local_mem__ wType *)ubWeightInputLowBitTotalBuffer_.GetPhyAddr(weightLowBitBufOffset) + weightLowBitOffset;
    if constexpr (IsSameType<wType, int4b_t>::value) {
        // int4每次处理128个数即为64B, 256>>2=64
        localAddressParam.weightLowBitPhyAddr1 = localAddressParam.weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    } else {
        localAddressParam.weightLowBitPhyAddr1 = localAddressParam.weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    }

    uint64_t weightF16BufIdx = ubComputeLoopIdx_ & 1;
    localAddressParam.weightF16PhyAddr0 = (__local_mem__ xType *)ubHighBitTotalBuffer_.GetPhyAddr(
        weightF16BufIdx * UB_BUFFER_INFO.highBitDataUbSingleBufferSize);
    localAddressParam.weightF16PhyAddr1 =
        localAddressParam.weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::WeightHighBitUbToL1(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1, uint64_t l1RealExternalLen)
{
    DataCopyParams params;
    if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
        if constexpr (wqmmConfig.bTrans) {
            params.blockLen = antiQuantRealN;
            params.blockCount = CeilDivide(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE));
            params.srcStride = WEIGHT_F16_UB_NZ_STRIDE - antiQuantRealN;
            params.dstStride = CeilAlign(l1RealExternalLen, static_cast<uint64_t>(BLOCK_CUBE)) - antiQuantRealN;
        } else {
            params.blockLen = antiQuantRealK;
            params.blockCount = CeilDivide(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE));
            params.srcStride = WEIGHT_F16_UB_NZ_STRIDE - antiQuantRealK;
            params.dstStride = CeilAlign(l1RealExternalLen, static_cast<uint64_t>(BLOCK_CUBE)) - antiQuantRealK;
        }
        DataCopy(weightHighBitL1[weightHighBitL1Offset],
                 ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & 1) * UB_BUFFER_INFO.highBitDataUbSingleBufferSize], params);
    } else {
        if constexpr (wqmmConfig.antiQuantType == QuantType::MX && !IsSameType<xType, fp8_e4m3fn_t>::value) {
            // 小数据量vfInnerRealLen或K与BLOCK_CUBE, MX_GROUPSIZE上对齐大小相同时, mte3对齐到BLOCK_CUBE
            if (antiQuantRealK < MX_GROUPSIZE || CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE)) ==
                                                     CeilAlign(antiQuantRealK, static_cast<uint64_t>(MX_GROUPSIZE))) {
                CopyWeightHighBitForAligned(weightHighBitL1Offset, antiQuantRealN, antiQuantRealK, weightHighBitL1);
            } else {
                // mte3对齐到MX_GROUPSIZE
                CopyWeightHighBitForUnaligned(weightHighBitL1Offset, antiQuantRealN, antiQuantRealK, weightHighBitL1);
            }
        } else {
            // 适用场景: s8s4 nz kn
            CopyWeightHighBitForAligned(weightHighBitL1Offset, antiQuantRealN, antiQuantRealK, weightHighBitL1);
        }
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightHighBitForAligned(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    if constexpr (!wqmmConfig.bTrans) {
        params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE)) *
                            CeilAlign(antiQuantRealN, static_cast<uint64_t>(C0_SIZE)) / VEC_REG_ELEM;
    } else {
        params.blockCount = CeilAlign(antiQuantRealK, static_cast<uint64_t>(C0_SIZE)) *
                            CeilAlign(antiQuantRealN, static_cast<uint64_t>(BLOCK_CUBE)) / VEC_REG_ELEM;
    }

    params.blockLen = (IsSameType<xType, int8_t>::value || IsSameType<xType, fp8_e4m3fn_t>::value)
                          ? VEC_REG_ELEM / ONE_BLK_SIZE
                          : VEC_REG_ELEM / BLOCK_CUBE;
    params.srcStride = (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1) * params.blockLen;
    params.dstStride = 0;  // dst地址连续
    DataCopy(
        weightHighBitL1[weightHighBitL1Offset],
        ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) * VEC_REG_ELEM],
        params);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyWeightHighBitForUnaligned(
    uint64_t weightHighBitL1Offset, uint64_t antiQuantRealN, uint64_t antiQuantRealK,
    const LocalTensor<xType> &weightHighBitL1)
{
    DataCopyParams params;
    // 跳写UB避免bank冲突，A16跳1024B; MTE3对应跳读
    uint64_t innerDstStride = VEC_MAX_ELEM_B16 * UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum;
    uint64_t blockCubeAlignOfk = CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t mxGroupSizeAlignOfk = CeilAlign(antiQuantRealK, static_cast<uint64_t>(MX_GROUPSIZE));
    for (int i = 0; i < CeilDivide(antiQuantRealN, static_cast<uint64_t>(C0_SIZE)); i++) {
        params.blockCount =
            CeilAlign(antiQuantRealK, static_cast<uint64_t>(BLOCK_CUBE)) / (VEC_MAX_ELEM_B16 / BLOCK_CUBE);
        params.blockLen = VEC_MAX_ELEM_B16 / BLOCK_CUBE;
        params.srcStride = (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1) * params.blockLen;
        params.dstStride = 0; // dst地址连续
        weightHighBitL1Offset += i * blockCubeAlignOfk * BLOCK_CUBE;
        // dst需要再加i * 32对齐后的blockCount * innerDstStride作为地址偏置
        DataCopy(weightHighBitL1[weightHighBitL1Offset],
                 ubHighBitTotalBuffer_[(ubComputeLoopIdx_ & (UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum - 1)) *
                                           VEC_MAX_ELEM_B16 +
                                       i * (mxGroupSizeAlignOfk / (VEC_MAX_ELEM_B16 / BLOCK_CUBE) * innerDstStride)],
                 params);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::AntiQuantYWithKc(uint64_t nRealL0Size,
                                                                                        uint64_t mRealL0Size)
{
    if (unlikely(mRealL0Size == 0)) {
        SetFlag<HardEvent::V_MTE2>(
            vecEventIdAntiQuantYVToMte2_[(ubMte2AntiquantYLoopIdx_ - 1) & (UB_ANTI_QUANT_Y_BUFFER_NUM - 1)]);
        return;
    }

    LocalAddressYParam<yType> addressParam;
    CalLocalAddrForYVf(addressParam);
    uint16_t loopN = CeilDivide(nRealL0Size, ANTIQUANT_Y_STANDARD_N_SIZE);

    if (hasBias_) {
        AntiQuantYB32<yType, true>(addressParam, CeilAlign(nRealL0Size, ANTIQUANT_Y_STANDARD_N_SIZE),
                                   ANTIQUANT_Y_STANDARD_N_SIZE, loopN, (uint16_t)mRealL0Size);
    } else {
        AntiQuantYB32<yType, false>(addressParam, CeilAlign(nRealL0Size, ANTIQUANT_Y_STANDARD_N_SIZE),
                                    ANTIQUANT_Y_STANDARD_N_SIZE, loopN, (uint16_t)mRealL0Size);
    }

    SetFlag<HardEvent::V_MTE2>(
        vecEventIdAntiQuantYVToMte2_[(ubMte2AntiquantYLoopIdx_ - 1) & (UB_ANTI_QUANT_Y_BUFFER_NUM - 1)]);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CalLocalAddrForYVf(LocalAddressYParam<yType> &localAddressParam)
{
    uint64_t mte3BufIdx = (ubMte2AntiquantYLoopIdx_ - 1) & (UB_ANTI_QUANT_Y_BUFFER_NUM - 1);
    uint64_t antiquantParamOffset = mte3BufIdx * ANTI_QUANT_Y_PER_CHANNEL_SCALE_SINGLE_BUFFER_SIZE;
    localAddressParam.yOriginPhyAddr = (__local_mem__ int32_t *)ubHighBitTotalBuffer_.GetPhyAddr();
    localAddressParam.yPhyAddr = (__local_mem__ yType *)localAddressParam.yOriginPhyAddr;
    localAddressParam.cScalePhyAddr =
        (__local_mem__ float *)ubAntiQuantYPerChannelScaleTotalBuffer_.GetPhyAddr(antiquantParamOffset);
    localAddressParam.kScalePhyAddr =
        (__local_mem__ float *)ubAntiQuantYPerTokenScaleTotalBuffer_.GetPhyAddr(antiquantParamOffset);
    if (hasBias_) {
        localAddressParam.biasPhyAddr =
            (__local_mem__ float *)ubAntiQuantYBiasTotalBuffer_.GetPhyAddr(antiquantParamOffset);
    }
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void
GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::CopyYUbToGm(uint64_t nRealL0Size, uint64_t mRealL0Size,
                                                            __gm__ half *yGm, const BasicBlockOffsetParam &offsetParam,
                                                            uint64_t aivMOffset)
{
    if (unlikely(mRealL0Size == 0)) {
        return;
    }
    antiQuantYF16Global_.SetGlobalBuffer(yGm);
    uint64_t yGmAddrOffset = (offsetParam.mOffset + aivMOffset) * offsetParam.nSize + offsetParam.nOffset;

    DataCopyPad2D(antiQuantYF16Global_[yGmAddrOffset], ubHighBitTotalBuffer_.template ReinterpretCast<half>(),
                  mRealL0Size, nRealL0Size, CeilAlign(nRealL0Size, ANTIQUANT_Y_STANDARD_N_SIZE) * 2, offsetParam.nSize);

    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
}

GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VEC_ANTIQUANT_COMPUTE_BASIC_BLOCK_CLASS::End()
{
    TEventID vecEventIdVToMte2[QUADRUPLE_BUFFER_NUM] = {vecEventIdVToMte2_[0], vecEventIdVToMte2_[1],
                                                        vecEventIdVToMte2_[2], vecEventIdVToMte2_[3]};
    TEventID vecEventIdMte3ToV[QUADRUPLE_BUFFER_NUM] = {vecEventIdMte3ToV_[0], vecEventIdMte3ToV_[1],
                                                        vecEventIdMte3ToV_[2], vecEventIdMte3ToV_[3]};
    TEventID vecEventIdAntiQuantYVToMte2[UB_ANTI_QUANT_Y_BUFFER_NUM] = {vecEventIdAntiQuantYVToMte2_[0],
                                                                        vecEventIdAntiQuantYVToMte2_[1]};

    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        for (uint16_t idx = 0; idx < ubTransScaleLoopIdx_ && idx < DOUBLE_BUFFER_NUM; idx++) {
            WaitFlag<HardEvent::MTE3_V>(vecEventIdTransScaleMte3ToV_[idx]);
        }
    }

    for (uint16_t idx = 0; idx < ubComputeLoopIdx_ && idx < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; idx++) {
        WaitFlag<HardEvent::MTE3_V>(vecEventIdMte3ToV[idx]);
    }

    for (uint16_t idx = 0; idx < ubMte2LoopIdx_ && idx < vecConfig.ubMte2BufferNum; idx++) {
        WaitFlag<HardEvent::V_MTE2>(vecEventIdVToMte2[idx]);
    }

    if constexpr (IsSameType<xType, int8_t>::value) {
        for (uint16_t idx = 0; idx < ubMte2AntiquantYLoopIdx_ && idx < UB_ANTI_QUANT_Y_BUFFER_NUM; idx++) {
            WaitFlag<HardEvent::V_MTE2>(vecEventIdAntiQuantYVToMte2[idx]);
        }
    }

    for (uint16_t idx = 0; idx < vecConfig.ubMte2BufferNum; idx++) {
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vecEventIdVToMte2[idx]);
    }

    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        for (uint16_t idx = 0; idx < DOUBLE_BUFFER_NUM; idx++) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(vecEventIdTransScaleMte3ToV_[idx]);
        }
    }

    for (uint16_t idx = 0; idx < UB_BUFFER_INFO.ubWeightOutputHighBitBufferNum; idx++) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(vecEventIdMte3ToV[idx]);
    }
    if constexpr (IsSameType<xType, int8_t>::value) {
        for (uint16_t idx = 0; idx < UB_ANTI_QUANT_Y_BUFFER_NUM; idx++) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vecEventIdAntiQuantYVToMte2[idx]);
        }
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35

#endif // GROUPED_MATMUL_WEIGHT_QUANT_VEC_COMPUTE_H
