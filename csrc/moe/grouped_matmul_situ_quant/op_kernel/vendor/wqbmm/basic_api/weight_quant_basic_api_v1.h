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
 * \file weight_quant_basic_api_v1.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_API_V1_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_API_V1_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"

using AscendC::BLOCK_CUBE;
using AscendC::fp8_e8m0_t;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace WeightQuantBatchMatmulV2::Arch35 {
struct BasicApiParamsV1 {
    uint64_t isBias;
    uint64_t gmKOffset;
    uint64_t l1KaSize;
    uint64_t l1KbSize;
    uint64_t l0NSize;
    uint64_t l0MSize;
};

#define WQBMM_BASIC_API_V1_TEMPLATE_PARAM                                                                              \
    template <typename xType, typename biasType, typename yType, bool aTrans, bool bTrans>

#define WQBMM_BASIC_API_V1_CLASS WqbmmBasicApiV1<xType, biasType, yType, aTrans, bTrans>

WQBMM_BASIC_API_V1_TEMPLATE_PARAM class WqbmmBasicApiV1 {
    using L0DataType = typename AscendC::GetL0DataType<xType, true>::Type;

public:
    __aicore__ inline WqbmmBasicApiV1(){};
    __aicore__ inline void Init();
    __aicore__ inline void Iterate(bool isLast, bool isFirst, const LocalTensor<xType> &aL1,
                                   const LocalTensor<fp8_e8m0_t> &aMxScaleL1, const LocalTensor<xType> &bL1,
                                   const LocalTensor<fp8_e8m0_t> &bMxScaleL1, const LocalTensor<biasType> &biasL1,
                                   const BasicApiParamsV1 &basicApiParams);
    __aicore__ inline void GetTensorC(uint64_t mL0Size, uint64_t nL0Size, uint64_t nDstSize,
                                      const GlobalTensor<yType> &yGm);
    __aicore__ inline void End();

private:
    __aicore__ inline void LoadAL1ToL0A(uint64_t baseK, uint64_t loopId, uint64_t l1KOffset,
                                        const BasicApiParamsV1 &basicApiParams, const LocalTensor<xType> &aL1Tensor,
                                        const LocalTensor<fp8_e8m0_t> &scaleAL1Tensor);
    __aicore__ inline void LoadBL1ToL0B(uint64_t baseK, uint64_t loopId, uint64_t l1KOffset,
                                        const BasicApiParamsV1 &basicApiParams, const LocalTensor<xType> &bL1Tensor,
                                        const LocalTensor<fp8_e8m0_t> &scaleBL1Tensor);
    __aicore__ inline void LoadBiasToBt(uint64_t nL0Size, uint64_t loopId, const LocalTensor<biasType> &biasL1Tensor);
    __aicore__ inline void Mmad(bool isLast, bool isFirst, uint64_t kL0RealSize, uint64_t loopId,
                                const BasicApiParamsV1 &basicApiParams);

    static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t BIAS_TABLE_OFFSET_B32 = 2 * 256;
    static constexpr uint64_t MX_GROUP_SIZE = 32;
    static constexpr uint64_t MX_UNIT_BYTES = 32 / AscendC::BLOCK_CUBE;

    // unitFlag: 2表示累加中, 3表示累加结束
    static constexpr uint16_t UNIT_FLAG_ENABLE = 2;
    static constexpr uint16_t UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;

    TEventID eventIdMToMte1_ = 3;
    TEventID eventIdMte1ToM_ = 3;

    uint64_t l0LoopIdx_ = 0;

    LocalTensor<L0DataType> l0a_{TPosition::A2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<L0DataType> l0b_{TPosition::B2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};
    LocalTensor<float> l0c_{TPosition::CO1, 0, 256 * GetKBUnit<int8_t>() / sizeof(float)};
    LocalTensor<float> biasTable_{TPosition::C2, 0, 4 * GetKBUnit<int8_t>() / sizeof(float)};
};

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::Init()
{
    // 设置反向同步
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::Iterate(bool isLastGmK, bool isFirstGmK, const LocalTensor<xType> &aL1,
                                                         const LocalTensor<fp8_e8m0_t> &aMxScaleL1,
                                                         const LocalTensor<xType> &bL1,
                                                         const LocalTensor<fp8_e8m0_t> &bMxScaleL1,
                                                         const LocalTensor<biasType> &biasL1,
                                                         const BasicApiParamsV1 &basicApiParams)
{
    uint64_t l0KSize = (basicApiParams.l0MSize <= 128 && basicApiParams.l0NSize <= 128) ? 256 : 128;
    for (uint64_t l1KOffset = 0; l1KOffset < basicApiParams.l1KbSize; l1KOffset += l0KSize) {
        bool isLastL1K = l1KOffset + l0KSize >= basicApiParams.l1KbSize;
        uint64_t realL0k = isLastL1K ? basicApiParams.l1KbSize - l1KOffset : l0KSize;
        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM; // 等价于 l0LoopIdx_ % L0_BUF_NUM，减少scalar
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);
        LoadAL1ToL0A(realL0k, loopId, l1KOffset, basicApiParams, aL1, aMxScaleL1);
        LoadBL1ToL0B(realL0k, loopId, l1KOffset, basicApiParams, bL1, bMxScaleL1);
        if (basicApiParams.isBias && isFirstGmK && l1KOffset == 0) {
            LoadBiasToBt(basicApiParams.l0NSize, loopId, biasL1);
        }
        Mmad(isLastGmK && isLastL1K, isFirstGmK && l1KOffset == 0, realL0k, loopId, basicApiParams);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);
        l0LoopIdx_++;
    }
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::LoadAL1ToL0A(uint64_t baseK, uint64_t loopId, uint64_t l1KOffset,
                                                              const BasicApiParamsV1 &basicApiParams,
                                                              const LocalTensor<xType> &aL1Tensor,
                                                              const LocalTensor<fp8_e8m0_t> &scaleAL1Tensor)
{
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition =
        CeilDivide((l1KOffset + basicApiParams.gmKOffset) % basicApiParams.l1KaSize * sizeof(xType), 32UL); // 单位32B
    aL0Load2dParams.mStep =
        CeilDivide(basicApiParams.l0MSize, static_cast<uint64_t>(AscendC::BLOCK_CUBE)); // 单位16元素
    aL0Load2dParams.kStep = CeilDivide(baseK, 32UL);                                    // 单位32B
    aL0Load2dParams.srcStride = CeilDivide(basicApiParams.l0MSize * 32, 512UL);         // 32为K0，stride单位为512B
    aL0Load2dParams.dstStride = aL0Load2dParams.srcStride; // L1和L0 M大小一样，k方向stride也一样

    AscendC::LoadData2DMxParams aL0Load2dMx;
    aL0Load2dMx.xStartPosition = 0;
    aL0Load2dMx.yStartPosition = CeilDivide((l1KOffset + basicApiParams.gmKOffset) % MX_SCALE_K_L1_SIZE,
                                            MX_GROUP_SIZE * MX_UNIT_BYTES); // 单位是32B
    aL0Load2dMx.xStep =
        CeilDivide(basicApiParams.l0MSize, static_cast<uint64_t>(AscendC::BLOCK_CUBE)); // 单位是1个32B分形
    aL0Load2dMx.yStep =
        CeilDivide(baseK, MX_GROUP_SIZE * MX_UNIT_BYTES); // mxK = K/32, (baseK / MX_GROUP_SIZE) / MX_UNIT_BYTES
    aL0Load2dMx.srcStride = CeilDivide(MX_SCALE_K_L1_SIZE, MX_GROUP_SIZE * MX_UNIT_BYTES);
    aL0Load2dMx.dstStride = CeilDivide(baseK, MX_GROUP_SIZE * MX_UNIT_BYTES);
    AscendC::LoadData(l0a_[loopId * L0_BUF_OFFSET_B8], aL1Tensor, scaleAL1Tensor, aL0Load2dParams, aL0Load2dMx);
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::LoadBL1ToL0B(uint64_t baseK, uint64_t loopId, uint64_t l1KOffset,
                                                              const BasicApiParamsV1 &basicApiParams,
                                                              const LocalTensor<xType> &bL1Tensor,
                                                              const LocalTensor<fp8_e8m0_t> &scaleBL1Tensor)
{
    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition = CeilDivide(static_cast<uint64_t>(l1KOffset * sizeof(xType)), 32UL); // 单位32B
    bL0Load2dParams.mStep =
        CeilDivide(basicApiParams.l0NSize, static_cast<uint64_t>(AscendC::BLOCK_CUBE)); // 单位16元素
    bL0Load2dParams.kStep = CeilDivide(baseK, 32UL);                                    // 单位32B
    bL0Load2dParams.srcStride = CeilDivide(basicApiParams.l0NSize * 32, 512UL);         // 32为K0，stride单位为512B
    bL0Load2dParams.dstStride = bL0Load2dParams.srcStride; // L1和L0 N方向大小一样，k方向stride也一样

    AscendC::LoadData2DMxParams bL0Load2dMx;
    bL0Load2dMx.xStartPosition = 0;
    bL0Load2dMx.yStartPosition = CeilDivide((l1KOffset + basicApiParams.gmKOffset) % MX_SCALE_K_L1_SIZE,
                                            MX_GROUP_SIZE * MX_UNIT_BYTES); // 单位是32B
    bL0Load2dMx.xStep =
        CeilDivide(basicApiParams.l0NSize, static_cast<uint64_t>(AscendC::BLOCK_CUBE)); // 单位是1个32B分形，
    bL0Load2dMx.yStep = CeilDivide(baseK, MX_GROUP_SIZE * MX_UNIT_BYTES);               // baseK / 32 / 2
    bL0Load2dMx.srcStride = CeilDivide(MX_SCALE_K_L1_SIZE, MX_GROUP_SIZE * MX_UNIT_BYTES);
    bL0Load2dMx.dstStride = CeilDivide(baseK, MX_GROUP_SIZE * MX_UNIT_BYTES);
    AscendC::LoadData(l0b_[loopId * L0_BUF_OFFSET_B8], bL1Tensor, scaleBL1Tensor, bL0Load2dParams, bL0Load2dMx);
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::LoadBiasToBt(uint64_t nL0Size, uint64_t loopId,
                                                              const LocalTensor<biasType> &biasL1Tensor)
{
    AscendC::DataCopy(biasTable_[loopId * BIAS_TABLE_OFFSET_B32], biasL1Tensor,
                      {1, static_cast<uint16_t>(CeilDivide(nL0Size * sizeof(float), 64UL)), 0, 0});
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::Mmad(bool isLastK, bool isFirstK, uint64_t kL0RealSize,
                                                      uint64_t loopId, const BasicApiParamsV1 &basicApiParams)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = basicApiParams.l0MSize;
    mmadParams.n = basicApiParams.l0NSize;
    mmadParams.k = kL0RealSize;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = isLastK ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;
    mmadParams.disableGemv = true;
    SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
    WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
    if (basicApiParams.isBias && isFirstK) {
        mmadParams.cmatrixInitVal = false;
        AscendC::Mmad(l0c_, l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8],
                      biasTable_[loopId * BIAS_TABLE_OFFSET_B32], mmadParams);
    } else {
        mmadParams.cmatrixInitVal = isFirstK;
        AscendC::Mmad(l0c_, l0a_[loopId * L0_BUF_OFFSET_B8], l0b_[loopId * L0_BUF_OFFSET_B8], mmadParams);
    }
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::GetTensorC(uint64_t mL0Size, uint64_t nL0Size, uint64_t nDstSize,
                                                            const GlobalTensor<yType> &yGm)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;

    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    fixParams.mSize = mL0Size;
    fixParams.nSize = nL0Size;
    fixParams.srcStride = CeilAlign(mL0Size, static_cast<int64_t>(BLOCK_CUBE));
    fixParams.dstStride = nDstSize;
    if constexpr (IsSameType<yType, bfloat16_t>::value) {
        fixParams.quantPre = QuantMode_t::QF322BF16_PRE;
    } else {
        fixParams.quantPre = QuantMode_t::QF322F16_PRE;
    }

    fixParams.deqScalar = FP32_64_AS_UINT64;
    fixParams.unitFlag = UNIT_FLAG_ENABLE_AUTO_CLOSE;
    fixParams.params.ndNum = 1;
    AscendC::Fixpipe<yType, float, AscendC::CFG_ROW_MAJOR>(yGm, l0c_, fixParams);
}

WQBMM_BASIC_API_V1_TEMPLATE_PARAM
__aicore__ inline void WQBMM_BASIC_API_V1_CLASS::End()
{
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35
#endif