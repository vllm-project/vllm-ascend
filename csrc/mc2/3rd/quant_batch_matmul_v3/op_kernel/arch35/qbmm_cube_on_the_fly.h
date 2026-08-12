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
 * \file qbmm_cube_on_the_fly.h
 * \brief
 */

#ifndef MC2_QBMM_CUBE_ON_THE_FLY_H
#define MC2_QBMM_CUBE_ON_THE_FLY_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_asw_block.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                                    \
    template <class x1Type, class x2Type, class inputScaleType, class biasType, class yType, CubeFormat formatX1,      \
              CubeFormat formatX2, CubeFormat formatY, bool aTrans, bool bTrans>
#define LOCAL_TEMPLATE_FUNC_PARAMS                                                                                     \
    x1Type, x2Type, inputScaleType, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans
namespace AscendC {

constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;

LOCAL_TEMPLATE_CLASS_PARAMS
class MatMulASWKernel {
public:
    __aicore__ inline MatMulASWKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *que);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR perTokenScale,
                                            GM_ADDR scale, GM_ADDR cGM, GM_ADDR workSpace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeBmmOptiLoop();
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void SetMMParaAndCompute();

    uint32_t blockIdx_;
    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *quantBmmTilingData_;

    GlobalTensor<x1Type> aGlobal_;
    GlobalTensor<x2Type> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock block_;

    GlobalTensor<fp8_e8m0_t> scaleAGlobal_;
    GlobalTensor<fp8_e8m0_t> scaleBGlobal_;
    GlobalTensor<uint64_t> scaleGlobal_;

    using scaleType =
        typename AscendC::Conditional<IsSameType<inputScaleType, int64_t>::value, uint64_t, inputScaleType>::type;
    using aType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, formatX1, x1Type, aTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, formatX1, x1Type, aTrans>>::type;
    using cType = matmul::MatmulType<AscendC::TPosition::GM, formatY, yType>;
    using biasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;
    using bType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, formatX2, x2Type, bTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, formatX2, x2Type, bTrans>>::type;
    using MmType = typename AscendC::Conditional<
        DequantBmm::IsMxType<scaleType>(),
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG,
                           MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy>,
        matmul::MatmulImpl<aType, bType, cType, biasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>>::type;
    MmType mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias,
                                                                         GM_ADDR scale, GM_ADDR perTokenScale,
                                                                         GM_ADDR cGM, GM_ADDR workSpace,
                                                                         const void *tilingData, TPipe *que)
{
    if ASCEND_IS_AIV {
        return;
    }
    quantBmmTilingData_ = static_cast<const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, cGM, workSpace);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&quantBmmTilingData_->matmulTiling, que);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM,
                                                                                     GM_ADDR bias, GM_ADDR scale,
                                                                                     GM_ADDR perTokenScale, GM_ADDR cGM,
                                                                                     GM_ADDR workSpace)
{
    block_.Init(quantBmmTilingData_, blockIdx_);
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        scaleAGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)perTokenScale);
        scaleBGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)scale);
    } else {
        if (static_cast<bool>(quantBmmTilingData_->params.isDoubleScale)) { // doubleScale
            float deqScale = (*((__gm__ float *)scale)) * (*((__gm__ float *)perTokenScale));
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t *>(&deqScale));
            block_.offset_.scaleScalar = uint32Scale & DEQ_SCALE_MUL;            // fixpipe只能取高19位
        } else if (static_cast<bool>(quantBmmTilingData_->params.isPerTensor)) { // pertensor
            if constexpr (!IsSameType<scaleType, uint64_t>::value) {
                uint32_t uint32Scale = 0;
                if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                    uint16_t uint16Scale = *((__gm__ uint16_t *)scale);
                    uint32Scale = static_cast<uint32_t>(uint16Scale << BMM_BLOCK_NUM);
                } else {
                    uint32Scale = *((__gm__ uint32_t *)scale);
                }
                block_.offset_.scaleScalar = uint32Scale & DEQ_SCALE_MUL;
            } else {
                block_.offset_.scaleScalar = *((__gm__ uint64_t *)scale);
            }
        } else {
            scaleGlobal_.SetGlobalBuffer((__gm__ uint64_t *)scale);
        }
    }

    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ x1Type *)aGM);
    bGlobal_.SetGlobalBuffer((__gm__ x2Type *)bGM);
    if (static_cast<bool>(quantBmmTilingData_->matmulTiling.isBias)) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType *)bias);
    }
    cGlobal_.SetGlobalBuffer((__gm__ yType *)cGM);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (quantBmmTilingData_->params.batchC == 1UL) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ComputeBasicOptiLoop();
    } else {
        ComputeBmmOptiLoop();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBmmOptiLoop()
{
    uint64_t batchC3C4 =
        static_cast<uint64_t>(quantBmmTilingData_->params.batchC3) * quantBmmTilingData_->params.batchC4;
    uint64_t batchC2C3C4 = quantBmmTilingData_->params.batchC2 * batchC3C4;
    uint64_t batchB3B4 =
        static_cast<uint64_t>(quantBmmTilingData_->params.batchB3) * quantBmmTilingData_->params.batchB4;
    uint64_t batchB2B3B4 = quantBmmTilingData_->params.batchB2 * batchB3B4;
    uint64_t batchA3A4 =
        static_cast<uint64_t>(quantBmmTilingData_->params.batchA3) * quantBmmTilingData_->params.batchA4;
    uint64_t batchA2A3A4 = quantBmmTilingData_->params.batchA2 * batchA3A4;
    uint32_t multiA1C1 = quantBmmTilingData_->params.batchA1 / quantBmmTilingData_->params.batchC1;
    uint32_t multiA2C2 = quantBmmTilingData_->params.batchA2 / quantBmmTilingData_->params.batchC2;
    uint32_t multiA3C3 = quantBmmTilingData_->params.batchA3 / quantBmmTilingData_->params.batchC3;
    uint32_t multiA4C4 = quantBmmTilingData_->params.batchA4 / quantBmmTilingData_->params.batchC4;
    uint32_t multiB1C1 = quantBmmTilingData_->params.batchB1 / quantBmmTilingData_->params.batchC1;
    uint32_t multiB2C2 = quantBmmTilingData_->params.batchB2 / quantBmmTilingData_->params.batchC2;
    uint32_t multiB3C3 = quantBmmTilingData_->params.batchB3 / quantBmmTilingData_->params.batchC3;
    uint32_t multiB4C4 = quantBmmTilingData_->params.batchB4 / quantBmmTilingData_->params.batchC4;
    for (uint64_t b1Index = 0; b1Index < quantBmmTilingData_->params.batchC1; ++b1Index) {
        uint64_t batchC1Offset = b1Index * batchC2C3C4;
        uint64_t batchA1Offset = b1Index * batchA2A3A4 * multiA1C1;
        uint64_t batchB1Offset = b1Index * batchB2B3B4 * multiB1C1;
        for (uint64_t b2Index = 0; b2Index < quantBmmTilingData_->params.batchC2; ++b2Index) {
            uint64_t batchC2Offset = b2Index * batchC3C4 + batchC1Offset;
            uint64_t batchA2Offset = b2Index * batchA3A4 * multiA2C2 + batchA1Offset;
            uint64_t batchB2Offset = b2Index * batchB3B4 * multiB2C2 + batchB1Offset;
            for (uint64_t b3Index = 0; b3Index < quantBmmTilingData_->params.batchC3; ++b3Index) {
                uint64_t batchC3Offset = b3Index * quantBmmTilingData_->params.batchC4 + batchC2Offset;
                uint64_t batchA3Offset = b3Index * quantBmmTilingData_->params.batchA4 * multiA3C3 + batchA2Offset;
                uint64_t batchB3Offset = b3Index * quantBmmTilingData_->params.batchB4 * multiB3C3 + batchB2Offset;
                for (uint64_t b4Index = 0; b4Index < quantBmmTilingData_->params.batchC4; ++b4Index) {
                    block_.offset_.batchCOffset = batchC3Offset + b4Index;
                    block_.offset_.batchAOffset = batchA3Offset + b4Index * multiA4C4;
                    block_.offset_.batchBOffset = batchB3Offset + b4Index * multiB4C4;
                    block_.ResetAddressOffsets();
                    ComputeBasicOptiLoop();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 1. Set single core param
        block_.template UpdateBlockParams<bTrans, formatX2>(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                           quantBmmTilingData_->matmulTiling.Ka);
        // 2. compute offset
        block_.template CalcGMOffset<aTrans, bTrans, x1Type, scaleType, formatX2>();
        // 3. set offset and compute
        SetMMParaAndCompute();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void MatMulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (DequantBmm::IsMxType<scaleType>()) {
        mm_.SetTensorScaleA(scaleAGlobal_[block_.offset_.offsetPerTokenScale], aTrans);
        mm_.SetTensorScaleB(scaleBGlobal_[block_.offset_.offsetScale], bTrans);
    } else {
        if (static_cast<bool>(quantBmmTilingData_->params.isPerTensor) ||
            static_cast<bool>(quantBmmTilingData_->params.isDoubleScale)) {
            mm_.SetQuantScalar(block_.offset_.scaleScalar);
        } else {
            mm_.SetQuantVector(scaleGlobal_[block_.offset_.offsetScale]);
        }
    }

    if (quantBmmTilingData_->matmulTiling.isBias) {
        mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
    }
    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC]);
}
} // namespace AscendC
#endif // QBMM_CUBE_ON_THE_FLY_H
