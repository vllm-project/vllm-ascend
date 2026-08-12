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
 * \file mc2_quant_batch_matmul.h
 * \brief
 */

#ifndef NEW_MC2_QUANT_BATCH_MATMUL
#define NEW_MC2_QUANT_BATCH_MATMUL

#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_base.h"
#include "mc2_quant_batch_matmul_asw_block.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                                    \
    template <class X1Type, class X2Type, class ScaleType, class BiasType, class YType, CubeFormat FormatX1,           \
              CubeFormat FormatX2, CubeFormat FormatY, bool ATrans, bool BTrans>
#define LOCAL_TEMPLATE_FUNC_PARAMS                                                                                     \
    X1Type, X2Type, ScaleType, BiasType, YType, FormatX1, FormatX2, FormatY, ATrans, BTrans
namespace Mc2MatmulV3 {

constexpr uint64_t DEQ_SCALE_MUL = 0xFFFFE000;
using namespace AscendC;
LOCAL_TEMPLATE_CLASS_PARAMS
class Mc2QuantBatchMatmulASWKernel {
public:
    __aicore__ inline Mc2QuantBatchMatmulASWKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *que,
                                const Mc2Tiling::RCSTiling &cfg, bool isTail, bool isGather, uint64_t preCoreNum);
    __aicore__ inline void UpdateSlice(uint32_t idx, bool isTail);
    __aicore__ inline void Process(bool isLast = true);
    __aicore__ inline uint64_t GetPreCoreNum();
    __aicore__ inline void MmEnd();

protected:
    __aicore__ inline void ComputeBasicOptiLoop(bool isLast = false);
    __aicore__ inline void SetMMParaAndCompute();
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR perTokenScale,
                                            GM_ADDR scale, GM_ADDR cGM, GM_ADDR workSpace);

    uint32_t blockIdx_;
    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *quantBmmTilingData_;

    GlobalTensor<X1Type> aGlobal_;
    GlobalTensor<X2Type> bGlobal_;
    GlobalTensor<YType> cGlobal_;
    GlobalTensor<BiasType> biasGlobal_;
    Mc2MatmulV3::QuantBatchMatmulAswBlock block_;

    GlobalTensor<fp8_e8m0_t> scaleAGlobal_;
    GlobalTensor<fp8_e8m0_t> scaleBGlobal_;
    GlobalTensor<uint64_t> scaleGlobal_;

    using AType = typename AscendC::Conditional<
        DequantBmm::IsMxType<ScaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, FormatX1, X1Type, ATrans>,
        matmul::MatmulType<AscendC::TPosition::GM, FormatX1, X1Type, ATrans>>::type;
    using BType = typename AscendC::Conditional<
        DequantBmm::IsMxType<ScaleType>(),
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, FormatX2, X2Type, BTrans>,
        matmul::MatmulType<AscendC::TPosition::GM, FormatX2, X2Type, BTrans>>::type;
    using CType = matmul::MatmulType<AscendC::TPosition::GM, FormatY, YType>;
    using BiasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    using MmType = typename AscendC::Conditional<
        DequantBmm::IsMxType<ScaleType>(),
        matmul::MatmulImpl<AType, BType, CType, BiasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG,
                           MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy>,
        matmul::MatmulImpl<AType, BType, CType, BiasMatmulType, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>>::type;
    MmType mm_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace,
    const void *tilingData, TPipe *que, const Mc2Tiling::RCSTiling &cfg, bool isTail, bool isGather,
    uint64_t preCoreNum)
{
    if ASCEND_IS_AIV {
        return;
    }
    quantBmmTilingData_ = static_cast<const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *>(tilingData);
    blockIdx_ = GetBlockIdx();
    block_.Init(quantBmmTilingData_, blockIdx_);
    block_.InitForMC2(cfg, isTail, isGather, preCoreNum);
    UpdateGlobalAddr(aGM, bGM, bias, scale, perTokenScale, cGM, workSpace);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&quantBmmTilingData_->matmulTiling, que);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline uint64_t Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetPreCoreNum()
{
    return block_.preCoreNum_;
}


LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace)
{
    if constexpr (DequantBmm::IsMxType<ScaleType>()) {
        scaleAGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)perTokenScale);
        scaleBGlobal_.SetGlobalBuffer((__gm__ fp8_e8m0_t *)scale);
    } else {
        if (static_cast<bool>(quantBmmTilingData_->params.isDoubleScale)) { // doubleScale
            float deqScale = (*((__gm__ float *)scale)) * (*((__gm__ float *)perTokenScale));
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t *>(&deqScale));
            block_.offset_.scaleScalar = uint32Scale & DEQ_SCALE_MUL;            // fixpipe只能取高19位
        } else if (static_cast<bool>(quantBmmTilingData_->params.isPerTensor)) { // pertensor
            if constexpr (!IsSameType<ScaleType, uint64_t>::value) {
                uint32_t uint32Scale = 0;
                if constexpr (IsSameType<ScaleType, bfloat16_t>::value) {
                    uint16_t uint16Scale = *((__gm__ uint16_t *)scale);
                    uint32Scale = uint16Scale << 16;
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
    aGlobal_.SetGlobalBuffer((__gm__ X1Type *)aGM);
    bGlobal_.SetGlobalBuffer((__gm__ X2Type *)bGM);
    if (static_cast<bool>(quantBmmTilingData_->matmulTiling.isBias)) {
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType *)bias);
    }
    cGlobal_.SetGlobalBuffer((__gm__ YType *)cGM);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateSlice(uint32_t idx, bool isTail)
{
    this->block_.template UpdateOffset<ScaleType>(idx, isTail);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process(bool isLast)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (quantBmmTilingData_->params.batchC == 1) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ComputeBasicOptiLoop(isLast);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop(bool isLast)
{
    block_.Update();
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        if (roundIndex + 1 == this->block_.params_.round && isLast) {
            uint64_t lastRoundTileCnt =
                (block_.params_.totalCnt - block_.tilingData_->matmulTiling.usedCoreNum * roundIndex) *
                block_.params_.totalTailTile;
            block_.doTailTile_ = (lastRoundTileCnt <= block_.tilingData_->matmulTiling.usedCoreNum);
        }
        uint64_t newBlockIdx =
            block_.doTailTile_ ? (GetBlockIdx() / this->block_.params_.totalTailTile) : GetBlockIdx();

        if ((roundIndex == 0) && (newBlockIdx < this->block_.preCoreNum_)) {
            continue;
        }

        if (!block_.UpdateBasicIndex(roundIndex, newBlockIdx)) {
            continue;
        }
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex, isLast);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                           quantBmmTilingData_->matmulTiling.Ka);
        // 2. compute offset
        block_.template CalcGMOffset<ATrans, BTrans, X1Type, ScaleType, FormatX2>();
        // 3. set offset and compute

        SetMMParaAndCompute();
    }
    this->block_.preCoreNum_ = this->block_.params_.totalCnt % (this->block_.tilingData_->matmulTiling.usedCoreNum);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (DequantBmm::IsMxType<ScaleType>()) {
        mm_.SetTensorScaleA(scaleAGlobal_[block_.offset_.offsetPerTokenScale], ATrans);
        mm_.SetTensorScaleB(scaleBGlobal_[block_.offset_.offsetScale], BTrans);
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
    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], ATrans);
    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], BTrans);
    mm_.Iterate();
    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC]);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void Mc2QuantBatchMatmulASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::MmEnd()
{
    mm_.End();
}
} // namespace Mc2MatmulV3
#endif // MC2_QUANT_BATCH_MATMUL
