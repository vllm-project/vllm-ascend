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
 * \file weight_quant_vcv_basic_block.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
#define GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_tool.h"
#include "weight_quant_cube_compute.h"
#include "weight_quant_vcv_basic_block_base.h"
#include "weight_quant_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace WeightQuantBatchMatmulV2::Arch35 {
#define GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM                                                              \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,             \
              typename perTokenScaleType, typename biasType, typename yType, const WqmmConfig &wqmmConfig, \
              const VecAntiQuantConfig &vecConfig>

#define GMM_WQ_VCV_BASIC_BLOCK_CLASS                                                                                \
    WeightQuantVcvMatmulBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, \
                                   wqmmConfig, vecConfig>

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
class WeightQuantVcvMatmulBasicBlock : public WeightQuantVcvMatmulBasicBlockBaseClass {
public:
    __aicore__ inline WeightQuantVcvMatmulBasicBlock(){};
    __aicore__ inline void Init(bool hasBias, uint64_t antiQuantGroupSize, uint64_t aPrefetchSize,
                                const TCubeTiling *__restrict matmulTiling, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight,
                                            __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
                                            __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale,
                                            __gm__ biasType *bias, __gm__ yType *y, const bool hasBias,
                                            const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                             const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
    __aicore__ inline void ComputeBasicBlockAivNzKn(const BasicBlockOffsetParam &curOffsetParam,
                                                    const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void IterateNzKnWithKAiv(uint64_t &kMte2Offset, uint64_t kMte2Limit, uint64_t mte2RealN,
                                               uint64_t nL1Offset, const BasicBlockOffsetParam &curOffsetParam);
    __aicore__ inline void IterateNzKnWithKAic(const BasicBlockOffsetParam &curOffsetParam);

    template <const pipe_t pipe>
    __aicore__ inline void SetAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAivToAic(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void SetAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag + FLAG_ID_MAX);
        CrossCoreSetFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    template <const pipe_t pipe>
    __aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE4, pipe>(syncFlag);
#endif
    };

    BasicBlockLibVectorAntiQuantCompute<xType, wType, antiQuantScaleType, biasType, yType, wqmmConfig, vecConfig>
        vecCompute_;
    using MMImpl = MatmulImpl<MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.aTrans>,
                              MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.bTrans>,
                              MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, int32_t>,
                              MatmulType<TPosition::TSCM, CubeFormat::ND, int32_t>, CFG_MDL,
                              matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>, WQBmmCustomPolicy>;
    WeightQuantBatchMatmulV2CubeCompute<xType, int32_t, antiQuantScaleType, perTokenScaleType, int32_t, wqmmConfig,
                                        MMImpl>
        cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightS8L1DbOffset_ = 0;

    LocalTensor<xType> weightS8L1_;
    LocalTensor<int32_t> ubOutputS32Buffer_;
};

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::Init(bool hasBias, uint64_t antiQuantGroupSize,
                                                          uint64_t aPrefetchSize,
                                                          const TCubeTiling *__restrict matmulTiling, TPipe *tPipe)
{
    uint64_t weightL1Space = matmulTiling->baseN * matmulTiling->stepKb * matmulTiling->baseK;  // weight单块大小

    weightS8L1DbOffset_ = 512 * GetKBUnit<int8_t>() - weightL1Space;
    weightS8L1_ = LocalTensor<xType>(TPosition::TSCM, 0, 512 * 1024 / sizeof(xType));

    ubOutputS32Buffer_ = LocalTensor<int32_t>(TPosition::LCM, 0, 248 * 1024 / sizeof(int32_t));

    if ASCEND_IS_AIC {
        cubeCompute_.Init(512 * 1024, weightL1Space, aPrefetchSize, matmulTiling, tPipe, 0);
    } else {
        LocalTensor<xType> ubWeightS8Buffer = ubOutputS32Buffer_.template ReinterpretCast<xType>();
        vecCompute_.InitKCG(antiQuantGroupSize, hasBias, ubWeightS8Buffer, 128 * 1024);
    }
    cvLoopIdx_ = 0;
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ wType *weight, __gm__ antiQuantScaleType *antiquantScale, __gm__ xType *antiquantOffset,
    __gm__ scaleType *scale, __gm__ perTokenScaleType *perTokenScale, __gm__ biasType *bias, __gm__ yType *y,
    const bool hasBias, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, nullptr, nullptr, nullptr, nullptr, nullptr, hasBias);
    } else {
        vecCompute_.UpdateGlobalAddr(weight, antiquantScale, nullptr, perTokenScale, scale, bias, weightL2Cacheable);
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                                                                       const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        ComputeBasicBlockAivNzKn(curOffsetParam, lastOffsetParam);
    } else {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputS32Buffer_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
            cubeCompute_.ClearAFullLoadFlag();  // 清除A全载时之前循环的set同步标记
        }
        IterateNzKnWithKAic(curOffsetParam);
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::ComputeBasicBlockAivNzKn(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;
    // vec上两core切n
    uint64_t mte2RealN = curOffsetParam.nL1Size > C0_SIZE_B8
                             ? CeilAlign(curOffsetParam.nL1Size >> 1, static_cast<uint64_t>(C0_SIZE_B8))
                             : curOffsetParam.nL1Size;
    uint64_t nL1Offset = GetSubBlockIdx() * mte2RealN;
    mte2RealN = GetSubBlockIdx() == 0 ? mte2RealN : curOffsetParam.nL1Size - mte2RealN;

    IterateNzKnWithKAiv(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size), mte2RealN,
                        nL1Offset, curOffsetParam);

    if (curCvLoopIdx > 0) {
        // y反量化在vec上两core切m
        uint64_t lastBasicBlockMSize = lastOffsetParam.mL1Size - (lastOffsetParam.mL1Size >> 1);
        uint64_t lastBasicBlockMOffset = GetSubBlockIdx() == 0 ? 0 : lastBasicBlockMSize;
        lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (lastOffsetParam.mL1Size >> 1);

        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
        vecCompute_.AntiQuantYWithKc(lastOffsetParam.nL1Size, lastBasicBlockMSize);
        vecCompute_.CopyYUbToGm(lastOffsetParam.nL1Size, lastBasicBlockMSize,
                                reinterpret_cast<__gm__ half *>(lastOffsetParam.yGmAddr), lastOffsetParam,
                                lastBasicBlockMOffset);
    }
    uint64_t antiquantYMSize = curOffsetParam.mL1Size - (curOffsetParam.mL1Size >> 1);
    uint64_t antiquantYMOffset = GetSubBlockIdx() == 0 ? 0 : antiquantYMSize;
    antiquantYMSize = GetSubBlockIdx() == 0 ? antiquantYMSize : (curOffsetParam.mL1Size >> 1);
    IterateNzKnWithKAiv(kMte2Offset, curOffsetParam.kSize, mte2RealN, nL1Offset, curOffsetParam);
    vecCompute_.CopyKcScaleBiasGmToUb(curOffsetParam.nL1Size, antiquantYMSize, curOffsetParam.nOffset,
                                      curOffsetParam.mOffset + antiquantYMOffset);
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzKnWithKAiv(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                                                         uint64_t mte2RealN, uint64_t nL1Offset,
                                                                         const BasicBlockOffsetParam &curOffsetParam)
{
    UbConsumeConfig ubConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealN = mte2RealN;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    L1ConsumeConfig l1ConsumeConfig;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = nL1Offset;
    l1ConsumeConfig.l1RealExternalLen = curOffsetParam.nL1Size;

    for (; kMte2Offset < kMte2Limit; kMte2Offset += vecConfig.ubMte2InnerSize) {
        uint64_t mte2RealK = (kMte2Offset + vecConfig.ubMte2InnerSize) > kMte2Limit
                                 ? kMte2Limit - kMte2Offset
                                 : vecConfig.ubMte2InnerSize;  // vec总共需要搬运的K方向的实际量（考虑尾块）
        vecCompute_.WaitVToMTE2();
        vecCompute_.CopyGmToUb(mte2RealN, mte2RealK, curOffsetParam.nOffset + nL1Offset, kMte2Offset, curOffsetParam);
        for (uint64_t antiquantKOffset = 0; antiquantKOffset < mte2RealK;
             antiquantKOffset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
            uint64_t antiquantRealK = (antiquantKOffset + curOffsetParam.kbL1Size) >= mte2RealK
                                          ? mte2RealK - antiquantKOffset
                                          : curOffsetParam.kbL1Size;
            if (cvLoopIdx_ > 1) {
                WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            }

            ubConsumeConfig.l1RequireVfComputeRealK = antiquantRealK;
            ubConsumeConfig.kWeightLowBitUbOffset = antiquantKOffset;
            vecCompute_.WeightAntiQuantCompute(ubConsumeConfig, weightS8L1_[(cvLoopIdx_ & 1) * weightS8L1DbOffset_],
                                               l1ConsumeConfig, nullptr);

            SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        }
        vecCompute_.SetVToMTE2();
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::IterateNzKnWithKAic(const BasicBlockOffsetParam &curOffsetParam)
{
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize
                                    ? curOffsetParam.kSize - kbL1Offset
                                    : curOffsetParam.kbL1Size;
        cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);

        cubeCompute_.LaunchMatmul(weightS8L1_[(cvLoopIdx_ & 1) * weightS8L1DbOffset_], kbL1Offset, kbL1RealSize,
                                  cvLoopIdx_,
                                  curOffsetParam); // mte1 mmad fixp流水

        cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    cubeCompute_.PrefetchA(aPrefetchSize, xSizeLimit);
}

GMM_WQ_VCV_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_VCV_BASIC_BLOCK_CLASS::End(const BasicBlockOffsetParam &curOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputS32Buffer_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();
    } else {
        if (cvLoopIdx_ > 0) {
            SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            uint64_t lastBasicBlockMSize = curOffsetParam.mL1Size - (curOffsetParam.mL1Size >> 1);
            uint64_t lastBasicBlockMOffset = GetSubBlockIdx() == 0 ? 0 : lastBasicBlockMSize;
            lastBasicBlockMSize = GetSubBlockIdx() == 0 ? lastBasicBlockMSize : (curOffsetParam.mL1Size >> 1);
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
            vecCompute_.AntiQuantYWithKc(curOffsetParam.nL1Size, lastBasicBlockMSize);
            vecCompute_.CopyYUbToGm(curOffsetParam.nL1Size, lastBasicBlockMSize,
                                    reinterpret_cast<__gm__ half *>(curOffsetParam.yGmAddr), curOffsetParam,
                                    lastBasicBlockMOffset);
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        vecCompute_.End();
    }
}
}  // namespace WeightQuantBatchMatmulV2::Arch35

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_VCV_BASIC_BLOCK_H
