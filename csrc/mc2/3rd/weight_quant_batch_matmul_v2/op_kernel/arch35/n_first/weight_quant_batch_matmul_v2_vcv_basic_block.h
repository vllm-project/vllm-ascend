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
 * \file weight_quant_batch_matmul_v2_vcv_basic_block.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_VCV_BASIC_BLOCK_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_VCV_BASIC_BLOCK_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_cube_compute.h"
#include "weight_quant_batch_matmul_v2_vec_compute.h"

using AscendC::GetSubBlockIdx;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
class WeightQuantVcvMatmulBasicBlock {
public:
    __aicore__ inline WeightQuantVcvMatmulBasicBlock(){};
    __aicore__ inline void Init(bool hasBias, uint64_t antiQuantGroupSize, uint64_t aPreloadSize,
                                const TCubeTiling *__restrict matmulTiling, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ wType *weight, __gm__ half *antiquantScale,
                                            __gm__ float *scale, __gm__ float *perTokenScale, __gm__ biasType *bias,
                                            const bool hasBias, const bool weightL2Cacheable);
    __aicore__ inline void ComputeBasicBlockNzKn(const BasicBlockOffsetParam &curOffsetParam,
                                                 const BasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void End(const BasicBlockOffsetParam &curOffsetParam);

protected:
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

    BasicBlockLibVectorAntiQuantCompute<xType, wType, half, yType, wqmmConfig, vecConfig> vecCompute_;
    using MMImpl = MatmulImpl<MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.aTrans>,
                              MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, xType, wqmmConfig.bTrans>,
                              MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, int32_t>,
                              MatmulType<TPosition::TSCM, CubeFormat::ND, int32_t>, CFG_MDL,
                              matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>, WQBmmCustomPolicy>;
    WeightQuantBatchMatmulV2CubeCompute<xType, int32_t, int32_t, wqmmConfig, MMImpl> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightS8L1DbOffset_ = 0;

    LocalTensor<xType> weightS8L1_;
    LocalTensor<int32_t> ubOutputS32Buffer_;
};

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::Init(
    bool hasBias, uint64_t antiQuantGroupSize, uint64_t aPreloadSize, const TCubeTiling *__restrict matmulTiling,
    TPipe *tPipe)
{
    uint64_t weightL1Space = matmulTiling->baseN * matmulTiling->stepKb * matmulTiling->baseK; // weight单块大小

    TBuf<TPosition::TSCM> l1Tbuf;
    tPipe->InitBuffer(l1Tbuf, 512 * 1024);
    weightS8L1DbOffset_ = 512 * GetKBUnit<int8_t>() - weightL1Space;
    weightS8L1_ = l1Tbuf.Get<xType>();

    TBuf<> ubBuffer;
    tPipe->InitBuffer(ubBuffer, 248 * 1024);
    ubOutputS32Buffer_ = ubBuffer.Get<int32_t>();

    if ASCEND_IS_AIC {
        cubeCompute_.Init(l1Tbuf, weightL1Space, aPreloadSize, matmulTiling, tPipe);
    } else {
        LocalTensor<xType> ubWeightS8Buffer = ubOutputS32Buffer_.template ReinterpretCast<xType>();
        vecCompute_.InitKCG(antiQuantGroupSize, hasBias, ubBuffer, ubWeightS8Buffer, 128 * 1024);
    }
    cvLoopIdx_ = 0;
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void
WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::UpdateGlobalAddr(
    __gm__ xType *x, __gm__ wType *weight, __gm__ half *antiquantScale, __gm__ float *scale,
    __gm__ float *perTokenScale, __gm__ biasType *bias, const bool hasBias, const bool weightL2Cacheable)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(x, nullptr, nullptr, nullptr, hasBias);
    } else {
        vecCompute_.UpdateGlobalAddr(weight, antiquantScale, nullptr, perTokenScale, scale, bias, weightL2Cacheable);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void
WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::ComputeBasicBlockNzKn(
    const BasicBlockOffsetParam &curOffsetParam, const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        uint64_t kMte2Offset = 0;
        uint64_t curCvLoopIdx = cvLoopIdx_;
        // vec上两core切n
        uint64_t mte2RealN = curOffsetParam.nL1Size > C0_SIZE_B8 ?
                                 CeilAlign(curOffsetParam.nL1Size >> 1, static_cast<uint64_t>(C0_SIZE_B8)) :
                                 curOffsetParam.nL1Size;
        uint64_t nL1Offset = GetSubBlockIdx() * mte2RealN;
        mte2RealN = GetSubBlockIdx() == 0 ? mte2RealN : curOffsetParam.nL1Size - mte2RealN;

        IterateNzKnWithKAiv(kMte2Offset, Min(curOffsetParam.kSize, DOUBLE_BUFFER_NUM * curOffsetParam.kbL1Size),
                            mte2RealN, nL1Offset, curOffsetParam);

        if (curCvLoopIdx > 0) {
            // y反量化再vec上两core切m
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
    } else {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputS32Buffer_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
            cubeCompute_.ClearAFullLoadFlag(); // 清除A全载时之前循环的set同步标记
        }
        IterateNzKnWithKAic(curOffsetParam);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void
WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::IterateNzKnWithKAiv(
    uint64_t &kMte2Offset, uint64_t kMte2Limit, uint64_t mte2RealN, uint64_t nL1Offset,
    const BasicBlockOffsetParam &curOffsetParam)
{
    UbConsumeConfig ubConsumeConfig;
    ubConsumeConfig.l1RequireVfComputeRealN = mte2RealN;
    ubConsumeConfig.nWeightLowBitUbOffset = 0;
    L1ConsumeConfig l1ConsumeConfig;
    l1ConsumeConfig.l1SplitTwoVecExternalOffset = nL1Offset;
    l1ConsumeConfig.l1RealExternalLen = curOffsetParam.nL1Size;

    for (; kMte2Offset < kMte2Limit; kMte2Offset += vecConfig.ubMte2InnerSize) {
        uint64_t mte2RealK = (kMte2Offset + vecConfig.ubMte2InnerSize) > kMte2Limit ?
                                 kMte2Limit - kMte2Offset :
                                 vecConfig.ubMte2InnerSize; // vec总共需要搬运的K方向的实际量（考虑尾块）
        vecCompute_.WaitVToMTE2();
        vecCompute_.CopyGmToUb(mte2RealN, mte2RealK, curOffsetParam.nOffset + nL1Offset, kMte2Offset, curOffsetParam);
        for (uint64_t antiquantKOffset = 0; antiquantKOffset < mte2RealK;
             antiquantKOffset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
            uint64_t antiquantRealK = (antiquantKOffset + curOffsetParam.kbL1Size) >= mte2RealK ?
                                          mte2RealK - antiquantKOffset :
                                          curOffsetParam.kbL1Size;
            if (cvLoopIdx_ > 1) {
                WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            }

            ubConsumeConfig.l1RequireVfComputeRealK = antiquantRealK;
            ubConsumeConfig.kWeightLowBitUbOffset = antiquantKOffset;
            vecCompute_.WeightAntiQuantCompute(ubConsumeConfig, weightS8L1_[(cvLoopIdx_ & 1) * weightS8L1DbOffset_],
                                               l1ConsumeConfig);

            SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        }
        vecCompute_.SetVToMTE2();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void
WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::IterateNzKnWithKAic(
    const BasicBlockOffsetParam &curOffsetParam)
{
    for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
         kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbL1Offset + curOffsetParam.kbL1Size) >= curOffsetParam.kSize ?
                                    curOffsetParam.kSize - kbL1Offset :
                                    curOffsetParam.kbL1Size;
        cubeCompute_.WaitMTE1ToMTE2(cvLoopIdx_);
        cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, kbL1RealSize, curOffsetParam.nL1Size, cvLoopIdx_);
        WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);

        cubeCompute_.LaunchMatmul(weightS8L1_[(cvLoopIdx_ & 1) * weightS8L1DbOffset_], kbL1Offset, kbL1RealSize,
                                  curOffsetParam,
                                  cvLoopIdx_); // mte1 mmad fixp流水

        cubeCompute_.SetMTE1ToMTE2(cvLoopIdx_);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
__aicore__ inline void WeightQuantVcvMatmulBasicBlock<xType, wType, biasType, yType, wqmmConfig, vecConfig>::End(
    const BasicBlockOffsetParam &curOffsetParam)
{
    if ASCEND_IS_AIC {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(ubOutputS32Buffer_);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync(cvLoopIdx_);
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
        }
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        if (cvLoopIdx_ > 1) {
            WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        }
        vecCompute_.End();
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_H
