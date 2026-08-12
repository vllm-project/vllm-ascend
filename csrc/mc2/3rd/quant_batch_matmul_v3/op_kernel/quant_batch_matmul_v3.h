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
 * \file quant_batch_matmul_v3.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_H
#define MC2_QUANT_BATCH_MATMUL_V3_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
constexpr MatmulConfig BMM_DEQUANT_MDL_CFG = GetMDLConfig(false, false, 0, true);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
constexpr MatmulConfig BMM_DEQUANT_MDL_CFG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, true, false, false, true);
#else
constexpr MatmulConfig BMM_DEQUANT_MDL_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
constexpr MatmulConfig BMM_DEQUANT_PRELOAD_CFG =
    GetMDLConfig(false, false, 2, false, false, false, false, true, true, false, false, true);
#else
constexpr MatmulConfig BMM_DEQUANT_PRELOAD_CFG = GetMDLConfig(false, false, 2);
#endif

template <typename xType, typename wType, int32_t fFormat, int32_t wFormat, typename biasType, typename scaleType,
          typename yType, bool aTrans, bool bTrans, const MatmulConfig &BMM_DEQUANT_CFG = BMM_DEQUANT_MDL_CFG>
class BmmDequant {
public:
    __aicore__ inline BmmDequant(){};

    __aicore__ inline void InitTbuf(TBuf<TPosition::VECCALC> tbuf)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        mmLocalWorkSpace_ = tbuf;
#endif
    }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR y, GM_ADDR workSpace,
                                const Mc2QuantBatchMatmulV3TilingData *tilingData, TPipe *tPipe)
    {
        InitTilingData(tilingData);

        // init global buffer
        UpdateGlobalAddr(x1, x2, bias, scale, y, workSpace);

        mm_.SetSubBlockIdx(0);
        mm_.Init(&(tilingData->matmulTiling), tPipe);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
        if (tilingData->params.ubSize > 0) {
            tPipe->InitBuffer(mmLocalWorkSpace_, tilingData->params.ubSize);
            LocalTensor<uint8_t> tmpUb = mmLocalWorkSpace_.template Get<uint8_t>();
            mm_.SetLocalWorkspace(tmpUb);
        }
#endif
    }

    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR y,
                                            GM_ADDR workSpace)
    {
        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ scaleType *)scale);
        }
        // update global buffer
        xGm_.SetGlobalBuffer((__gm__ xType *)x1);
        weightGm_.SetGlobalBuffer((__gm__ wType *)x2);
        if (m_ <= baseM_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (hasBias_ != 0U) {
            biasGm_.SetGlobalBuffer((__gm__ biasType *)bias);
        }
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType *)scale);
    }

    /** main logical function
     */
    __aicore__ inline void Process(bool enAtomic = false)
    {
        if ASCEND_IS_AIV {
            return;
        }
        uint32_t batchDim = DequantBmm::CeilDiv(batch_, singleCoreBatch_);
        uint32_t mDim = DequantBmm::CeilDiv(m_, singleCoreM_);
        uint32_t nDim = DequantBmm::CeilDiv(n_, singleCoreN_);
        uint32_t kDim = DequantBmm::CeilDiv(k_, singleCoreK_);
        logicCoreNum_ = batchDim * mDim * nDim * kDim;
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        if (blockIdx >= usedCoreNum_) {
            return;
        }

        uint32_t divideBatchcoreNum = logicCoreNum_ / batchDim;
        for (uint32_t logicBlockIdx = blockIdx; logicBlockIdx < logicCoreNum_; logicBlockIdx += usedCoreNum_) {
            uint32_t kCoreIndx = (logicBlockIdx % divideBatchcoreNum) % kDim;
            uint32_t mCoreIndx = ((logicBlockIdx % divideBatchcoreNum) / kDim) % mDim; // 必须沿着N 轴方向输出
            uint32_t nCoreIndx = ((logicBlockIdx % divideBatchcoreNum) / kDim) / mDim;
            uint32_t batchCoreIndx = logicBlockIdx / divideBatchcoreNum;

            uint32_t gmUseBatch = batch_ - batchCoreIndx * singleCoreBatch_;
            uint32_t singleCoreBatchUpdate = gmUseBatch < singleCoreBatch_ ? gmUseBatch : singleCoreBatch_;
            uint32_t gmUseM = m_ - mCoreIndx * singleCoreM_;
            uint32_t singleCoreMUpdate = gmUseM < singleCoreM_ ? gmUseM : singleCoreM_;

            uint32_t gmUseN = n_ - nCoreIndx * singleCoreN_;
            uint32_t singleCoreNUpdate = gmUseN < singleCoreN_ ? gmUseN : singleCoreN_;

            uint32_t gmUseK = k_ - kCoreIndx * singleCoreK_;
            uint32_t singleCoreKUpdate = gmUseK < singleCoreK_ ? gmUseK : singleCoreK_;

            mm_.SetSingleShape(singleCoreMUpdate, singleCoreNUpdate, singleCoreKUpdate);
            for (uint32_t i = 0; i < singleCoreBatchUpdate; ++i) {
                CalcOffset(i, batchCoreIndx, mCoreIndx, nCoreIndx, kCoreIndx);
                MMCompute(enAtomic, kCoreIndx);
            }
        }
    }

protected:
    // define matmul
    static constexpr CubeFormat aFormat = DequantBmm::GetFormat(fFormat);
    static constexpr CubeFormat bFormat = DequantBmm::GetFormat(wFormat);

    using AMatmulType = matmul::MatmulType<TPosition::GM, aFormat, xType, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, bFormat, wType, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    using CMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, BMM_DEQUANT_CFG> mm_;

    GlobalTensor<xType> xGm_;
    GlobalTensor<wType> weightGm_;
    GlobalTensor<biasType> biasGm_;
    GlobalTensor<yType> yGm_;

    GlobalTensor<scaleType> scaleGm_;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    TBuf<> mmLocalWorkSpace_;
#endif

    uint32_t hasBias_;
    uint64_t scaleScalar_;
    uint64_t offsetA_;
    uint64_t offsetB_;
    uint64_t offsetC_;
    uint64_t offsetBias_;
    uint64_t offsetScale_;
    // tiling data

    uint32_t isPerTensor_;
    uint32_t biasThreeDim_;
    uint32_t batch_;
    uint32_t batchA_;
    uint32_t batchB_;
    uint32_t singleCoreBatch_;
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
    uint32_t singleCoreM_;
    uint32_t singleCoreN_;
    uint32_t singleCoreK_;
    uint32_t usedCoreNum_;
    uint32_t baseM_;
    uint32_t logicCoreNum_;

    /** init function for TilingData of mm1 and mm2.
     */
    __aicore__ inline void InitTilingData(const Mc2QuantBatchMatmulV3TilingData *tilingData)
    {
        hasBias_ = tilingData->matmulTiling.isBias;
        batchA_ = tilingData->params.batchA;
        batchB_ = tilingData->params.batchB;
        batch_ = tilingData->params.batchC;
        singleCoreBatch_ = tilingData->params.singleCoreBatch;
        isPerTensor_ = tilingData->params.isPerTensor;
        biasThreeDim_ = tilingData->params.biasThreeDim;
        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        k_ = tilingData->matmulTiling.Ka;
        singleCoreM_ = tilingData->matmulTiling.singleCoreM;
        singleCoreN_ = tilingData->matmulTiling.singleCoreN;
        singleCoreK_ = tilingData->matmulTiling.singleCoreK;
        baseM_ = tilingData->matmulTiling.baseM;
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
    }

    __aicore__ inline void CalcOffset(uint32_t batchIndex, uint32_t batchCoreIndx, uint32_t mCoreIndx,
                                      uint32_t nCoreIndx, uint32_t kCoreIndx)
    {
        uint64_t batchAOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchA_);
        uint64_t batchBOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchB_);
        uint64_t batchCOffset = static_cast<uint64_t>(batchCoreIndx * (singleCoreBatch_) + batchIndex);
        uint64_t mOffset = static_cast<uint64_t>(mCoreIndx * singleCoreM_);
        uint64_t nOffset = static_cast<uint64_t>(nCoreIndx * singleCoreN_);
        uint64_t kOffset = static_cast<uint64_t>(kCoreIndx * singleCoreK_);
        if constexpr (AMatmulType::format == CubeFormat::ND) {
            offsetA_ = (aTrans > 0) ? mOffset + kOffset * m_ : mOffset * k_ + kOffset;
            offsetA_ = batchAOffset * static_cast<uint64_t>(k_ * m_) + offsetA_;
        } else if constexpr (AMatmulType::format == CubeFormat::NZ) {
            if constexpr (aTrans) {
                // m1, k1, k0, m0
                offsetA_ = DequantBmm::Align(mOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM) +
                           DequantBmm::Align(kOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           batchBOffset * DequantBmm::Align(m_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            } else {
                // k1, m1, m0, k0
                offsetA_ = DequantBmm::Align(mOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           DequantBmm::Align(kOffset, K0_INT8) * DequantBmm::Align(m_, BMM_BLOCK_NUM) +
                           batchBOffset * DequantBmm::Align(m_, BMM_BLOCK_NUM) * DequantBmm::Align(k_, K0_INT8);
            }
        }

        if constexpr (BMatmulType::format == CubeFormat::ND) {
            offsetB_ = (bTrans > 0) ? nOffset * k_ + kOffset : nOffset + kOffset * n_;
            offsetB_ = batchBOffset * static_cast<uint64_t>(k_ * n_) + offsetB_;
        } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
            if constexpr (bTrans) {
                // k1, n1, n0, k0
                offsetB_ = DequantBmm::Align(nOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           DequantBmm::Align(kOffset, K0_INT8) * DequantBmm::Align(n_, BMM_BLOCK_NUM) +
                           batchBOffset * DequantBmm::Align(n_, BMM_BLOCK_NUM) * DequantBmm::Align(k_, K0_INT8);
            } else {
                // n1, k1, k0, n0
                offsetB_ = DequantBmm::Align(nOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM) +
                           DequantBmm::Align(kOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           batchBOffset * DequantBmm::Align(n_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            }
        }

        if constexpr (CMatmulType::format == CubeFormat::ND || CMatmulType::format == CubeFormat::ND_ALIGN) {
            offsetC_ = batchCOffset * static_cast<uint64_t>(m_ * n_) + mOffset * n_ + nOffset;
        } else if constexpr (CMatmulType::format == CubeFormat::NZ) {
            offsetC_ = m_ * nOffset + mOffset * BMM_BLOCK_NUM;
            offsetC_ =
                batchCOffset * DequantBmm::Align(n_, BMM_BLOCK_NUM) * DequantBmm::Align(m_, BMM_BLOCK_NUM) + offsetC_;
        }

        offsetBias_ = nOffset;
        offsetScale_ = offsetBias_;
        if (biasThreeDim_) {
            offsetBias_ = batchCOffset * n_ + offsetBias_;
        }
    }

    /** mm_ computation function
     */
    __aicore__ inline void MMCompute(bool enAtomic, uint32_t kCoreIndx)
    {
        if constexpr (!IsSameType<yType, int32_t>::value) {
            if (isPerTensor_) {
                mm_.SetQuantScalar(scaleScalar_);
            } else {
                mm_.SetQuantVector(scaleGm_[offsetScale_]);
            }
        }
        mm_.SetTensorA(xGm_[offsetA_], aTrans);
        mm_.SetTensorB(weightGm_[offsetB_], bTrans);
        if (hasBias_ != 0UL && kCoreIndx == 0UL) {
            mm_.SetBias(biasGm_[offsetBias_]);
        }
        mm_.IterateAll(yGm_[offsetC_], enAtomic);
        mm_.End();
    }
};
} // namespace AscendC

#endif // QUANT_BATCH_MATMUL_V3_H
