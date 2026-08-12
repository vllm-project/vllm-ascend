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
 * \file quant_batch_matmul_v3_pertoken_opt.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_PERTOKEN_OPT_H
#define MC2_QUANT_BATCH_MATMUL_V3_PERTOKEN_OPT_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {

// 当前mc2不调用该接口，修改接口仍需check
template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType, bool aTrans,
          bool bTrans, bool isAllAiv = false>
class BmmDequantPertokenOpt {
public:
    __aicore__ inline BmmDequantPertokenOpt()
    {
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR y,
                                GM_ADDR workSpace, const Mc2QuantBatchMatmulV3TilingData *__restrict tilingData,
                                TPipe *tPipe)
    {
        blockIdx_ = GetBlockIdx();
        if constexpr (isAllAiv) {
            blockIdx_ /= GetTaskRation();
            if (GetSubBlockIdx() > 0) {
                return;
            }
        }
        InitTilingData(tilingData);
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        mm.SetSubBlockIdx(0);
        mm.Init(&(tilingData->matmulTiling), tPipe);
        pipe_ = tPipe;

        // init global buffer
        UpdateGlobalAddr(x1, x2, bias, scale, pertokenScale, y, workSpace);

        // init ub local buffer
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        // int32->fp32
        pipe_->InitBuffer(vecFp32Src_, ubCalcM_ * ubCalcN_ * sizeof(float));
        if (biasDtype_ != DT_INT32) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
            if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                pipe_->InitBuffer(vecFp32Scale_, ubCalcN_ * sizeof(float));
            }
        }
        // pertoken
        pipe_->InitBuffer(vecQuePertokenScale_, 1, DequantBmm::Align(baseM_, 8U) * sizeof(float));
        pipe_->InitBuffer(broadcastFp32Tmp_, baseM_ * ubCalcN_ * sizeof(float));
        pertokenScaleLocal_ = vecQuePertokenScale_.AllocTensor<float>();
    }

    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR pertokenScale,
                                            GM_ADDR y, GM_ADDR workSpace)
    {
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0) {
            return;
        }
        // update global buffer
        xGm_.SetGlobalBuffer((__gm__ xType *)x1);
        weightGm_.SetGlobalBuffer((__gm__ wType *)x2);
        if (m_ <= baseM_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (biasDtype_ == DT_BF16) {
            biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t *)bias);
        } else if (biasDtype_ == DT_FLOAT16) {
            biasGmFp16_.SetGlobalBuffer((__gm__ half *)bias);
        } else if (biasDtype_ == DT_FLOAT) {
            biasGmFp32_.SetGlobalBuffer((__gm__ float *)bias);
        } else if (hasBias_ != 0U) {
            biasGmInt32_.SetGlobalBuffer((__gm__ int32_t *)bias);
        }
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType *)scale);
        if (isPerTensor_) {
            if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                scaleScalar_ = ToFloat(scaleGm_.GetValue(0));
            } else {
                scaleScalar_ = scaleGm_.GetValue(0);
            }
        }
        pertokenScaleGm_.SetGlobalBuffer((__gm__ float *)pertokenScale);
        uint32_t singleCoreOffset = baseM_ * DequantBmm::Align(singleCoreN_, baseN_);
        mmOutGm_.SetGlobalBuffer((__gm__ int32_t *)workSpace, usedCoreNum_ * singleCoreOffset);
    }

    /** main logical function
     */
    __aicore__ inline void Process()
    {
        uint32_t batchDim = DequantBmm::CeilDiv(batch_, singleCoreBatch_);
        uint32_t mDim = DequantBmm::CeilDiv(m_, singleCoreM_);
        uint32_t nDim = DequantBmm::CeilDiv(n_, singleCoreN_);
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0) {
            return;
        }

        uint32_t divideBatchcoreNum = usedCoreNum_ / batchDim;

        uint32_t mCoreIndx = (blockIdx_ % divideBatchcoreNum) % mDim;
        uint32_t nCoreIndx = (blockIdx_ % divideBatchcoreNum) / mDim;
        uint32_t batchCoreIndx = blockIdx_ / divideBatchcoreNum;

        uint32_t gmUseBatch = batch_ - batchCoreIndx * singleCoreBatch_;
        uint32_t singleCoreBatchUpdate = gmUseBatch < singleCoreBatch_ ? gmUseBatch : singleCoreBatch_;
        uint32_t gmUseM = m_ - mCoreIndx * singleCoreM_;
        uint32_t singleCoreMUpdate = gmUseM < singleCoreM_ ? gmUseM : singleCoreM_;
        uint32_t gmUseN = n_ - nCoreIndx * singleCoreN_;
        uint32_t singleCoreNUpdate = gmUseN < singleCoreN_ ? gmUseN : singleCoreN_;

        uint32_t mLoops = DequantBmm::CeilDiv(singleCoreMUpdate, baseM_);
        uint32_t nLoops = DequantBmm::CeilDiv(singleCoreNUpdate, baseN_);
        for (uint32_t i = 0; i < singleCoreBatchUpdate; ++i) {
            // 不同batch复用相同workspace，需设置c等v的搬运同步，避免workspace的读写冲突
            CalcOffset(i, batchCoreIndx, mCoreIndx, nCoreIndx);
            for (uint32_t j = 0; j < mLoops; ++j) {
                uint32_t singleM = j == mLoops - 1 ? singleCoreMUpdate - (mLoops - 1) * baseM_ : baseM_;
                CalcMAxisOffset(j, nLoops);
                if ASCEND_IS_AIV {
                    PertokenGm2Ub(singleM);
                }
                // 不同batch复用相同workspace，需设置c等v的搬运同步，避免workspace的读写冲突
                bool isNotFirst = (i != 0) || (j != 0);
                bool isNotLast = (j != mLoops - 1) || (i != singleCoreBatchUpdate - 1);

                MMDequantCompute(singleM, singleCoreNUpdate, isNotFirst, isNotLast);
            }
        }
        if ASCEND_IS_AIC {
            mm.End();
        }
        if ASCEND_IS_AIV {
            vecQuePertokenScale_.FreeTensor(pertokenScaleLocal_);
        }
    }

    // define matmul
    static constexpr CubeFormat aFormat = DequantBmm::GetFormat(fFormat);
    static constexpr CubeFormat bFormat = DequantBmm::GetFormat(wFormat);

    using AMatmulType = matmul::MatmulType<TPosition::GM, aFormat, xType, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, bFormat, wType, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the tpos of ctype must be ub given by mm api when iterate<false>, but actually we can move data to gm
    // then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm;

protected:
    GlobalTensor<xType> xGm_;
    GlobalTensor<wType> weightGm_;
    GlobalTensor<int32_t> biasGmInt32_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<half> biasGmFp16_;
    GlobalTensor<float> biasGmFp32_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<scaleType> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<int32_t> mmOutGm_;

    TPipe *pipe_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    TBuf<TPosition::VECCALC> vecFp32Src_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;
    TBuf<TPosition::VECCALC> vecFp32Scale_;
    // used when bias type is bf16/fp16/fp32, deqaunt result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;

    LocalTensor<float> pertokenScaleLocal_;

    uint32_t curBlockIdx_;
    uint32_t blockIdx_;

    uint32_t hasBias_;
    uint32_t biasDtype_ = 0;
    uint32_t biasDtypeSize_ = 0;
    float scaleScalar_;
    uint64_t offsetA_;
    uint64_t offsetB_;
    uint64_t offsetC_; // mm out: workspace
    uint64_t offsetY_; // op final out(y): ddr
    uint64_t offsetBias_;
    uint64_t offsetScale_;
    uint64_t offsetPertokenScale_;

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
    uint32_t baseN_;

    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    /** init function for TilingData of mm
     */
    __aicore__ inline void InitTilingData(const Mc2QuantBatchMatmulV3TilingData *tilingData)
    {
        hasBias_ = tilingData->matmulTiling.isBias;
        isPerTensor_ = tilingData->params.isPerTensor;
        biasDtype_ = tilingData->params.biasDtype;
        if (biasDtype_ == DT_INT32 || biasDtype_ == DT_FLOAT) {
            biasDtypeSize_ = sizeof(int32_t);
        } else {
            biasDtypeSize_ = sizeof(half);
        }
        batchA_ = tilingData->params.batchA;
        batchB_ = tilingData->params.batchB;
        batch_ = tilingData->params.batchC;
        singleCoreBatch_ = tilingData->params.singleCoreBatch;
        biasThreeDim_ = tilingData->params.biasThreeDim;
        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        k_ = tilingData->matmulTiling.Ka;
        singleCoreM_ = tilingData->params.realSingleCoreM; // calcM of each core
        singleCoreN_ = tilingData->params.realSingleCoreN; // calcN of each core
        singleCoreK_ = tilingData->matmulTiling.singleCoreK;
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;

        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;
        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
    }

    __aicore__ inline void CalcOffset(uint32_t batchIndex, uint32_t batchCoreIndx, uint32_t mCoreIndx,
                                      uint32_t nCoreIndx)
    {
        uint64_t batchAOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchA_);
        uint64_t batchBOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchB_);
        uint64_t batchCOffset = static_cast<uint64_t>(batchCoreIndx * (singleCoreBatch_) + batchIndex);
        uint64_t nOffset = static_cast<uint64_t>(nCoreIndx * singleCoreN_);
        uint64_t mOffset = static_cast<uint64_t>(mCoreIndx * singleCoreM_);
        if constexpr (AMatmulType::format == CubeFormat::ND) {
            if constexpr (aTrans) {
                offsetA_ = batchAOffset * static_cast<uint64_t>(k_ * m_) + mOffset;
            } else {
                offsetA_ = batchAOffset * static_cast<uint64_t>(k_ * m_) + mOffset * k_;
            }
        } else if constexpr (AMatmulType::format == CubeFormat::NZ) {
            if constexpr (aTrans) {
                // m1, k1, k0, m0
                offsetA_ = batchBOffset * DequantBmm::Align(k_, BMM_BLOCK_NUM) * DequantBmm::Align(m_, K0_INT8) +
                           DequantBmm::Align(mOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            } else {
                // k1, m1, m0, k0
                offsetA_ = batchAOffset * DequantBmm::Align(m_, BMM_BLOCK_NUM) * DequantBmm::Align(k_, K0_INT8) +
                           DequantBmm::Align(mOffset, BMM_BLOCK_NUM) * K0_INT8;
            }
        }

        if constexpr (BMatmulType::format == CubeFormat::ND) {
            offsetB_ = (bTrans > 0) ? nOffset * k_ : nOffset;
            offsetB_ = batchBOffset * static_cast<uint64_t>(k_ * n_) + offsetB_;
        } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
            if constexpr (bTrans) {
                // k1, n1, n0, k0
                offsetB_ = DequantBmm::Align(nOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           batchBOffset * DequantBmm::Align(n_, BMM_BLOCK_NUM) * DequantBmm::Align(k_, K0_INT8);
            } else {
                // n1, k1, k0, n0
                offsetB_ = DequantBmm::Align(nOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM) +
                           batchBOffset * DequantBmm::Align(n_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            }
        }

        // mm out按照base块连续写到workspace中，每个核上可能多个base块
        offsetC_ = blockIdx_ * baseM_ * DequantBmm::Align(singleCoreN_, baseN_);
        // the output of mm only support ND/ND_ALIGN for vector
        offsetY_ = batchCOffset * (m_ * n_) + mOffset * n_ + nOffset;

        offsetBias_ = nOffset;
        offsetScale_ = offsetBias_;
        offsetPertokenScale_ = mOffset;
        if (biasThreeDim_) {
            offsetBias_ = batchCOffset * n_ + offsetBias_;
        }
    }

    __aicore__ inline void CalcMAxisOffset(uint32_t loopIdx, uint32_t nLoops)
    {
        if (loopIdx == 0) {
            return;
        }
        if constexpr (AMatmulType::format == CubeFormat::ND) {
            if constexpr (aTrans > 0) {
                offsetA_ += baseM_;
            } else {
                offsetA_ += baseM_ * k_;
            }
        } else if constexpr (AMatmulType::format == CubeFormat::NZ) {
            if constexpr (aTrans > 0) {
                offsetA_ += baseM_ * k_;
            } else {
                offsetA_ += baseM_ * K0_INT8;
            }
        }
        offsetPertokenScale_ += baseM_;

        // clear n offset and add m offset
        // -----------------
        // |     |     | m |
        // -----------------
        // |m + 1|     |   |
        // -----------------
        uint64_t nOffset = baseN_ * (nLoops - 1);
        offsetY_ = offsetY_ - nOffset + baseM_ * n_;
        if (nLoops > 1) {
            if constexpr (BMatmulType::format == CubeFormat::ND) {
                if constexpr (bTrans > 0) {
                    offsetB_ -= nOffset * k_;
                } else {
                    offsetB_ -= nOffset;
                }
            } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
                if constexpr (bTrans) {
                    offsetB_ -= DequantBmm::Align(nOffset, BMM_BLOCK_NUM) * K0_INT8;
                } else {
                    offsetB_ -= DequantBmm::Align(nOffset, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
                }
            }
            offsetBias_ -= nOffset;
            offsetScale_ -= nOffset;
        }
    }

    __aicore__ inline void CalcNAxisOffset(uint32_t loopIdx)
    {
        if (loopIdx == 0) {
            return;
        }
        if constexpr (BMatmulType::format == CubeFormat::ND) {
            offsetB_ += (bTrans > 0) ? baseN_ * k_ : baseN_;
        } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
            if constexpr (bTrans) {
                offsetB_ += DequantBmm::Align(baseN_, BMM_BLOCK_NUM) * K0_INT8;
            } else {
                offsetB_ += DequantBmm::Align(baseN_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            }
        }

        offsetC_ += baseM_ * baseN_;
        offsetY_ += baseN_;
        offsetBias_ += baseN_;
        offsetScale_ += baseN_;
    }

    /** mm computation function
     */
    __aicore__ inline void MMCompute(uint32_t singleM, uint32_t singleN)
    {
        mm.SetSingleShape(singleM, singleN, singleCoreK_);
        mm.SetTensorB(weightGm_[offsetB_], bTrans);
        if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
            mm.SetBias(biasGmInt32_[offsetBias_]);
        }
        mm.Iterate();                               // matmultiling singleM_ * baseN_
        mm.GetTensorC(mmOutGm_[offsetC_], 0, true); // baseM * baseN 连续写
    }

    __aicore__ inline void PertokenGm2Ub(uint32_t singleM)
    {
        DataCopyParams pertoken2UbParams{1, static_cast<uint16_t>(singleM * sizeof(float)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(pertokenScaleLocal_, pertokenScaleGm_[offsetPertokenScale_], pertoken2UbParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }

    __aicore__ inline void ScaleGm2Ub(uint32_t curAivN)
    {
        LocalTensor<scaleType> scaleLocal = vecQueScale_.AllocTensor<scaleType>();
        DataCopyParams scale2UbParams{1, static_cast<uint16_t>(curAivN * sizeof(scaleType)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(scaleLocal, scaleGm_[offsetScale_], scale2UbParams, padParams);
        vecQueScale_.EnQue<scaleType>(scaleLocal);
    }

    __aicore__ inline void BiasGm2Ub(uint32_t curAivN)
    {
        DataCopyParams bias2UbParams{1, static_cast<uint16_t>(curAivN * biasDtypeSize_), 0, 0};
        DataCopyPadParams padParams;
        if (biasDtype_ == DT_BF16) {
            LocalTensor<bfloat16_t> biasBf16 = vecQueBias_.AllocTensor<bfloat16_t>();
            DataCopyPad(biasBf16, biasGmBf16_[offsetBias_], bias2UbParams, padParams);
            vecQueBias_.EnQue<bfloat16_t>(biasBf16);
        } else if (biasDtype_ == DT_FLOAT16) {
            LocalTensor<half> biasFp16 = vecQueBias_.AllocTensor<half>();
            DataCopyPad(biasFp16, biasGmFp16_[offsetBias_], bias2UbParams, padParams);
            vecQueBias_.EnQue<half>(biasFp16);
        } else if (biasDtype_ == DT_FLOAT) {
            LocalTensor<float> biasFp32 = vecQueBias_.AllocTensor<float>();
            DataCopyPad(biasFp32, biasGmFp32_[offsetBias_], bias2UbParams, padParams);
            vecQueBias_.EnQue<float>(biasFp32);
        }
    }

    __aicore__ inline void CastBias2Fp32(LocalTensor<float> &biasFp32, uint32_t computedAivN)
    {
        if (biasDtype_ == DT_INT32) {
            return;
        }
        biasFp32 = biasFp32Tmp_.Get<float>();
        if (biasDtype_ == DT_BF16) {
            LocalTensor<bfloat16_t> oriBiasBf16 = vecQueBias_.DeQue<bfloat16_t>();
            Cast(biasFp32, oriBiasBf16, RoundMode::CAST_NONE, computedAivN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasBf16);
        } else if (biasDtype_ == DT_FLOAT16) {
            LocalTensor<half> oriBiasFp16 = vecQueBias_.DeQue<half>();
            Cast(biasFp32, oriBiasFp16, RoundMode::CAST_NONE, computedAivN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp16);
        } else if (biasDtype_ == DT_FLOAT) {
            LocalTensor<float> oriBiasFp32 = vecQueBias_.DeQue<float>();
            biasFp32 = oriBiasFp32;
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp32);
        }
    }

    __aicore__ inline void CalBiasAdd(const LocalTensor<float> &dstLocalFp32, LocalTensor<float> &biasFp32,
                                      uint32_t curAivM, uint32_t computedAivN)
    {
        for (uint32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
            Add(dstLocalFp32[mIdx * computedAivN], dstLocalFp32[mIdx * computedAivN], biasFp32, computedAivN);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MulPertokenAndScale(LocalTensor<float> &broadcastFp32, LocalTensor<float> &scaleLocalFp32,
                                               const uint32_t (&broadCastDst)[2], uint32_t calCount)
    {
        uint32_t singleM = broadCastDst[0];
        uint16_t computedAivN = broadCastDst[1];
        for (uint32_t mIdx = 0; mIdx < singleM; ++mIdx) {
            Mul(broadcastFp32[computedAivN * mIdx], broadcastFp32[computedAivN * mIdx], scaleLocalFp32, calCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadCastMN(LocalTensor<float> &broadcastFp32, uint32_t singleM, uint32_t computedAivN)
    {
        const uint32_t broadCastDst[M_N_TWO_DIMS] = {singleM, computedAivN};
        const uint32_t broadCastSrc[M_N_TWO_DIMS] = {singleM, 1};
        BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, pertokenScaleLocal_, broadCastDst, broadCastSrc);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PreUbProcess(LocalTensor<float> &broadcastFp32, uint32_t singleM, uint32_t curAicN)
    {
        // 提前做好m,n方向的scale相乘矩阵，这部分cv并行,按行组成m * baseN的fp32矩阵, 最大L0C Size
        // 全部弄到shape 16对齐
        uint32_t computedAivN = DequantBmm::Align(curAicN);
        const uint32_t broadCastDims[M_N_TWO_DIMS] = {singleM, computedAivN};
        // 清空broadcastFp32，只pertoken
        BroadCastMN(broadcastFp32, singleM, computedAivN);

        if (isPerTensor_) {
            Muls(broadcastFp32, broadcastFp32, scaleScalar_, singleM * computedAivN);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // 搬运该base块所需n scale
            ScaleGm2Ub(curAicN);
            LocalTensor<scaleType> scaleLocal = vecQueScale_.DeQue<scaleType>();

            if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                LocalTensor<float> scaleLocalFp32 = vecFp32Scale_.Get<float>();
                Cast(scaleLocalFp32, scaleLocal, RoundMode::CAST_NONE, computedAivN);
                AscendC::PipeBarrier<PIPE_V>();
                MulPertokenAndScale(broadcastFp32, scaleLocalFp32, broadCastDims, DequantBmm::Align(curAicN));
            } else {
                MulPertokenAndScale(broadcastFp32, scaleLocal, broadCastDims,
                                    DequantBmm::Align(curAicN, 8U)); // 8: 32 / sizeof(float)
            }
            vecQueScale_.FreeTensor(scaleLocal);
        }
        if (biasDtype_ != DT_INT32) {
            BiasGm2Ub(curAicN);
        }
    }

    __aicore__ inline void PostUbProcess(LocalTensor<float> &resFp32, uint32_t curAicM, uint32_t curAicN)
    {
        LocalTensor<float> biasFp32;
        uint32_t curAivM = ubCalcM_;
        // calcN in ub is equal to aicN
        uint32_t curAivN = curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(curAicM, ubCalcM_);
        // blockCount, blockLen, srcStride, dstStride
        DataCopyParams gm2UbParams{static_cast<uint16_t>(curAivM), static_cast<uint16_t>(curAivN * sizeof(int32_t)), 0,
                                   0};
        DataCopyExtParams ub2GmParams{static_cast<uint16_t>(curAivM), static_cast<uint32_t>(curAivN * sizeof(yType)), 0,
                                      static_cast<uint32_t>((n_ - curAivN) * sizeof(yType)), 0};
        DataCopyPadParams padParams;
        uint32_t computedAivN = DequantBmm::Align(curAivN); // 16: sizeof(yType) is 2, 32B / 2
        if (computedAivN != DequantBmm::Align(curAivN, 8U)) {
            gm2UbParams.dstStride = 1; // dst ub, 1 dataBlock(32B), shape 16 aligned
        }
        // 后续可以把biasFp32搞成类成员变量，则可以把cast提到前处理中
        CastBias2Fp32(biasFp32, computedAivN);

        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx != 0 && mUbLoopIdx == mUbLoops - 1) {
                curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
                gm2UbParams.blockCount = curAivM;
                ub2GmParams.blockCount = curAivM;
            }
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            LocalTensor<float> srcFp32 = vecFp32Src_.AllocTensor<float>();
            LocalTensor<yType> dstLocal = vecQueOut_.AllocTensor<yType>();

            uint64_t curAicAivOffset = offsetC_ + mUbLoopIdx * ubCalcM_ * curAivN;
            DataCopyPad(srcLocal, mmOutGm_[curAicAivOffset], gm2UbParams, padParams);
            vecQueSrc_.EnQue<int32_t>(srcLocal);
            srcLocal = vecQueSrc_.DeQue<int32_t>();
            // int32 srcLocal -> fp32 srcFp32
            Cast(srcFp32, srcLocal, RoundMode::CAST_RINT, curAivM * computedAivN);
            AscendC::PipeBarrier<PIPE_V>();
            // 只能取部分相乘好的scale
            uint32_t curScaleOffset = mUbLoopIdx * ubCalcM_ * computedAivN;
            LocalTensor curResFp32 = resFp32[curScaleOffset];
            Mul(curResFp32, curResFp32, srcFp32, curAivM * computedAivN);
            AscendC::PipeBarrier<PIPE_V>();
            if (biasDtype_ != DT_INT32) {
                CalBiasAdd(curResFp32, biasFp32, curAivM, computedAivN);
            }

            Cast(dstLocal, curResFp32, RoundMode::CAST_RINT, curAivM * computedAivN);
            vecQueOut_.EnQue<yType>(dstLocal);
            vecQueSrc_.FreeTensor(srcLocal);
            // dst from ub -> gm
            uint64_t yOffset = offsetY_ + mUbLoopIdx * ubCalcM_ * n_;
            dstLocal = vecQueOut_.DeQue<yType>();
            DataCopyPad(yGm_[yOffset], dstLocal, ub2GmParams);
            vecQueOut_.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void MMDequantCompute(uint32_t singleM, uint32_t singleN, bool isNotFirstBatch,
                                            bool isNotLastBatch)
    {
        // 增量场景 m < baseM, fixpMtimes = 1，stepM/N = 1, 去掉for m循环减少scalar
        if ASCEND_IS_AIC {
            mm.SetTensorA(xGm_[offsetA_], aTrans);
        }
        uint32_t fixpNTimes = DequantBmm::CeilDiv(singleN, baseN_);
        uint32_t curAicN = baseN_;
        // 硬同步计数器取值范围[0,
        // 15]，如果K比较小cube计算比较快，连续发NotifyEvent，计数器可能溢出会挂死，所以要拆个15的循环
        uint32_t fixpNInnerTimes = 15;
        uint32_t fixpNOuterTimes = DequantBmm::CeilDiv(fixpNTimes, fixpNInnerTimes);
        uint32_t realFixpInnerIdx = 0;
        for (uint32_t fixpOuterIdx = 0; fixpOuterIdx < fixpNOuterTimes; ++fixpOuterIdx) {
            if ASCEND_IS_AIC {
                if (isNotFirstBatch || fixpOuterIdx > 0) {
                    // set cv sync id 5: v -> c
                    WaitEvent(0x5);
                }
            }

            for (uint32_t fixpInnerIdx = 0; fixpInnerIdx < fixpNInnerTimes; ++fixpInnerIdx) {
                if (realFixpInnerIdx == fixpNTimes - 1) {
                    curAicN = singleN - baseN_ * (fixpNTimes - 1);
                    fixpNInnerTimes = fixpInnerIdx; // 结束循环
                }
                CalcNAxisOffset(realFixpInnerIdx);
                if ASCEND_IS_AIC {
                    MMCompute(singleM, curAicN);
                    // set cv sync id 4: c -> v
                    NotifyEvent<PIPE_FIX>(0x4);
                }

                if ASCEND_IS_AIV {
                    LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
                    PreUbProcess(broadcastFp32, singleM, curAicN);

                    // set cv sync id 4: c -> v
                    WaitEvent(0x4);
                    PostUbProcess(broadcastFp32, singleM, curAicN);
                }

                ++realFixpInnerIdx;
            }

            if ASCEND_IS_AIV {
                if (isNotLastBatch || fixpOuterIdx != fixpNOuterTimes - 1) {
                    // set cv sync id 5: v -> c
                    NotifyEvent<PIPE_MTE2>(0x5);
                }
            }
        }
    }
};
} // namespace AscendC

#endif // QUANT_BATCH_MATMUL_V3_PERTOKEN_OPT_H
