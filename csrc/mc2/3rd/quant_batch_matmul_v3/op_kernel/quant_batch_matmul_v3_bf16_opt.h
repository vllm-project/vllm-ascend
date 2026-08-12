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
 * \file quant_batch_matmul_v3_bf16_opt.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_BF16_OPT_H
#define MC2_QUANT_BATCH_MATMUL_V3_BF16_OPT_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {

// 当前mc2不调用该接口，修改接口仍需check
template <typename xType, typename wType, int fFormat, int wFormat, typename scaleType, typename yType, bool aTrans,
          bool bTrans, bool isAllAiv = false>
class BmmDequantBf16Opt {
public:
    __aicore__ inline BmmDequantBf16Opt()
    {
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR y, GM_ADDR workSpace,
                                const Mc2QuantBatchMatmulV3TilingData *__restrict tilingData, TPipe *tPipe)
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

        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ scaleType *)scale);
        }

        // init global buffer
        UpdateGlobalAddr(x1, x2, bias, scale, y, workSpace);

        // init ub local buffer
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        if (biasDtype_ != DT_INT32) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
        }
    }

    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scale, GM_ADDR y,
                                            GM_ADDR workSpace)
    {
        if (blockIdx_ >= usedCoreNum_ || GetSubBlockIdx() > 0) {
            return;
        }
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
            if (biasDtype_ == DT_BF16) {
                biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t *)bias);
            } else if (biasDtype_ == DT_FLOAT16) {
                biasGmFp16_.SetGlobalBuffer((__gm__ half *)bias);
            } else if (biasDtype_ == DT_FLOAT) {
                biasGmFp32_.SetGlobalBuffer((__gm__ float *)bias);
            } else {
                biasGmInt32_.SetGlobalBuffer((__gm__ int32_t *)bias);
            }
        }

        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType *)scale);
        mmOutGm_.SetGlobalBuffer((__gm__ int32_t *)workSpace, batch_ * m_ * n_);
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

        uint32_t mCoreIndx = (blockIdx_ % divideBatchcoreNum) % mDim; // 必须沿着N 轴方向输出
        uint32_t nCoreIndx = (blockIdx_ % divideBatchcoreNum) / mDim;
        uint32_t batchCoreIndx = blockIdx_ / divideBatchcoreNum;

        uint32_t gmUseBatch = batch_ - batchCoreIndx * singleCoreBatch_;
        uint32_t singleCoreBatchUpdate = gmUseBatch < singleCoreBatch_ ? gmUseBatch : singleCoreBatch_;
        uint32_t gmUseM = m_ - mCoreIndx * singleCoreM_;
        uint32_t singleCoreMUpdate = gmUseM < singleCoreM_ ? gmUseM : singleCoreM_;

        uint32_t gmUseN = n_ - nCoreIndx * singleCoreN_;
        uint32_t singleCoreNUpdate = gmUseN < singleCoreN_ ? gmUseN : singleCoreN_;

        mm.SetOrgShape(m_, n_, k_);
        uint32_t mLoops = DequantBmm::CeilDiv(singleCoreMUpdate, singleTimeM_);
        uint32_t nLoops = DequantBmm::CeilDiv(singleCoreNUpdate, singleTimeN_);
        for (uint32_t i = 0; i < singleCoreBatchUpdate; ++i) {
            CalcOffset(i, batchCoreIndx, mCoreIndx, nCoreIndx);
            for (uint32_t j = 0; j < mLoops; ++j) {
                uint32_t singleM = j == mLoops - 1 ? singleCoreMUpdate - (mLoops - 1) * singleTimeM_ : singleTimeM_;
                CalcMAxisOffset(j, nLoops);
                for (uint32_t k = 0; k < nLoops; ++k) {
                    uint32_t singleN = k == nLoops - 1 ? singleCoreNUpdate - (nLoops - 1) * singleTimeN_ : singleTimeN_;
                    CalcNAxisOffset(k);
                    MMDequantCompute(singleM, singleN);
                }
            }
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
    GlobalTensor<int32_t> mmOutGm_;

    TPipe *pipe_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    // used when bias type is bf16/fp16/fp32, deqaunt result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;

    uint32_t curBlockIdx_;
    uint32_t blockIdx_;

    uint32_t hasBias_;
    uint32_t biasDtype_ = 0;
    uint32_t biasDtypeSize_ = 0;
    scaleType scaleScalar_;
    uint64_t offsetA_;
    uint64_t offsetB_;
    uint64_t offsetC_;
    uint64_t offsetBias_;
    uint64_t offsetBatchC_;
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
    uint32_t singleTimeM_;
    uint32_t singleTimeN_;
    uint32_t singleCoreK_;
    uint32_t usedCoreNum_;
    uint32_t baseM_;
    uint32_t baseN_;
    bool isMouter_;

    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    /** init function for TilingData of mm1
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
        singleCoreM_ = tilingData->params.realSingleCoreM;   // calcM of each core
        singleCoreN_ = tilingData->params.realSingleCoreN;   // calcN of each core
        singleTimeM_ = tilingData->matmulTiling.singleCoreM; // calcM of each mm iterate
        singleTimeN_ = tilingData->matmulTiling.singleCoreN; // calcN of each mm iterate
        singleCoreK_ = tilingData->matmulTiling.singleCoreK;
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;

        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;
        isMouter_ = tilingData->matmulTiling.iterateOrder == 0;
        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
    }

    __aicore__ inline void CalcOffset(uint32_t batchIndex, uint32_t batchCoreIndx, uint32_t mCoreIndx,
                                      uint32_t nCoreIndx)
    {
        offsetBatchC_ = batchCoreIndx * (singleCoreBatch_) + batchIndex;
        uint64_t batchAOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchA_);
        uint64_t batchBOffset = static_cast<uint64_t>((batchCoreIndx * (singleCoreBatch_) + batchIndex) % batchB_);
        uint64_t mOffset = static_cast<uint64_t>(mCoreIndx * singleCoreM_);
        uint64_t nOffset = static_cast<uint64_t>(nCoreIndx * singleCoreN_);
        if constexpr (AMatmulType::format == CubeFormat::ND) {
            offsetA_ = (aTrans > 0) ? mOffset : mOffset * k_;
            offsetA_ = batchAOffset * static_cast<uint64_t>(k_ * m_) + offsetA_;
        } else if constexpr (AMatmulType::format == CubeFormat::NZ) {
            if constexpr (aTrans) {
                // m1, k1, k0, m0
                offsetA_ = DequantBmm::Align(k_, BMM_BLOCK_NUM) * DequantBmm::Align(mOffset, K0_INT8) +
                           batchAOffset * DequantBmm::Align(k_, BMM_BLOCK_NUM) * DequantBmm::Align(m_, K0_INT8);
            } else {
                // k1, m1, m0, k0
                offsetA_ = DequantBmm::Align(mOffset, BMM_BLOCK_NUM) * K0_INT8 +
                           batchAOffset * DequantBmm::Align(k_, K0_INT8) * DequantBmm::Align(m_, BMM_BLOCK_NUM);
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

        // the output of mm only support ND/ND_ALIGN for vector
        offsetC_ = mOffset * n_ + nOffset;

        offsetBias_ = nOffset;
        offsetScale_ = offsetBias_;
        if (biasThreeDim_) {
            offsetBias_ = offsetBatchC_ * n_ + offsetBias_;
        }
    }

    __aicore__ inline void CalcMAxisOffset(uint32_t loopIdx, uint32_t nLoops)
    {
        if (loopIdx == 0) {
            return;
        }
        if constexpr (AMatmulType::format == CubeFormat::ND) {
            offsetA_ += (aTrans > 0) ? singleTimeM_ : singleTimeM_ * k_;
        } else if constexpr (AMatmulType::format == CubeFormat::NZ) {
            offsetA_ += (aTrans > 0) ? singleTimeM_ * k_ : singleTimeM_ * K0_INT8;
        }

        // clear n offset and add m offset
        // -----------------
        // |     |     | m |
        // -----------------
        // |m + 1|     |   |
        // -----------------
        uint64_t nOffset = singleTimeN_ * (nLoops - 1);
        offsetC_ = offsetC_ - nOffset + singleTimeM_ * n_;
        if (nLoops > 1) {
            if constexpr (BMatmulType::format == CubeFormat::ND) {
                offsetB_ -= (bTrans > 0) ? nOffset * k_ : nOffset;
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
            offsetB_ += (bTrans > 0) ? singleTimeN_ * k_ : singleTimeN_;
        } else if constexpr (BMatmulType::format == CubeFormat::NZ) {
            if constexpr (bTrans) {
                offsetB_ += DequantBmm::Align(singleTimeN_, BMM_BLOCK_NUM) * K0_INT8;
            } else {
                offsetB_ += DequantBmm::Align(singleTimeN_, K0_INT8) * DequantBmm::Align(k_, BMM_BLOCK_NUM);
            }
        }

        offsetC_ += singleTimeN_;
        offsetBias_ += singleTimeN_;
        offsetScale_ += singleTimeN_;
    }

    /** mm computation function
     */
    __aicore__ inline void MMCompute(uint32_t singleM, uint32_t singleN)
    {
        mm.SetSingleShape(singleM, singleN, singleCoreK_);
        mm.SetTensorA(xGm_[offsetA_], aTrans);
        mm.SetTensorB(weightGm_[offsetB_], bTrans);
        if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
            mm.SetBias(biasGmInt32_[offsetBias_]);
        }
        mm.IterateAll(mmOutGm_[offsetC_], false); // matmultiling singleTimeM_ * singleTimeN_
    }

    __aicore__ inline void BiasGm2Ub(LocalTensor<float> &dstLocalFp32, LocalTensor<float> &biasFp32,
                                     LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16,
                                     LocalTensor<float> &oriBiasFp32, DataCopyPadParams padParams,
                                     uint64_t baseNOfffset, uint32_t curAivN)
    {
        dstLocalFp32 = outFp32Tmp_.Get<float>();
        biasFp32 = biasFp32Tmp_.Get<float>();
        DataCopyParams bias2UbParams{1, 0, 0, 0};
        bias2UbParams.blockLen = curAivN * biasDtypeSize_;
        uint64_t biasOffset = offsetBias_ + baseNOfffset;

        if (biasDtype_ == DT_BF16) {
            oriBiasBf16 = vecQueBias_.AllocTensor<bfloat16_t>(); // free in CalBiasAdd
            DataCopyPad(oriBiasBf16, biasGmBf16_[biasOffset], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT16) {
            oriBiasFp16 = vecQueBias_.AllocTensor<half>(); // free in CalBiasAdd
            DataCopyPad(oriBiasFp16, biasGmFp16_[biasOffset], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT) {
            oriBiasFp32 = vecQueBias_.AllocTensor<float>(); // free in CalBiasAdd
            DataCopyPad(oriBiasFp32, biasGmFp32_[biasOffset], bias2UbParams, padParams);
        }
    }

    __aicore__ inline void Bf16ScaleGm2Ub(LocalTensor<scaleType> &scaleLocal, GlobalTensor<scaleType> &scaleGm_,
                                          DataCopyPadParams padParams, uint64_t baseNOfffset, uint32_t curAivN)
    {
        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivN * sizeof(scaleType);
        uint64_t scaleOffset = offsetScale_ + baseNOfffset;
        DataCopyPad(scaleLocal, scaleGm_[scaleOffset], scale2UbParams, padParams);
    }

    __aicore__ inline void CalBiasAdd(LocalTensor<float> &dstLocalFp32, LocalTensor<float> &biasFp32,
                                      LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16,
                                      LocalTensor<float> &oriBiasFp32, LocalTensor<yType> &dstLocal, uint32_t curAivN,
                                      uint32_t curAivM)
    {
        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U); // 8: 32B aligned for int32_t
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN);    // 16: sizeof(yType) is 2, 32B / 2
        AscendC::PipeBarrier<PIPE_V>();
        if (biasDtype_ == DT_BF16) {
            Cast(biasFp32, oriBiasBf16, RoundMode::CAST_NONE, ubResAlignedN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasBf16);
        } else if (biasDtype_ == DT_FLOAT16) {
            Cast(biasFp32, oriBiasFp16, RoundMode::CAST_NONE, ubResAlignedN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp16);
        } else if (biasDtype_ == DT_FLOAT) {
            biasFp32 = oriBiasFp32;
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp32);
        }

        LocalTensor<float> dstLocalFp32Pad;
        if (computedAivN != ubResAlignedN) {
            dstLocalFp32Pad = vecQueTmp_.Get<float>();
            for (uint32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
                Add(dstLocalFp32Pad[mIdx * ubResAlignedN], dstLocalFp32[mIdx * computedAivN], biasFp32, computedAivN);
            }
        } else {
            for (uint32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
                Add(dstLocalFp32[mIdx * ubResAlignedN], dstLocalFp32[mIdx * ubResAlignedN], biasFp32, ubResAlignedN);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (computedAivN != ubResAlignedN) {
            Cast(dstLocal, dstLocalFp32Pad, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        } else {
            Cast(dstLocal, dstLocalFp32, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyGm2Ub(LocalTensor<int32_t> &srcLocal, GlobalTensor<int32_t> &curMmOutGm,
                                         DataCopyParams &gm2UbParams, DataCopyPadParams &padParams,
                                         uint64_t curAicOffset)
    {
        if (gm2UbParams.srcStride > 65535) { // instr stride only support uint16 which largest is 65535
            uint64_t curAivOffset = 0;
            uint32_t mLoop = gm2UbParams.blockCount;
            gm2UbParams.blockCount = 1;
            gm2UbParams.srcStride = 0;
            for (uint32_t mIdx = 0; mIdx < mLoop; mIdx++) {
                DataCopyPad(srcLocal[curAivOffset], curMmOutGm[curAicOffset], gm2UbParams, padParams);
                curAicOffset += n_;
                curAivOffset += gm2UbParams.blockLen / sizeof(int32_t);
            }
            return;
        }
        DataCopyPad(srcLocal, curMmOutGm[curAicOffset], gm2UbParams, padParams);
    }

    __aicore__ inline void DequantCompute(GlobalTensor<int32_t> &curMmOutGm, uint64_t baseMOfffset,
                                          uint64_t baseNOfffset, uint32_t curAicM, uint32_t curAicN)
    {
        LocalTensor<float> dstLocalFp32;
        LocalTensor<float> biasFp32;
        LocalTensor<bfloat16_t> oriBiasBf16;
        LocalTensor<half> oriBiasFp16;
        LocalTensor<float> oriBiasFp32;
        uint32_t curAivM = ubCalcM_;
        // calcN in ub is equal to aicN
        uint32_t curAivN = curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(curAicM, ubCalcM_);
        DataCopyParams gm2UbParams{1, 0, 0, 0};
        DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        DataCopyPadParams padParams;
        DequantParams dequantParams;
        DequantBmm::CalcDequantParams(mUbLoops == 1 ? curAicM : ubCalcM_, curAicN, dequantParams);
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = curAicM - ubCalcM_ * (mUbLoops - 1);
                DequantBmm::CalcDequantParams(curAivM, curAicN, dequantParams, mUbLoops != 1 && curAivM != ubCalcM_);
            }
            LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
            LocalTensor<yType> dstLocal = vecQueOut_.AllocTensor<yType>();
            LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
            // datacopypad 32B aligned
            gm2UbParams.blockLen = curAivN * sizeof(int32_t);
            gm2UbParams.blockCount = curAivM;
            gm2UbParams.srcStride = (n_ - curAivN) * sizeof(int32_t);
            uint64_t curAicAivOffset = offsetC_ + baseMOfffset + baseNOfffset + mUbLoopIdx * ubCalcM_ * n_;
            DataCopyGm2Ub(srcLocal, curMmOutGm, gm2UbParams, padParams, curAicAivOffset);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (biasDtype_ != DT_INT32) {
                BiasGm2Ub(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams, baseNOfffset,
                          curAicN);
            }
            if (isPerTensor_) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (biasDtype_ != DT_INT32) {
                    AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
                } else {
                    AscendDequant(dstLocal, srcLocal, scaleScalar_, tmpLocal, dequantParams);
                }
            } else {
                LocalTensor<scaleType> scaleLocal = vecQueScale_.AllocTensor<scaleType>();
                Bf16ScaleGm2Ub(scaleLocal, scaleGm_, padParams, baseNOfffset, curAicN);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (biasDtype_ != DT_INT32) {
                    AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
                } else {
                    AscendDequant(dstLocal, srcLocal, scaleLocal, tmpLocal, dequantParams);
                }
                vecQueScale_.FreeTensor(scaleLocal);
            }
            if (biasDtype_ != DT_INT32) {
                CalBiasAdd(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, dstLocal, curAivN, curAivM);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            vecQueSrc_.FreeTensor(srcLocal);
            // dst from ub -> gm
            ub2GmParams.blockLen = curAivN * sizeof(yType);
            ub2GmParams.blockCount = curAivM;
            ub2GmParams.dstStride = (n_ - curAivN) * sizeof(yType);
            uint64_t aivOffset = mUbLoopIdx * ubCalcM_ * n_;
            uint64_t gmOffset = offsetBatchC_ * m_ * n_ + offsetC_ + baseMOfffset + baseNOfffset + aivOffset;
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopyPad(yGm_[gmOffset], dstLocal, ub2GmParams);
            vecQueOut_.FreeTensor(dstLocal);
        }
    }

    __aicore__ inline void DequantMOuterSplit(uint32_t singleM, uint32_t singleN, uint32_t fixpMtimes,
                                              uint32_t fixpNtimes)
    {
        uint32_t curAicOuter = baseM_;
        uint32_t curAicInner = baseN_;
        for (uint32_t fixpOuterIdx = 0; fixpOuterIdx < fixpMtimes; ++fixpOuterIdx) {
            if (fixpOuterIdx == fixpMtimes - 1) {
                curAicOuter = singleM - baseM_ * (fixpMtimes - 1);
            }
            for (uint32_t fixpInnerIdx = 0; fixpInnerIdx < fixpNtimes; ++fixpInnerIdx) {
                if (fixpInnerIdx == fixpNtimes - 1) {
                    curAicInner = singleN - baseN_ * (fixpNtimes - 1);
                }
                DequantCompute(mmOutGm_, static_cast<uint64_t>(fixpOuterIdx) * baseM_ * n_, fixpInnerIdx * baseN_,
                               curAicOuter, curAicInner);
            }
        }
    }

    __aicore__ inline void DequantNOuterSplit(uint32_t singleM, uint32_t singleN, uint32_t fixpMtimes,
                                              uint32_t fixpNtimes)
    {
        uint32_t curAicOuter = baseN_;
        uint32_t curAicInner = baseM_;
        for (uint32_t fixpOuterIdx = 0; fixpOuterIdx < fixpNtimes; ++fixpOuterIdx) {
            if (fixpOuterIdx == fixpNtimes - 1) {
                curAicOuter = singleN - baseN_ * (fixpNtimes - 1);
            }
            for (uint32_t fixpInnerIdx = 0; fixpInnerIdx < fixpMtimes; ++fixpInnerIdx) {
                if (fixpInnerIdx == fixpMtimes - 1) {
                    curAicInner = singleM - baseM_ * (fixpMtimes - 1);
                }
                DequantCompute(mmOutGm_, static_cast<uint64_t>(fixpInnerIdx) * baseM_ * n_, fixpOuterIdx * baseN_,
                               curAicInner, curAicOuter);
            }
        }
    }

    __aicore__ inline void MMDequantCompute(uint32_t singleM, uint32_t singleN)
    {
        if ASCEND_IS_AIC {
            MMCompute(singleM, singleN);
            // set cv sync id 4
            NotifyEvent<PIPE_FIX>(0x4);
            mm.End();
        }

        if ASCEND_IS_AIV {
            // set cv sync id 4
            WaitEvent(0x4);
            uint32_t fixpMtimes = DequantBmm::CeilDiv(singleM, baseM_);
            uint32_t fixpNTimes = DequantBmm::CeilDiv(singleN, baseN_);
            // iterateOrder: 0: M-N, 1: N-M
            if (isMouter_) {
                DequantMOuterSplit(singleM, singleN, fixpMtimes, fixpNTimes);
            } else {
                DequantNOuterSplit(singleM, singleN, fixpMtimes, fixpNTimes);
            }
        }
    }
};
} // namespace AscendC

#endif // QUANT_BATCH_MATMUL_V3_BF16_OPT_H
