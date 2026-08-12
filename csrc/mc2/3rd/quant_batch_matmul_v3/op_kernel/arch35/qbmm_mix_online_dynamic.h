/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file qbmm_mix_online_dynamic.h
 * \brief
 */
#ifndef MC2_QBMM_MIX_ONLINE_DYNAMIC_H
#define MC2_QBMM_MIX_ONLINE_DYNAMIC_H

#include "mm_extension_interface/qbmm_custom_mm_policy.h"
#include "qbmm_asw_block.h"
#include "../quant_batch_matmul_v3_base.h"

#define LOCAL_TEMPLATE_CLASS_MIX_PARAMS                                                                                \
    template <class aType, class bType, class scaleType, class biasType, class ptScaleType, class cType,               \
              CubeFormat aFormat, CubeFormat bFormat, CubeFormat cFormat, bool aTrans, bool bTrans, class l0cDtype,    \
              class blockType, const MatmulConfig &mmCfg>
#define LOCAL_TEMPLATE_FUNC_MIX_PARAMS                                                                                 \
    aType, bType, scaleType, biasType, ptScaleType, cType, aFormat, bFormat, cFormat, aTrans, bTrans, l0cDtype,        \
        blockType, mmCfg

namespace Mc2QuantBatchMatmulV3 {
using AscendC::AIC;
using AscendC::AIV;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

constexpr AscendC::MicroAPI::CastTrait ctInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctFp322Half = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait ctHalf2Fp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

template <class aType, class bType, class scaleType, class biasType, class ptScaleType, class cType, CubeFormat aFormat,
          CubeFormat bFormat, CubeFormat cFormat, bool aTrans, bool bTrans, class l0cDtype,
          class blockType = Mc2QuantBmmAswBlock, const MatmulConfig &mmCfg = MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>
class Mc2QuantBmmPertokenRegbaseKernel {
public:
    __aicore__ inline Mc2QuantBmmPertokenRegbaseKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR ptScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR ptScale,
                                            GM_ADDR cGM, GM_ADDR workSpace);
    __aicore__ inline void Process();

public:
    using aT = AscendC::MatmulType<TPosition::GM, aFormat, aType, aTrans>;
    using bT = AscendC::MatmulType<TPosition::GM, bFormat, bType, bTrans>;
    using biasT = AscendC::MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    using cT = AscendC::MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, l0cDtype>;
    AscendC::MatmulImpl<aT, bT, cT, biasT, mmCfg, AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                        AscendC::QBmmCustomMatmulPolicy>
        mm;

protected:
    __aicore__ inline void ProcessWithoutBatch();
    __aicore__ inline void ProcessWithBatch();
    __aicore__ inline void MMCompute();
    __aicore__ inline void DequantCompute();
    __aicore__ inline void VFDoDequantWithX1Pertoken(__ubuf__ cType *dequantOutInUbAddr,
                                                     __ubuf__ l0cDtype *l0cOutUbAddr, uint64_t offsetPtScale,
                                                     uint16_t mSize);
    __aicore__ inline void VFDoDequantWithX1Pertensor(__ubuf__ cType *dequantOutInUbAddr,
                                                      __ubuf__ l0cDtype *l0cOutUbAddr, uint16_t mSize);
    __aicore__ inline void VFDoDequantWithoutPertokenScale(__ubuf__ cType *dequantOutInUbAddr,
                                                           __ubuf__ l0cDtype *l0cOutUbAddr, uint16_t mSize);
    template <bool isPertensor, BasicQuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
    __aicore__ inline void VFDoDequant(__ubuf__ cType *dst, __ubuf__ l0cDtype *l0cOut, __ubuf__ scaleType *scale,
                                       __ubuf__ ptScaleType *perTokenScale, __ubuf__ BiasDtype *bias, uint16_t mSize,
                                       uint16_t nSize);
    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }
    __aicore__ inline void CopyDataFromGm2Ub();
    __aicore__ inline void CopyX1ScaleFromGm2Ub(LocalTensor<ptScaleType> &dst, uint64_t blockLen, uint64_t offset);
    __aicore__ inline void CopyX2ScaleFromGm2Ub(LocalTensor<scaleType> &dst);
    template <class BiasDtype>
    __aicore__ inline void CopyBiasFromGm2Ub(LocalTensor<BiasDtype> &dst);
    __aicore__ inline void CopyDequantResFromUb2Gm(uint64_t blockCount, uint64_t offset, LocalTensor<cType> &src);
    __aicore__ inline void FreeUbTensor();

protected:
    uint32_t blockIdx_;
    uint32_t subBlockIdx_;
    blockType block_;
    TPipe *pipe_;
    GlobalTensor<aType> aGlobal_;
    GlobalTensor<bType> bGlobal_;
    GlobalTensor<cType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<float> biasGlobalFloat_;
    GlobalTensor<bfloat16_t> biasGlobalB16_;
    GlobalTensor<scaleType> scaleGlobal_;
    GlobalTensor<ptScaleType> pertokenScaleGlobal_;
    LocalTensor<l0cDtype> l0cOutUb_;
    LocalTensor<scaleType> scaleUb_;
    LocalTensor<ptScaleType> ptScaleUb_;
    LocalTensor<bfloat16_t> biasUbB16_;
    LocalTensor<float> biasUbFloat_;
    float scaleScalar_;
    float pertokenScaleScalar_;

    // define the que
    TQue<QuePosition::VECIN, 1> vecQueMMRes_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;

    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *tilingData_;
    uint32_t biasDtype_;
    bool isBiasEpilogue_ = false;
};

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR ptScale, GM_ADDR cGM,
    GM_ADDR workSpace, const void *tilingData, TPipe *pipe)
{
    blockIdx_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / AscendC::GetTaskRation();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
    }
    pipe_ = pipe;
    tilingData_ = static_cast<const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *>(tilingData);

    biasDtype_ = tilingData_->params.biasDtype;
    isBiasEpilogue_ = IsSameType<aType, int8_t>::value &&
                      (biasDtype_ == DT_BF16 || biasDtype_ == DT_FLOAT16 || biasDtype_ == DT_FLOAT);
    UpdateGlobalAddr(aGM, bGM, scale, bias, ptScale, cGM, workSpace);
    if ASCEND_IS_AIC {
        mm.Init(&tilingData_->matmulTiling, pipe);
    }
    uint32_t mForSingleVec = DequantBmm::CeilDiv(tilingData_->matmulTiling.baseM, CV_RATIO);
    pipe_->InitBuffer(vecQueMMRes_, 1, mForSingleVec * tilingData_->matmulTiling.baseN * sizeof(l0cDtype));
    l0cOutUb_ = vecQueMMRes_.AllocTensor<l0cDtype>();
    // 仅AIV相关的buffer
    if ASCEND_IS_AIV {
        if (!static_cast<bool>(tilingData_->params.isPerTensor)) {
            pipe_->InitBuffer(vecQueScale_, 1, tilingData_->matmulTiling.baseN * sizeof(scaleType));
        }
        if (static_cast<bool>(tilingData_->params.isPertoken)) {
            pipe_->InitBuffer(
                vecQuePertokenScale_, 1,
                DequantBmm::Align(mForSingleVec * sizeof(ptScaleType), static_cast<uint64_t>(DATA_BLOCK)));
        }
        if (isBiasEpilogue_) {
            if (biasDtype_ == DT_FLOAT) {
                pipe_->InitBuffer(vecQueBias_, 1, tilingData_->matmulTiling.baseN * sizeof(float));
            } else {
                pipe_->InitBuffer(vecQueBias_, 1, tilingData_->matmulTiling.baseN * sizeof(bfloat16_t));
            }
        }
        // fp16/bf16分两次输出，fp32分四次输出
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM,
                          DequantBmm::CeilDiv(mForSingleVec, FP32_OUTPUT_TIMES) * tilingData_->matmulTiling.baseN *
                              sizeof(cType));
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::UpdateGlobalAddr(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR ptScale, GM_ADDR cGM, GM_ADDR workSpace)
{
    block_.Init(tilingData_, blockIdx_);
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer((__gm__ aType *)aGM);
        bGlobal_.SetGlobalBuffer((__gm__ bType *)bGM);
        // bias在L0C上累加的场景
        if (static_cast<bool>(tilingData_->matmulTiling.isBias) && !isBiasEpilogue_) {
            biasGlobal_.SetGlobalBuffer((__gm__ biasType *)bias);
        }
    }
    if ASCEND_IS_AIV {
        if (static_cast<bool>(tilingData_->params.isPerTensor)) {
            if constexpr (IsSameType<scaleType, bfloat16_t>::value) {
                bfloat16_t scaleBf16 = *((__gm__ scaleType *)scale);
                uint16_t uint16Scale = *(reinterpret_cast<uint16_t *>(&scaleBf16));
                uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << BMM_BLOCK_NUM);
                scaleScalar_ = *(reinterpret_cast<float *>(&uint32Scale));
            } else {
                scaleScalar_ = *((__gm__ float *)scale);
            }
        } else {
            scaleGlobal_.SetGlobalBuffer((__gm__ scaleType *)scale);
        }
        if (static_cast<bool>(tilingData_->params.isDoubleScale)) {
            pertokenScaleScalar_ = *((__gm__ ptScaleType *)ptScale);
        } else if (static_cast<bool>(tilingData_->params.isPertoken)) {
            pertokenScaleGlobal_.SetGlobalBuffer((__gm__ ptScaleType *)ptScale);
        }
        // bias 在UB上累加的场景
        if (static_cast<bool>(tilingData_->matmulTiling.isBias) && isBiasEpilogue_) {
            if (biasDtype_ == DT_FLOAT) {
                biasGlobalFloat_.SetGlobalBuffer((__gm__ float *)bias);
            } else {
                biasGlobalB16_.SetGlobalBuffer((__gm__ bfloat16_t *)bias);
            }
        }
        cGlobal_.SetGlobalBuffer((__gm__ cType *)cGM);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::Process()
{
    if (tilingData_->params.batchC == 1) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ProcessWithoutBatch();
    } else {
        ProcessWithBatch();
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::ProcessWithoutBatch()
{
    bool isVecSetSyncCom = false;
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        block_.UpdateBasicIndex(j);
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.template UpdateBlockParams<bTrans, bFormat>(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<aTrans, bTrans, aType, scaleType, bFormat>();
                if ASCEND_IS_AIC {
                    mm.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                                      tilingData_->matmulTiling.singleCoreK);
                    if (block_.offset_.batchCOffset * block_.params_.round + j > 0) {
                        WaitForVector();
                    }
                    MMCompute();
                    NotifyVector();
                }
                isVecSetSyncCom = true;
                if ASCEND_IS_AIV {
                    WaitForCube();
                    DequantCompute();
                    NotifyCube();
                }
            }
        }
    }
    // 由于vec最后一次会通过NotifyCube多发一次硬同步，所以cube侧需要额外加一次硬同步
    if ASCEND_IS_AIC {
        if (block_.offset_.batchCOffset == tilingData_->params.batchC - 1 && isVecSetSyncCom) {
            WaitForVector();
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::ProcessWithBatch()
{
    uint64_t batchC3C4 = static_cast<uint64_t>(tilingData_->params.batchC3) * tilingData_->params.batchC4;
    uint64_t batchC2C3C4 = tilingData_->params.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(tilingData_->params.batchB3) * tilingData_->params.batchB4;
    uint64_t batchB2B3B4 = tilingData_->params.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(tilingData_->params.batchA3) * tilingData_->params.batchA4;
    uint64_t batchA2A3A4 = tilingData_->params.batchA2 * batchA3A4;
    uint32_t multiA1C1 = tilingData_->params.batchA1 / tilingData_->params.batchC1;
    uint32_t multiA2C2 = tilingData_->params.batchA2 / tilingData_->params.batchC2;
    uint32_t multiA3C3 = tilingData_->params.batchA3 / tilingData_->params.batchC3;
    uint32_t multiA4C4 = tilingData_->params.batchA4 / tilingData_->params.batchC4;
    uint32_t multiB1C1 = tilingData_->params.batchB1 / tilingData_->params.batchC1;
    uint32_t multiB2C2 = tilingData_->params.batchB2 / tilingData_->params.batchC2;
    uint32_t multiB3C3 = tilingData_->params.batchB3 / tilingData_->params.batchC3;
    uint32_t multiB4C4 = tilingData_->params.batchB4 / tilingData_->params.batchC4;
    for (uint64_t b1Index = 0; b1Index < tilingData_->params.batchC1; ++b1Index) {
        uint64_t batchC1Offset = b1Index * batchC2C3C4;
        uint64_t batchA1Offset = b1Index * batchA2A3A4 * multiA1C1;
        uint64_t batchB1Offset = b1Index * batchB2B3B4 * multiB1C1;
        for (uint64_t b2Index = 0; b2Index < tilingData_->params.batchC2; ++b2Index) {
            uint64_t batchC2Offset = b2Index * batchC3C4 + batchC1Offset;
            uint64_t batchA2Offset = b2Index * batchA3A4 * multiA2C2 + batchA1Offset;
            uint64_t batchB2Offset = b2Index * batchB3B4 * multiB2C2 + batchB1Offset;
            for (uint64_t b3Index = 0; b3Index < tilingData_->params.batchC3; ++b3Index) {
                uint64_t batchC3Offset = b3Index * tilingData_->params.batchC4 + batchC2Offset;
                uint64_t batchA3Offset = b3Index * tilingData_->params.batchA4 * multiA3C3 + batchA2Offset;
                uint64_t batchB3Offset = b3Index * tilingData_->params.batchB4 * multiB3C3 + batchB2Offset;
                for (uint64_t b4Index = 0; b4Index < tilingData_->params.batchC4; ++b4Index) {
                    block_.offset_.batchCOffset = batchC3Offset + b4Index;
                    block_.offset_.batchAOffset = batchA3Offset + b4Index * multiA4C4;
                    block_.offset_.batchBOffset = batchB3Offset + b4Index * multiB4C4;
                    block_.ResetAddressOffsets();
                    ProcessWithoutBatch();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::MMCompute()
{
    mm.SetTensorA(aGlobal_[block_.offset_.offsetA], aTrans);
    mm.SetTensorB(bGlobal_[block_.offset_.offsetB], bTrans);
    if (static_cast<bool>(tilingData_->matmulTiling.isBias) && !isBiasEpilogue_) {
        mm.SetBias(biasGlobal_[block_.offset_.offsetBias]);
    }
    mm.Iterate();
    mm.GetTensorC(l0cOutUb_, 0, true);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::DequantCompute()
{
    auto halfSingleM = DequantBmm::CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    if (singleMInVec == 0) {
        return;
    }
    uint64_t mOffset = static_cast<uint64_t>(subBlockIdx_ * halfSingleM);
    CopyDataFromGm2Ub();
    // UB空间受限，分四次输出
    uint16_t splitNumOfOut = singleMInVec >= 4 ? 4 : singleMInVec;
    auto mSizeForOnce = DequantBmm::CeilDiv(singleMInVec, static_cast<uint64_t>(splitNumOfOut));
    for (uint16_t i = 0; i < splitNumOfOut; i++) {
        // do dequant in vector
        uint64_t offsetL0c =
            i * mSizeForOnce *
            DequantBmm::Align(block_.params_.singleCoreN, static_cast<uint64_t>(DATA_BLOCK / sizeof(l0cDtype)));
        if (i * mSizeForOnce >= singleMInVec) {
            break;
        }
        auto mSize = singleMInVec - i * mSizeForOnce >= mSizeForOnce ? mSizeForOnce : singleMInVec - i * mSizeForOnce;
        LocalTensor<cType> dequantOutInUB = vecQueOut_.AllocTensor<cType>();

        __ubuf__ cType *dequantOutInUbAddr = (__ubuf__ cType *)dequantOutInUB.GetPhyAddr();
        __ubuf__ l0cDtype *l0cOutUbAddr = (__ubuf__ l0cDtype *)l0cOutUb_.GetPhyAddr();
        l0cOutUbAddr = l0cOutUbAddr + offsetL0c;

        if (static_cast<bool>(tilingData_->params.isDoubleScale)) {
            VFDoDequantWithX1Pertensor(dequantOutInUbAddr, l0cOutUbAddr, mSize);
        } else if (static_cast<bool>(tilingData_->params.isPertoken)) {
            uint64_t offsetPtScale = i * mSizeForOnce;
            VFDoDequantWithX1Pertoken(dequantOutInUbAddr, l0cOutUbAddr, offsetPtScale, mSize);
        } else {
            VFDoDequantWithoutPertokenScale(dequantOutInUbAddr, l0cOutUbAddr, mSize);
        }
        vecQueOut_.EnQue<cType>(dequantOutInUB);
        // mmDequant result: UB -> GM
        dequantOutInUB = vecQueOut_.DeQue<cType>();
        CopyDequantResFromUb2Gm(mSize, (mOffset + i * mSizeForOnce) * tilingData_->matmulTiling.N, dequantOutInUB);
        vecQueOut_.FreeTensor(dequantOutInUB);
    }
    FreeUbTensor();
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDataFromGm2Ub()
{
    auto halfSingleM = DequantBmm::CeilDiv(block_.params_.singleCoreM, static_cast<uint64_t>(2)); // 分配给2个AIV计算
    auto singleMInVec = subBlockIdx_ == 1 ? block_.params_.singleCoreM - halfSingleM : halfSingleM;
    // scale: GM -> UB
    if (!static_cast<bool>(tilingData_->params.isPerTensor)) {
        scaleUb_ = vecQueScale_.AllocTensor<scaleType>();
        CopyX2ScaleFromGm2Ub(scaleUb_);
        vecQueScale_.EnQue<scaleType>(scaleUb_);
        scaleUb_ = vecQueScale_.DeQue<scaleType>();
    }

    uint64_t mOffset = subBlockIdx_ * halfSingleM;
    // perTokenScale: GM -> UB
    if (static_cast<bool>(tilingData_->params.isPertoken)) {
        ptScaleUb_ = vecQuePertokenScale_.AllocTensor<ptScaleType>();
        CopyX1ScaleFromGm2Ub(ptScaleUb_, singleMInVec * sizeof(ptScaleType),
                             block_.offset_.offsetPerTokenScale + mOffset);
        vecQuePertokenScale_.EnQue<ptScaleType>(ptScaleUb_);
        ptScaleUb_ = vecQuePertokenScale_.DeQue<ptScaleType>();
    }
    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            biasUbFloat_ = vecQueBias_.AllocTensor<float>();
            CopyBiasFromGm2Ub<float>(biasUbFloat_);
            vecQueBias_.EnQue<float>(biasUbFloat_);
            biasUbFloat_ = vecQueBias_.DeQue<float>();
        } else {
            biasUbB16_ = vecQueBias_.AllocTensor<bfloat16_t>();
            CopyBiasFromGm2Ub<bfloat16_t>(biasUbB16_);
            vecQueBias_.EnQue<bfloat16_t>(biasUbB16_);
            biasUbB16_ = vecQueBias_.DeQue<bfloat16_t>();
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX1ScaleFromGm2Ub(
    LocalTensor<ptScaleType> &dst, uint64_t blockLen, uint64_t offset)
{
    DataCopyParams ptScale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    ptScale2UbParams.blockLen = blockLen;
    AscendC::DataCopyPad(dst, pertokenScaleGlobal_[offset], ptScale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyX2ScaleFromGm2Ub(LocalTensor<scaleType> &dst)
{
    DataCopyParams scale2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    scale2UbParams.blockLen = block_.params_.singleCoreN * sizeof(scaleType);
    AscendC::DataCopyPad(dst, scaleGlobal_[block_.offset_.offsetScale], scale2UbParams, padParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
template <class BiasDtype>
__aicore__ inline void
Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyBiasFromGm2Ub(LocalTensor<BiasDtype> &dst)
{
    DataCopyParams bias2UbParams{1, 0, 0, 0};
    DataCopyPadParams padParams;
    if constexpr (IsSameType<BiasDtype, float>::value) {
        bias2UbParams.blockLen = block_.params_.singleCoreN * sizeof(float);
        AscendC::DataCopyPad(dst, biasGlobalFloat_[block_.offset_.offsetBias], bias2UbParams, padParams);
    } else {
        bias2UbParams.blockLen = block_.params_.singleCoreN * sizeof(bfloat16_t);
        AscendC::DataCopyPad(dst, biasGlobalB16_[block_.offset_.offsetBias], bias2UbParams, padParams);
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::CopyDequantResFromUb2Gm(
    uint64_t blockCount, uint64_t offset, LocalTensor<cType> &src)
{
    DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockLen = block_.params_.singleCoreN * sizeof(cType);
    ub2GmParams.blockCount = blockCount;
    ub2GmParams.dstStride = (tilingData_->matmulTiling.N - block_.params_.singleCoreN) * sizeof(cType);
    AscendC::DataCopyPad(cGlobal_[block_.offset_.offsetC + offset], src, ub2GmParams);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::FreeUbTensor()
{
    if (!static_cast<bool>(tilingData_->params.isPerTensor)) {
        vecQueScale_.FreeTensor(scaleUb_);
    }

    if (static_cast<bool>(tilingData_->params.isPertoken)) {
        vecQuePertokenScale_.FreeTensor(ptScaleUb_);
    }

    if (isBiasEpilogue_) {
        if (biasDtype_ == DT_FLOAT) {
            vecQueBias_.FreeTensor(biasUbFloat_);
        } else {
            vecQueBias_.FreeTensor(biasUbB16_);
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithX1Pertoken(
    __ubuf__ cType *dequantOutInUbAddr, __ubuf__ l0cDtype *l0cOutUbAddr, uint64_t offsetPtScale, uint16_t mSize)
{
    __ubuf__ ptScaleType *ptScaleUbAddr = (__ubuf__ ptScaleType *)ptScaleUb_.GetPhyAddr();
    ptScaleUbAddr = ptScaleUbAddr + offsetPtScale;
    if (!isBiasEpilogue_) {
        if (static_cast<bool>(tilingData_->params.isPerTensor)) {
            VFDoDequant<true, BasicQuantMode::PERTOKEN_MODE, false, float>(
                dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, nullptr, mSize, block_.params_.singleCoreN);
        } else {
            VFDoDequant<false, BasicQuantMode::PERTOKEN_MODE, false, float>(
                dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr, nullptr,
                mSize, block_.params_.singleCoreN);
        }
    } else {
        if (biasDtype_ == DT_FLOAT) {
            if (static_cast<bool>(tilingData_->params.isPerTensor)) {
                VFDoDequant<true, BasicQuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            } else {
                VFDoDequant<false, BasicQuantMode::PERTOKEN_MODE, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            }
        } else if (biasDtype_ == DT_BF16) {
            if (static_cast<bool>(tilingData_->params.isPerTensor)) {
                VFDoDequant<true, BasicQuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr,
                    (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            } else {
                VFDoDequant<false, BasicQuantMode::PERTOKEN_MODE, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            }
        } else if (biasDtype_ == DT_FLOAT16) {
            if (static_cast<bool>(tilingData_->params.isPerTensor)) {
                VFDoDequant<true, BasicQuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, ptScaleUbAddr, (__ubuf__ half *)biasUbB16_.GetPhyAddr(),
                    mSize, block_.params_.singleCoreN);
            } else {
                VFDoDequant<false, BasicQuantMode::PERTOKEN_MODE, true, half>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), ptScaleUbAddr,
                    (__ubuf__ half *)biasUbB16_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithX1Pertensor(
    __ubuf__ cType *dequantOutInUbAddr, __ubuf__ l0cDtype *l0cOutUbAddr, uint16_t mSize)
{
    VFDoDequant<false, BasicQuantMode::PERTENSOR_MODE, false, float>(
        dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr, nullptr, mSize,
        block_.params_.singleCoreN);
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
__aicore__ inline void
Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequantWithoutPertokenScale(
    __ubuf__ cType *dequantOutInUbAddr, __ubuf__ l0cDtype *l0cOutUbAddr, uint16_t mSize)
{
    if (!isBiasEpilogue_) {
        VFDoDequant<false, BasicQuantMode::DEFAULT, false, float>(dequantOutInUbAddr, l0cOutUbAddr,
                                                                  (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                                                                  nullptr, mSize, block_.params_.singleCoreN);
    } else {
        if (biasDtype_ == DT_FLOAT) {
            if (static_cast<bool>(tilingData_->params.isPerTensor)) {
                VFDoDequant<true, BasicQuantMode::DEFAULT, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ float *)biasUbFloat_.GetPhyAddr(),
                    mSize, block_.params_.singleCoreN);
            } else {
                VFDoDequant<false, BasicQuantMode::DEFAULT, true, float>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ float *)biasUbFloat_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            }
        } else if (biasDtype_ == DT_BF16) {
            if (static_cast<bool>(tilingData_->params.isPerTensor)) {
                VFDoDequant<true, BasicQuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, nullptr, nullptr, (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(),
                    mSize, block_.params_.singleCoreN);
            } else {
                VFDoDequant<false, BasicQuantMode::DEFAULT, true, bfloat16_t>(
                    dequantOutInUbAddr, l0cOutUbAddr, (__ubuf__ scaleType *)scaleUb_.GetPhyAddr(), nullptr,
                    (__ubuf__ bfloat16_t *)biasUbB16_.GetPhyAddr(), mSize, block_.params_.singleCoreN);
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_MIX_PARAMS
template <bool isPertensor, BasicQuantMode x1QuantMode, bool isBiasEpilogue, class BiasDtype>
__aicore__ inline void Mc2QuantBmmPertokenRegbaseKernel<LOCAL_TEMPLATE_FUNC_MIX_PARAMS>::VFDoDequant(
    __ubuf__ cType *dst, __ubuf__ l0cDtype *l0cOut, __ubuf__ scaleType *scale, __ubuf__ ptScaleType *perTokenScale,
    __ubuf__ BiasDtype *bias, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = GetVectorRegSize() / sizeof(l0cDtype);
    uint32_t nSrcUbAligned = DequantBmm::Align(nSize, static_cast<uint16_t>(DATA_BLOCK / sizeof(l0cDtype)));
    uint32_t nDstUbAligned = DequantBmm::Align(nSize, static_cast<uint16_t>(DATA_BLOCK / sizeof(cType)));
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskN4B16 =
            AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::RegTensor<l0cDtype> l0cOutReg;
                AscendC::MicroAPI::RegTensor<scaleType> scaleReg;
                AscendC::MicroAPI::RegTensor<ptScaleType> perTokenScaleReg;
                AscendC::MicroAPI::RegTensor<BiasDtype> biasReg;
                AscendC::MicroAPI::RegTensor<float> castSrcOutReg, castScaleReg, castScaleOneReg, mulScaleOutReg,
                    mulPtScaleOutReg, castBiasReg, castBiasOneReg, addBiasOutReg;
                AscendC::MicroAPI::RegTensor<cType> castResultOutReg;
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<l0cDtype>(elementNum);
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // cast l0cOut from int32 to float
                if constexpr (IsSameType<l0cDtype, int32_t>::value) {
                    AscendC::MicroAPI::Cast<float, l0cDtype, ctInt322Fp32>(castSrcOutReg, l0cOutReg, maskN);
                } else {
                    castSrcOutReg = l0cOutReg;
                }
                // l0c_out * scale
                if constexpr (isPertensor) {
                    AscendC::MicroAPI::Muls(mulScaleOutReg, castSrcOutReg, scaleScalar_, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(scaleReg, scale + vfBlockIdx * eleNumPerVf);
                    if constexpr (!IsSameType<scaleType, float>::value) { // cast scale from bf16 to float
                        AscendC::MicroAPI::Cast<float, scaleType, ctHalf2Fp32Zero>(castScaleReg, scaleReg, maskN);
                        AscendC::MicroAPI::Cast<float, scaleType, ctHalf2Fp32One>(castScaleOneReg, scaleReg, maskN4B16);
                        AscendC::MicroAPI::Interleave(castScaleReg, castScaleOneReg, castScaleReg, castScaleOneReg);
                    } else {
                        castScaleReg = scaleReg;
                    }
                    AscendC::MicroAPI::Mul(mulScaleOutReg, castSrcOutReg, castScaleReg, maskN);
                }
                // out * perTokenScale
                if constexpr (x1QuantMode == BasicQuantMode::PERTENSOR_MODE) {
                    AscendC::MicroAPI::Muls(mulPtScaleOutReg, mulScaleOutReg, pertokenScaleScalar_, maskN);
                } else if constexpr (x1QuantMode == BasicQuantMode::PERTOKEN_MODE) {
                    AscendC::MicroAPI::DataCopy<ptScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        perTokenScaleReg, perTokenScale + mIdx);
                    AscendC::MicroAPI::Mul(mulPtScaleOutReg, mulScaleOutReg, perTokenScaleReg, maskN);
                } else {
                    mulPtScaleOutReg = mulScaleOutReg;
                }
                // out + bias
                if constexpr (isBiasEpilogue) {
                    AscendC::MicroAPI::DataCopy(biasReg, bias + vfBlockIdx * eleNumPerVf);
                    // cast bias from bf16/fp16 to float
                    if constexpr (IsSameType<BiasDtype, bfloat16_t>::value || IsSameType<BiasDtype, half>::value) {
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32Zero>(castBiasReg, biasReg, maskN);
                        AscendC::MicroAPI::Cast<float, BiasDtype, ctHalf2Fp32One>(castBiasOneReg, biasReg, maskN4B16);
                        AscendC::MicroAPI::Interleave(castBiasReg, castBiasOneReg, castBiasReg, castBiasOneReg);
                    } else {
                        castBiasReg = biasReg;
                    }
                    AscendC::MicroAPI::Add(addBiasOutReg, mulPtScaleOutReg, castBiasReg, maskN);
                } else {
                    addBiasOutReg = mulPtScaleOutReg;
                }
                // cast dequant result from float to fp16/bf16
                if constexpr (!IsSameType<cType, float>::value) {
                    AscendC::MicroAPI::Cast<cType, float, ctFp322Half>(castResultOutReg, addBiasOutReg, maskN);
                } else {
                    castResultOutReg = addBiasOutReg;
                }
                // copy out from register to ub
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf;
                if constexpr (IsSameType<cType, float>::value) {
                    AscendC::MicroAPI::DataCopy<cType, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, castResultOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy<cType, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        dst + dstUbOffset, castResultOutReg, maskN);
                }
            }
        }
    }
}

} // namespace Mc2QuantBatchMatmulV3

#endif // QBMM_MIX_ONLINE_DYNAMIC_H
