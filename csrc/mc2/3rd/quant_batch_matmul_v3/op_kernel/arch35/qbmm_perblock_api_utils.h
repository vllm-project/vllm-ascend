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
 * \file qbmm_perblock_api_utils.h
 * \brief
 */

#ifndef MC2_QBMM_PERBLOCK_API_UTILS_H
#define MC2_QBMM_PERBLOCK_API_UTILS_H

#include "qbmm_asw_block.h"
#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_perblock_api_param_utils.h"

namespace Mc2QuantBatchMatmulV3 {

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
class MatMulPerBlock {
public:
    __aicore__ inline MatMulPerBlock(){};
    __aicore__ inline void Init(const TCubeTiling &tilingData, GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale,
                                GM_ADDR perTokenScale, GM_ADDR cGM, blockType &block, GM_ADDR workSpace, TPipe *pipe);
    __aicore__ inline void InitAic(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR cGM, uint64_t baseL0c);
    __aicore__ inline void AicBaseMadProcess(LocalTensor<aType> &aL1, LocalTensor<bType> &bL1, uint64_t &kInner,
                                             uint64_t kAL1Offset, bool isTailAL1, uint64_t kBL1Offset, bool isTailBL1);
    __aicore__ inline void WaitForVector(uint16_t &crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + crossPingPongID);
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + crossPingPongID + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyV2Mte3(uint16_t id)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdV2Mte3[id]);
        WaitFlag<HardEvent::V_MTE3>(eventIdV2Mte3[id]);
    }
    __aicore__ inline void NotifyVector(uint16_t &crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + crossPingPongID);
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + crossPingPongID + FLAG_ID_MAX);
    }
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensor(__ubuf__ l0cDtype *dst, __ubuf__ l0cDtype *l0cOut, float scaleScalar,
                                        uint16_t mSize, uint16_t nSize, uint32_t nSrcUbAligned);
    template <bool isFirstKLoop>
    __aicore__ inline void AivPerTensor(__ubuf__ l0cDtype *dst, __ubuf__ l0cDtype *l0cOut,
                                        __ubuf__ ptScaleType *muledScale, uint16_t mSize, uint16_t nSize,
                                        uint32_t nSrcUbAligned);
    __aicore__ inline scaleType GetScaleX2(GM_ADDR scaleX2AddrOff, uint64_t kOffset);
    __aicore__ inline uint64_t GetScaleX1Offset(const uint64_t scaleOffsetX1, uint64_t kOffset);
    __aicore__ inline void ProcessKBlock(LocalTensor<l0cDtype> &mmAddUb, LocalTensor<ptScaleType> &mmPertokenScaleUb,
                                         LocalTensor<scaleType> &mmMuledScaleUb, const uint64_t scaleOffsetX1,
                                         GM_ADDR scaleX2AddrOff, uint64_t kb, uint64_t kOffset, uint64_t alignedM);
    __aicore__ inline void PerformTensorOps(LocalTensor<l0cDtype> &mmAddUb, LocalTensor<scaleType> &mmMuledScaleUb,
                                            uint64_t kb);
    __aicore__ inline void UpdatePertokenScale(LocalTensor<ptScaleType> &mmPertokenScaleUb, uint64_t scaleX1GmOffset,
                                               uint64_t kb);
    __aicore__ inline void UpdatePingPongState();
    __aicore__ inline void ClearAL1Space();
    __aicore__ inline void CopyInA1Nd2Nz(const uint64_t &kOffset, const bool &isTailAL1);
    __aicore__ inline void CopyInB1Nd2Nz(const uint64_t &kOffset, const bool &isTailBL1);
    __aicore__ inline void CopyInA2(LocalTensor<aType> &aL1, const uint64_t &mAL1Offset, const uint64_t &kAL1Offset,
                                    const uint64_t &kOffset, const bool &isTailAL1);
    __aicore__ inline void CopyInB2(LocalTensor<bType> &bL1, const uint64_t &nBL1Offset, const uint64_t &kBL1Offset,
                                    const uint64_t &kOffset, const bool &isTailBL1);
    __aicore__ inline void MmadBase(const uint64_t &kOffset);
    __aicore__ inline void AivPostProcess(LocalTensor<l0cDtype> &mmAddUb);
    __aicore__ inline void ProcessAivSingleK();
    __aicore__ inline void ProcessAivSingleKPertile();
    __aicore__ inline void ProcessAicSingleK();
    __aicore__ inline void Iterate();
    __aicore__ inline void AivEnd();
    __aicore__ inline void AicEnd();

protected:
    blockType *block_;
    TPipe *pipe_;
    const TCubeTiling *matmulTiling_;
    MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS> matmulParam_;
    bool needAicWait_ = false; // 是否为第一个block，只有核的第一个基本块的前两次L0C2UB不需要等待AIV
    bool orderAL1BL1_ = false;
    bool isPertile_ = false;
    uint64_t baseCount_ = 0;
    uint16_t crossPingPongID_ = 0;
    uint16_t pertokenPingPongID_ = 0;
    uint64_t pertokenRoundIdx_ = 0;
    uint64_t scaleK_ = 0;
    uint64_t scaleM_ = 0;
    uint64_t scaleN_ = 0;
    uint64_t maxStepK_ = 0;
    uint64_t minStepK_ = 0;
    uint16_t aL1BlockNum_ = 0;
    uint32_t kCopySizeX1Scale_ = 16; // in one copy instruction, 16 elements copied in k direction for x1_scale
    GM_ADDR scaleX2Addr_;
    GM_ADDR scaleX1Addr_;
    GlobalTensor<aType> aGlobal_;
    GlobalTensor<bType> bGlobal_;
    GlobalTensor<cType> cGlobal_;
    GlobalTensor<ptScaleType> pertokenScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    LocalTensor<l0cDtype> mmResPing_;
    LocalTensor<l0cDtype> mmResPong_;
    LocalTensor<ptScaleType> mmPertokenScaleUbPing_;
    LocalTensor<ptScaleType> mmPertokenScaleUbPong_;
    MultiCopyParams<ptScaleType, 2> paramsMain_; // nddma params, 2 means copy 2 dim data
    // define the queue
    TQue<QuePosition::A1, 1> inQueueTensorAL1;
    TQue<QuePosition::B1, 1> inQueueTensorBL1;
    TQue<QuePosition::A2, 1> inQueueTensorAL0;
    TQue<QuePosition::B2, 1> inQueueTensorBL0;
    TQue<QuePosition::CO1, 1> inQueueTensorCL0;
    TQue<QuePosition::VECIN, 1> vecQueMMRes_;
    TQue<QuePosition::VECIN, 1> vecQueAdd_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TQue<QuePosition::VECIN, 1> vecQueMuledScale_;
    event_t eventIdV2Mte2[2] = {EVENT_ID0, EVENT_ID1};
    event_t eventIdV2Mte3[2] = {EVENT_ID0, EVENT_ID1};
    event_t eventIdMTE32V_ = EVENT_ID0;
};

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::Init(const TCubeTiling &tilingData, GM_ADDR aGM,
                                                                         GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale,
                                                                         GM_ADDR perTokenScale, GM_ADDR cGM,
                                                                         blockType &block, GM_ADDR workSpace,
                                                                         TPipe *pipe)
{
    matmulTiling_ = &tilingData;
    block_ = &block;
    pipe_ = pipe;
    matmulParam_.Init(block, tilingData);
    crossPingPongID_ = 0;
    needAicWait_ = false;
    baseCount_ = 0;

    // init global tensor
    // in core
    uint64_t baseL0c = matmulTiling_->baseM * matmulTiling_->baseN;
    InitAic(aGM, bGM, bias, cGM, baseL0c);
    pipe_->InitBuffer(vecQueMMRes_, BUFFER_NUM, baseL0c * sizeof(l0cDtype) / BUFFER_NUM);
    mmResPing_ = vecQueMMRes_.template AllocTensor<l0cDtype>();
    mmResPong_ = vecQueMMRes_.template AllocTensor<l0cDtype>();
    if ASCEND_IS_AIV {
        pipe_->InitBuffer(vecQueAdd_, 1, baseL0c * sizeof(l0cDtype) / BUFFER_NUM);
        if constexpr (!IsSameType<l0cDtype, cType>::value) {
            pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, baseL0c * sizeof(cType) / BUFFER_NUM / 2); // 分两次搬运出去
        }
        scaleK_ = DequantBmm::CeilDiv(matmulTiling_->Ka, PER_BLOCK_SIZE);
        scaleM_ = DequantBmm::CeilDiv(matmulTiling_->M, block_->params_.groupSizeM);
        scaleN_ = DequantBmm::CeilDiv(matmulTiling_->N, PER_BLOCK_SIZE);
        scaleX2Addr_ = scale;
        scaleX1Addr_ = perTokenScale;
        cGlobal_.SetGlobalBuffer((__gm__ cType *)cGM);
        pertokenScaleGlobal_.SetGlobalBuffer((__gm__ ptScaleType *)perTokenScale);
        isPertile_ = block_->params_.groupSizeM == 1;
        if (isPertile_) {
            pertokenPingPongID_ = 0;
            pertokenRoundIdx_ = 0;
            pipe_->InitBuffer(vecQuePertokenScale_, BUFFER_NUM, 16 * PER_BLOCK_SIZE * sizeof(ptScaleType));
            pipe_->InitBuffer(vecQueMuledScale_, 1, PER_BLOCK_SIZE * sizeof(ptScaleType));
        }
        eventIdMTE32V_ = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE3_V>());
        eventIdV2Mte2[0] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE2>());
        eventIdV2Mte2[1] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE2>());
        eventIdV2Mte3[0] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE3>());
        if constexpr (!IsSameType<cType, l0cDtype>::value) {
            eventIdV2Mte3[1] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::V_MTE3>());
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::InitAic(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias,
                                                                            GM_ADDR cGM, uint64_t baseL0c)
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer((__gm__ aType *)aGM);
        bGlobal_.SetGlobalBuffer((__gm__ bType *)bGM);
        if (static_cast<bool>(matmulTiling_->isBias)) {
            biasGlobal_.SetGlobalBuffer((__gm__ biasType *)bias);
        }
        orderAL1BL1_ = matmulTiling_->stepKa >= matmulTiling_->stepKb;
        maxStepK_ = (orderAL1BL1_ ? matmulTiling_->stepKa : matmulTiling_->stepKb) * matmulTiling_->baseK;
        minStepK_ = (orderAL1BL1_ ? matmulTiling_->stepKb : matmulTiling_->stepKa) * matmulTiling_->baseK;
        // L1 默认开乒乓
        auto aL1ElementNum = matmulTiling_->baseM * matmulTiling_->baseK * matmulTiling_->stepKa;
        aL1BlockNum_ = static_cast<uint16_t>(aL1ElementNum * sizeof(aType) / static_cast<uint64_t>(DATA_BLOCK));
        pipe_->InitBuffer(inQueueTensorAL1, BUFFER_NUM, aL1ElementNum);
        pipe_->InitBuffer(inQueueTensorBL1, BUFFER_NUM,
                          matmulTiling_->baseN * matmulTiling_->baseK * matmulTiling_->stepKb);
        // L0 默认开乒乓
        pipe_->InitBuffer(inQueueTensorAL0, BUFFER_NUM, matmulTiling_->baseM * matmulTiling_->baseK);
        pipe_->InitBuffer(inQueueTensorBL0, BUFFER_NUM, matmulTiling_->baseN * matmulTiling_->baseK);
        pipe_->InitBuffer(inQueueTensorCL0, BUFFER_NUM, baseL0c * sizeof(l0cDtype));
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::Iterate()
{
    if ASCEND_IS_AIC {
        ProcessAicSingleK();
    }
    if ASCEND_IS_AIV {
        if (isPertile_) {
            ProcessAivSingleKPertile();
        } else {
            ProcessAivSingleK();
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAicSingleK()
{
    if (!aTrans && block_->params_.singleCoreM == 1) {
        ClearAL1Space();
    }
    bool isTailAL1 = false;
    bool isTailBL1 = false;
    if (orderAL1BL1_) {
        for (uint64_t kOuter = 0; kOuter < matmulTiling_->Ka; kOuter += maxStepK_) {
            isTailAL1 = (kOuter + maxStepK_) >= matmulTiling_->Ka;
            CopyInA1Nd2Nz(kOuter, isTailAL1);
            auto aL1 = inQueueTensorAL1.template DeQue<aType>();
            for (uint64_t kInner = kOuter;
                 kInner < DequantBmm::Min(kOuter + maxStepK_, static_cast<uint64_t>(matmulTiling_->Ka));
                 kInner += minStepK_) {
                isTailBL1 = (kInner + minStepK_) >= matmulTiling_->Kb;
                CopyInB1Nd2Nz(kInner, isTailBL1);
                auto bL1 = inQueueTensorBL1.template DeQue<bType>();
                uint64_t kAL1Offset = kInner - kOuter;
                AicBaseMadProcess(aL1, bL1, kInner, kAL1Offset, isTailAL1, 0UL, isTailBL1);
                inQueueTensorBL1.FreeTensor(bL1);
            }
            inQueueTensorAL1.FreeTensor(aL1);
        }
    } else {
        for (uint64_t kOuter = 0; kOuter < matmulTiling_->Ka; kOuter += maxStepK_) {
            isTailBL1 = (kOuter + maxStepK_) >= matmulTiling_->Kb;
            CopyInB1Nd2Nz(kOuter, isTailBL1);
            auto bL1 = inQueueTensorBL1.template DeQue<bType>();
            for (uint64_t kInner = kOuter;
                 kInner < DequantBmm::Min(kOuter + maxStepK_, static_cast<uint64_t>(matmulTiling_->Ka));
                 kInner += minStepK_) {
                isTailAL1 = (kInner + minStepK_) >= matmulTiling_->Ka;
                CopyInA1Nd2Nz(kInner, isTailAL1);
                uint64_t kBL1Offset = kInner - kOuter;
                auto aL1 = inQueueTensorAL1.template DeQue<aType>();
                AicBaseMadProcess(aL1, bL1, kInner, 0UL, isTailAL1, kBL1Offset, isTailBL1);
                inQueueTensorAL1.FreeTensor(aL1);
            }
            inQueueTensorBL1.FreeTensor(bL1);
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void
MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AicBaseMadProcess(LocalTensor<aType> &aL1, LocalTensor<bType> &bL1,
                                                               uint64_t &kInner, uint64_t kAL1Offset, bool isTailAL1,
                                                               uint64_t kBL1Offset, bool isTailBL1)
{
    for (uint64_t kb = kInner; kb < DequantBmm::Min(kInner + minStepK_, static_cast<uint64_t>(matmulTiling_->Ka));
         kb += matmulTiling_->baseK) {
        CopyInA2(aL1, 0, kAL1Offset, kb, isTailAL1);
        CopyInB2(bL1, 0, kBL1Offset, kb, isTailBL1);
        MmadBase(kb);
        auto cL0 = inQueueTensorCL0.template DeQue<l0cDtype>();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams(block_->mmParams_.fixpipeN, block_->mmParams_.fixpipeM,
                                                              block_->mmParams_.fixSrcStride,
                                                              block_->mmParams_.fixpipeD);
        fixpipeParams.dualDstCtl = block_->mmParams_.fixpipeSplitN ? 2 : 1; // 2 表示 1:2切分N
        if (needAicWait_) {
            WaitForVector(crossPingPongID_);
        }
        Fixpipe<l0cDtype, l0cDtype, AscendC::Impl::CFG_ROW_MAJOR_UB>(crossPingPongID_ == 0 ? mmResPing_ : mmResPong_,
                                                                     cL0, fixpipeParams);
        NotifyVector(crossPingPongID_);
        needAicWait_ = needAicWait_ || crossPingPongID_ == 1;
        inQueueTensorCL0.FreeTensor(cL0);
        crossPingPongID_ = (crossPingPongID_ + 1UL) & 1UL;
        kAL1Offset = kAL1Offset + matmulTiling_->baseK;
        kBL1Offset = kBL1Offset + matmulTiling_->baseK;
        baseCount_++;
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AivPerTensor(__ubuf__ l0cDtype *dst,
                                                                                 __ubuf__ l0cDtype *l0cOut,
                                                                                 float scaleScalar, uint16_t mSize,
                                                                                 uint16_t nSize, uint32_t nSrcUbAligned)
{
    uint32_t eleNumPerVf = GetVectorRegSize() / sizeof(l0cDtype);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<l0cDtype>(elementNum);
                AscendC::MicroAPI::RegTensor<l0cDtype> l0cOutReg;
                AscendC::MicroAPI::RegTensor<l0cDtype> addReg;
                AscendC::MicroAPI::RegTensor<l0cDtype> ResReg, mulScaleOutReg;
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // l0c_out * scale
                AscendC::MicroAPI::Muls(mulScaleOutReg, l0cOutReg, scaleScalar, maskN);
                uint32_t dstUbOffset = l0cOutOffset;
                if constexpr (isFirstKLoop) {
                    AscendC::MicroAPI::DataCopy<l0cDtype, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, mulScaleOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(addReg, dst + l0cOutOffset);
                    AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                    // copy out from register to ub
                    AscendC::MicroAPI::DataCopy<l0cDtype, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, ResReg, maskN);
                }
            }
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
template <bool isFirstKLoop>
__aicore__ inline void
MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AivPerTensor(__ubuf__ l0cDtype *dst, __ubuf__ l0cDtype *l0cOut,
                                                          __ubuf__ ptScaleType *muledScale, uint16_t mSize,
                                                          uint16_t nSize, uint32_t nSrcUbAligned)
{
    uint32_t eleNumPerVf = GetVectorRegSize() / sizeof(l0cDtype);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    __VEC_SCOPE__
    {
        for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
            uint32_t elementNum = nSize;
            AscendC::MicroAPI::RegTensor<ptScaleType> muledScaleReg;
            AscendC::MicroAPI::DataCopy<ptScaleType, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(muledScaleReg,
                                                                                                muledScale + mIdx);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                AscendC::MicroAPI::MaskReg maskN = AscendC::MicroAPI::UpdateMask<l0cDtype>(elementNum);
                AscendC::MicroAPI::RegTensor<l0cDtype> l0cOutReg;
                AscendC::MicroAPI::RegTensor<l0cDtype> addReg;
                AscendC::MicroAPI::RegTensor<l0cDtype> ResReg, mulScaleOutReg;
                // copy input from ub to register, addr of ub should align to 32B
                uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf;
                AscendC::MicroAPI::DataCopy(l0cOutReg, l0cOut + l0cOutOffset);
                // l0c_out * scale
                AscendC::MicroAPI::Mul(mulScaleOutReg, l0cOutReg, muledScaleReg, maskN);
                uint32_t dstUbOffset = l0cOutOffset;
                if constexpr (isFirstKLoop) {
                    AscendC::MicroAPI::DataCopy<l0cDtype, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, mulScaleOutReg, maskN);
                } else {
                    AscendC::MicroAPI::DataCopy(addReg, dst + l0cOutOffset);
                    AscendC::MicroAPI::Add(ResReg, mulScaleOutReg, addReg, maskN);
                    // copy out from register to ub
                    AscendC::MicroAPI::DataCopy<l0cDtype, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
                        dst + dstUbOffset, ResReg, maskN);
                }
            }
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAivSingleK()
{
    block_->UpdatePerBlockUBParam();
    uint64_t scaleOffsetX1 = block_->offset_.batchAOffset * scaleK_ * scaleM_;
    uint64_t scaleOffsetX2 = block_->offset_.batchBOffset * scaleK_ * scaleN_;
    if constexpr (aTrans) {
        scaleOffsetX1 += block_->ubParams_.offsetScaleM;
    } else {
        scaleOffsetX1 += block_->ubParams_.offsetScaleM * scaleK_;
    }
    if constexpr (bTrans) {
        scaleOffsetX2 += block_->ubParams_.offsetScaleN * scaleK_;
    } else {
        scaleOffsetX2 += block_->ubParams_.offsetScaleN;
    }
    GM_ADDR scaleX1AddrOff = scaleX1Addr_ + sizeof(ptScaleType) * (scaleOffsetX1);
    GM_ADDR scaleX2AddrOff = scaleX2Addr_ + sizeof(scaleType) * (scaleOffsetX2);
    LocalTensor<l0cDtype> mmAddUb = vecQueAdd_.template AllocTensor<l0cDtype>();
    for (uint64_t kb = 0, kOffset = 0; kb < matmulTiling_->Ka; kb += matmulTiling_->baseK, kOffset++) {
        ptScaleType scaleX1 = 0;
        scaleType scaleX2 = 0;
        if constexpr (aTrans) {
            scaleX1 = *((__gm__ ptScaleType *)(scaleX1AddrOff + sizeof(ptScaleType) * kOffset * scaleM_));
        } else {
            scaleX1 = *((__gm__ ptScaleType *)(scaleX1AddrOff + sizeof(ptScaleType) * kOffset));
        }
        if constexpr (bTrans) {
            scaleX2 = *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset));
        } else {
            scaleX2 = *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset * scaleN_));
        }
        scaleType scaleMul = scaleX1 * scaleX2;
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + crossPingPongID_);
        auto mmUbInput = crossPingPongID_ == 0 ? mmResPing_ : mmResPong_;
        uint64_t mmAddUbAddr = reinterpret_cast<uint64_t>(mmAddUb.GetPhyAddr());
        uint64_t mmUbInputAddr = reinterpret_cast<uint64_t>(mmUbInput.GetPhyAddr());
        AscendC::PipeBarrier<PIPE_V>();
        if (kb == 0) {
            AivPerTensor<true>((__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr, scaleMul,
                               block_->ubParams_.validM, block_->ubParams_.validN, block_->ubParams_.singleN);
        } else {
            AivPerTensor<false>((__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr, scaleMul,
                                block_->ubParams_.validM, block_->ubParams_.validN, block_->ubParams_.singleN);
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + crossPingPongID_);
        crossPingPongID_ = (crossPingPongID_ + 1UL) & 1UL;
    }
    AivPostProcess(mmAddUb);
    vecQueAdd_.FreeTensor(mmAddUb);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AivPostProcess(LocalTensor<l0cDtype> &mmAddUb)
{
    if (block_->ubParams_.validM == 0 || block_->ubParams_.validN == 0) {
        return;
    }
    if constexpr (IsSameType<cType, l0cDtype>::value) {
        NotifyV2Mte3(0);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(block_->ubParams_.validM),
            static_cast<uint32_t>(block_->ubParams_.validN * sizeof(cType)),
            static_cast<int64_t>((block_->ubParams_.singleN - block_->ubParams_.validN) * sizeof(cType) /
                                 static_cast<uint64_t>(DATA_BLOCK)),
            static_cast<int64_t>((matmulTiling_->N - block_->ubParams_.validN) * sizeof(cType)), 0};
        DataCopyPad<cType>(this->cGlobal_[block_->ubParams_.offsetC], mmAddUb, copyParams);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE32V_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE32V_);
    } else {
        auto mmResPing = vecQueOut_.template AllocTensor<cType>();
        auto mmResPong = vecQueOut_.template AllocTensor<cType>();
        uint16_t mainM = DequantBmm::CeilDiv(block_->ubParams_.validM, 2UL); // 输出分两次搬运到GM
        AscendC::PipeBarrier<PIPE_V>();
        Cast(mmResPing, mmAddUb, RoundMode::CAST_RINT, mainM * block_->ubParams_.singleN);
        NotifyV2Mte3(0);
        DataCopyExtParams copyParams{
            mainM, static_cast<uint32_t>((block_->ubParams_.validN) * sizeof(cType)),
            static_cast<int64_t>((block_->ubParams_.singleN - block_->ubParams_.validN) * sizeof(cType) /
                                 static_cast<uint64_t>(DATA_BLOCK)),
            static_cast<int64_t>((matmulTiling_->N - block_->ubParams_.validN) * sizeof(cType)), 0};
        DataCopyPad<cType>(cGlobal_[block_->ubParams_.offsetC], mmResPing, copyParams);
        if (block_->ubParams_.validM - mainM != 0) {
            AscendC::PipeBarrier<PIPE_V>();
            Cast(mmResPong, mmAddUb[mainM * block_->ubParams_.singleN], RoundMode::CAST_RINT,
                 (block_->ubParams_.validM - mainM) * block_->ubParams_.singleN);
            NotifyV2Mte3(1);
            DataCopyExtParams copyParams1{
                static_cast<uint16_t>(block_->ubParams_.validM - mainM),
                static_cast<uint32_t>((block_->ubParams_.validN) * sizeof(cType)),
                static_cast<int64_t>((block_->ubParams_.singleN - block_->ubParams_.validN) * sizeof(cType) /
                                     static_cast<uint64_t>(DATA_BLOCK)),
                static_cast<int64_t>((matmulTiling_->N - block_->ubParams_.validN) * sizeof(cType)), 0};
            DataCopyPad<cType>(cGlobal_[block_->ubParams_.offsetC + mainM * matmulTiling_->N], mmResPong, copyParams1);
        }
        vecQueOut_.FreeTensor(mmResPing);
        vecQueOut_.FreeTensor(mmResPong);
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline scaleType MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::GetScaleX2(GM_ADDR scaleX2AddrOff,
                                                                                    uint64_t kOffset)
{
    if constexpr (bTrans) {
        return *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset));
    } else {
        return *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset * scaleN_));
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline uint64_t MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::GetScaleX1Offset(const uint64_t scaleOffsetX1,
                                                                                         uint64_t kOffset)
{
    if constexpr (aTrans) {
        return scaleOffsetX1 + kOffset * scaleM_;
    } else {
        return scaleOffsetX1 + kOffset;
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessKBlock(
    LocalTensor<l0cDtype> &mmAddUb, LocalTensor<ptScaleType> &mmPertokenScaleUb, LocalTensor<scaleType> &mmMuledScaleUb,
    const uint64_t scaleOffsetX1, GM_ADDR scaleX2AddrOff, uint64_t kb, uint64_t kOffset, uint64_t alignedM)
{
    // Get scale values
    scaleType scaleX2 = GetScaleX2(scaleX2AddrOff, kOffset);
    uint64_t scaleX1GmOffset = GetScaleX1Offset(scaleOffsetX1, kOffset);
    uint64_t pertokenIdx = kb / matmulTiling_->baseK % kCopySizeX1Scale_;

    // Handle pertoken scale updates
    if (pertokenIdx == 0 && pertokenRoundIdx_ > 0) {
        UpdatePertokenScale(mmPertokenScaleUb, scaleX1GmOffset, kb);
    }

    // Perform multiplication and accumulation
    AscendC::Muls(mmMuledScaleUb, mmPertokenScaleUb[pertokenIdx * alignedM], scaleX2, block_->ubParams_.validM);

    // Update ping-pong state if needed
    if (kb / matmulTiling_->baseK % kCopySizeX1Scale_ == kCopySizeX1Scale_ - 1 ||
        kb + matmulTiling_->baseK >= matmulTiling_->Ka) {
        UpdatePingPongState();
    }

    // Perform tensor operations
    PerformTensorOps(mmAddUb, mmMuledScaleUb, kb);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void
MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::PerformTensorOps(LocalTensor<l0cDtype> &mmAddUb,
                                                              LocalTensor<scaleType> &mmMuledScaleUb, uint64_t kb)
{
    uint64_t mmMuledScaleUbAddr = reinterpret_cast<uint64_t>(mmMuledScaleUb.GetPhyAddr());
    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + crossPingPongID_);
    auto mmUbInput = crossPingPongID_ == 0 ? mmResPing_ : mmResPong_;

    uint64_t mmAddUbAddr = reinterpret_cast<uint64_t>(mmAddUb.GetPhyAddr());
    uint64_t mmUbInputAddr = reinterpret_cast<uint64_t>(mmUbInput.GetPhyAddr());
    AscendC::PipeBarrier<PIPE_V>();

    if (kb == 0) {
        AivPerTensor<true>((__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr,
                           (__ubuf__ ptScaleType *)mmMuledScaleUbAddr, block_->ubParams_.validM,
                           block_->ubParams_.validN, block_->ubParams_.singleN);
    } else {
        AivPerTensor<false>((__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr,
                            (__ubuf__ ptScaleType *)mmMuledScaleUbAddr, block_->ubParams_.validM,
                            block_->ubParams_.validN, block_->ubParams_.singleN);
    }

    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + crossPingPongID_);
    crossPingPongID_ = (crossPingPongID_ + 1UL) & 1UL;
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::UpdatePingPongState()
{
    SetFlag<HardEvent::V_MTE2>(eventIdV2Mte2[pertokenPingPongID_]);
    pertokenPingPongID_ ^= 1;
    pertokenRoundIdx_++;
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void
MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::UpdatePertokenScale(LocalTensor<ptScaleType> &mmPertokenScaleUb,
                                                                 uint64_t scaleX1GmOffset, uint64_t kb)
{
    if constexpr (aTrans) {
        paramsMain_.loopInfo.loopSize[1] =
            DequantBmm::Min(static_cast<uint64_t>(kCopySizeX1Scale_), scaleK_ - (kb / matmulTiling_->baseK));
    } else {
        paramsMain_.loopInfo.loopSize[0] =
            DequantBmm::Min(static_cast<uint64_t>(kCopySizeX1Scale_), scaleK_ - (kb / matmulTiling_->baseK));
    }

    if (pertokenRoundIdx_ > 1) {
        WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2[pertokenPingPongID_]);
    }

    mmPertokenScaleUb = pertokenPingPongID_ == 0 ? mmPertokenScaleUbPing_ : mmPertokenScaleUbPong_;
    static constexpr MultiCopyConfig config = {false};
    constexpr uint32_t copyDim = 2; // dimension of data to be copied is 2
    DataCopy<ptScaleType, copyDim, config>(mmPertokenScaleUb, pertokenScaleGlobal_[scaleX1GmOffset], paramsMain_);
    SetFlag<HardEvent::MTE2_V>(pertokenPingPongID_);
    WaitFlag<HardEvent::MTE2_V>(pertokenPingPongID_);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAivSingleKPertile()
{
    block_->UpdatePerBlockUBParam();
    uint64_t scaleOffsetX1 = block_->offset_.batchAOffset * scaleK_ * scaleM_;
    uint64_t scaleOffsetX2 = block_->offset_.batchBOffset * scaleK_ * scaleN_;

    if constexpr (aTrans) {
        scaleOffsetX1 += block_->ubParams_.offsetScaleM;
    } else {
        scaleOffsetX1 += block_->ubParams_.offsetScaleM * scaleK_;
    }
    if constexpr (bTrans) {
        scaleOffsetX2 += block_->ubParams_.offsetScaleN * scaleK_;
    } else {
        scaleOffsetX2 += block_->ubParams_.offsetScaleN;
    }
    GM_ADDR scaleX2AddrOff = scaleX2Addr_ + sizeof(scaleType) * (scaleOffsetX2);
    if (scaleK_ == 16 && !aTrans) { // for unknown reason, when scaleK is 16, nddma will cause low mte2 bandwith
        // switch nddma k copy size to 2
        kCopySizeX1Scale_ = 2;
    }
    constexpr uint32_t copyDim = 2; // dimension of data to be copied is 2
    MultiCopyLoopInfo<copyDim> loopInfo;
    // Align and pad to 32 bytes in every dst row
    uint64_t alignedM = DequantBmm::Align(block_->ubParams_.validM, static_cast<uint64_t>(32 / sizeof(ptScaleType)));
    loopInfo.loopRpSize[0] = alignedM - block_->ubParams_.validM;
    if constexpr (aTrans) {
        loopInfo.loopSize[0] = block_->ubParams_.validM;
        loopInfo.loopSize[1] = kCopySizeX1Scale_;
        loopInfo.loopSrcStride[0] = 1;
        loopInfo.loopSrcStride[1] = scaleM_;
        loopInfo.loopDstStride[0] = 1;
        loopInfo.loopDstStride[1] = alignedM;
    } else {
        loopInfo.loopSize[0] = kCopySizeX1Scale_;
        loopInfo.loopSize[1] = block_->ubParams_.validM;
        loopInfo.loopSrcStride[0] = 1;
        loopInfo.loopSrcStride[1] = scaleK_;
        loopInfo.loopDstStride[0] = alignedM;
        loopInfo.loopDstStride[1] = 1;
    }
    static constexpr MultiCopyConfig config = {false};
    ptScaleType constValue = 0;
    paramsMain_ = {loopInfo, constValue};
    LocalTensor<l0cDtype> mmAddUb = vecQueAdd_.template AllocTensor<l0cDtype>();
    mmPertokenScaleUbPing_ = vecQuePertokenScale_.template AllocTensor<ptScaleType>();
    if (pertokenRoundIdx_ == 0) {
        if constexpr (aTrans) {
            paramsMain_.loopInfo.loopSize[1] = DequantBmm::Min(static_cast<uint64_t>(kCopySizeX1Scale_), scaleK_);
        } else {
            paramsMain_.loopInfo.loopSize[0] = DequantBmm::Min(static_cast<uint64_t>(kCopySizeX1Scale_), scaleK_);
        }
        DataCopy<ptScaleType, copyDim, config>(mmPertokenScaleUbPing_, pertokenScaleGlobal_[scaleOffsetX1],
                                               paramsMain_);
        SetFlag<HardEvent::MTE2_V>(pertokenPingPongID_);
        WaitFlag<HardEvent::MTE2_V>(pertokenPingPongID_);
    }
    mmPertokenScaleUbPong_ = vecQuePertokenScale_.template AllocTensor<ptScaleType>();
    auto mmPertokenScaleUb = mmPertokenScaleUbPing_;
    LocalTensor<scaleType> mmMuledScaleUb = vecQueMuledScale_.template AllocTensor<scaleType>();
    for (uint64_t kb = 0, kOffset = 0; kb < matmulTiling_->Ka; kb += matmulTiling_->baseK, kOffset++) {
        ProcessKBlock(mmAddUb, mmPertokenScaleUb, mmMuledScaleUb, scaleOffsetX1, scaleX2AddrOff, kb, kOffset, alignedM);
    }
    vecQuePertokenScale_.FreeTensor(mmPertokenScaleUbPing_);
    vecQuePertokenScale_.FreeTensor(mmPertokenScaleUbPong_);
    vecQueMuledScale_.FreeTensor(mmMuledScaleUb);
    AivPostProcess(mmAddUb);
    vecQueAdd_.FreeTensor(mmAddUb);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::CopyInA2(LocalTensor<aType> &aL1,
                                                                             const uint64_t &mAL1Offset,
                                                                             const uint64_t &kAL1Offset,
                                                                             const uint64_t &kOffset,
                                                                             const bool &isTailAL1)
{
    uint64_t offsetAL1 = matmulParam_.CalcAL1Offset(mAL1Offset, kAL1Offset, isTailAL1);
    LocalTensor<aType> aL0 = inQueueTensorAL0.template AllocTensor<aType>();
    LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsA(loadData2dParams, kOffset, isTailAL1);
    LoadData(aL0, aL1[offsetAL1], loadData2dParams);
    inQueueTensorAL0.EnQue(aL0);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::CopyInB2(LocalTensor<bType> &bL1,
                                                                             const uint64_t &nBL1Offset,
                                                                             const uint64_t &kBL1Offset,
                                                                             const uint64_t &kOffset,
                                                                             const bool &isTailBL1)
{
    uint64_t offsetBL1 = matmulParam_.CalcBL1Offset(nBL1Offset, kBL1Offset, isTailBL1);
    LocalTensor<bType> bL0 = inQueueTensorBL0.template AllocTensor<bType>();
    LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsB(loadData2dParams, kOffset, isTailBL1);
    LoadData(bL0, bL1[offsetBL1], loadData2dParams);
    inQueueTensorBL0.EnQue(bL0);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::MmadBase(const uint64_t &kOffset)
{
    auto aL0 = inQueueTensorAL0.template DeQue<aType>();
    auto bL0 = inQueueTensorBL0.template DeQue<bType>();
    LocalTensor<l0cDtype> cL0 = inQueueTensorCL0.template AllocTensor<l0cDtype>();
    uint32_t mmadK = DequantBmm::Min(static_cast<uint64_t>(matmulTiling_->baseK), matmulTiling_->Ka - kOffset);
    MmadParams mmadParams;
    if constexpr (aTrans) {
        mmadParams.m = DequantBmm::Align(block_->params_.singleCoreM, static_cast<uint64_t>(DATA_BLOCK));
    } else {
        if (block_->params_.singleCoreM == 1) {
            mmadParams.m = DequantBmm::Align(block_->params_.singleCoreM, static_cast<uint64_t>(BMM_BLOCK_NUM));
        } else {
            mmadParams.m = block_->params_.singleCoreM;
        }
    }
    if constexpr (bTrans) {
        mmadParams.n = block_->params_.singleCoreN;
    } else {
        mmadParams.n = DequantBmm::Align(block_->params_.singleCoreN, static_cast<uint64_t>(DATA_BLOCK));
    }
    mmadParams.k = mmadK;
    Mmad(cL0, aL0, bL0, mmadParams);
    inQueueTensorCL0.EnQue(cL0);
    inQueueTensorAL0.FreeTensor(aL0);
    inQueueTensorBL0.FreeTensor(bL0);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ClearAL1Space()
{
    LocalTensor<aType> L1A0 = inQueueTensorAL1.template AllocTensor<aType>();
    LocalTensor<aType> L1A1 = inQueueTensorAL1.template AllocTensor<aType>();
    LocalTensor<uint16_t> L1A0Tmp = L1A0.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> L1A1Tmp = L1A1.template ReinterpretCast<uint16_t>();
    InitConstValue(L1A0Tmp, {1, aL1BlockNum_, 0, 0U});
    InitConstValue(L1A1Tmp, {1, aL1BlockNum_, 0, 0U});
    inQueueTensorAL1.FreeTensor(L1A0);
    inQueueTensorAL1.FreeTensor(L1A1);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::CopyInA1Nd2Nz(const uint64_t &kOffset,
                                                                                  const bool &isTailAL1)
{
    LocalTensor<aType> aL1 = inQueueTensorAL1.template AllocTensor<aType>();
    uint64_t offset = matmulParam_.CalcAGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamA(nd2nzParam, isTailAL1);
    DataCopy(aL1, aGlobal_[offset], nd2nzParam);
    inQueueTensorAL1.EnQue(aL1);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::CopyInB1Nd2Nz(const uint64_t &kOffset,
                                                                                  const bool &isTailBL1)
{
    LocalTensor<bType> bL1 = inQueueTensorBL1.template AllocTensor<bType>();
    uint64_t offset = matmulParam_.CalcBGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamB(nd2nzParam, isTailBL1);
    DataCopy(bL1, bGlobal_[offset], nd2nzParam);
    inQueueTensorBL1.EnQue(bL1);
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AivEnd()
{
    if ASCEND_IS_AIV {
        if (pertokenRoundIdx_ > 1) {
            WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2[pertokenPingPongID_]);
            WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2[pertokenPingPongID_ ^ 1]);
        } else if (pertokenRoundIdx_ == 1) {
            WaitFlag<HardEvent::V_MTE2>(eventIdV2Mte2[pertokenPingPongID_ ^ 1]);
        }
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(eventIdV2Mte2[0]);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(eventIdV2Mte2[1]);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE32V_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(eventIdV2Mte3[0]);
        if constexpr (!IsSameType<cType, l0cDtype>::value) {
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(eventIdV2Mte3[1]);
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AicEnd()
{
    if ASCEND_IS_AIC {
        if (baseCount_ > 1) {
            WaitForVector(crossPingPongID_);
            crossPingPongID_ = crossPingPongID_ ^ 1;
            WaitForVector(crossPingPongID_);
        } else if (baseCount_ == 1) {
            crossPingPongID_ = 0;
            WaitForVector(crossPingPongID_);
        }
    }
}

} // namespace Mc2QuantBatchMatmulV3
#endif // QBMM_PERBLOCK_API_UTILS_H
