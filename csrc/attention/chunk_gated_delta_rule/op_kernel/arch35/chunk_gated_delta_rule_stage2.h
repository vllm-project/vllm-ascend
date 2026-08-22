/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file chunk_gated_delta_rule_stage2.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
template <typename cType>
using StageTwoMTT = matmul::MatmulImpl<aT2, bT2, MatmulType<TPosition::GM, CubeFormat::ND, cType>>;
using StageTwoMT = StageTwoMTT<bfloat16_t>;
using StageTwoMTFp32C = StageTwoMTT<float>;

template <typename stateType>
struct StageTwoParams {
    GlobalTensor<bfloat16_t> qPrime;        // (Nv, Sp, Dk)
    GlobalTensor<stateType> vInner;         // (Nv, Sp, Dv)
    GlobalTensor<bfloat16_t> vInnerBf16;    // (Nv, Sp, Dv) BF16 vInner for stage3/CalStateNew (FP32 path)
    GlobalTensor<float> gCum;               // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay;     // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> curStateBf16;  // (Nv, Dv, Dk) BF16 state for AIC
    GlobalTensor<stateType> stateIn;        // (Nv, Dv, Dk) state input
    GlobalTensor<stateType> stateOut;       // (Nv, Dv, Dk) state output
    GlobalTensor<bfloat16_t> kg;
    GlobalTensor<bfloat16_t> out;
    GM_ADDR ws;
    StageTwoMT *mm1Bf16;
    StageTwoMTFp32C *mm1Fp32;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    int64_t stateStride1;
    bool useInitialState;
    bool gOptional;
    bool isFirstGroup;
};

template <typename stateType = bfloat16_t>
class Stage2 {
public:
    static constexpr bool kIsFp32 = std::is_same_v<stateType, float>;

    __aicore__ inline void Init(StageTwoParams<stateType> *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curChunkSize_ = chunkSize_;
        useInitialState_ = sTP_->useInitialState;
        stateStride1_ = (useInitialState_) ? sTP_->stateStride1 : Dv_ * Dk_;
        gOptional_ = sTP_->gOptional;
        isFirstGroup_ = sTP_->isFirstGroup;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (kIsFp32) {
            uint64_t inSize = Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ * sizeof(float);
            uint64_t vInnerCastInSize = chunkSize_ * Dv_ * sizeof(float);
            uint64_t vInnerCastOutSize = chunkSize_ * Dv_ * sizeof(bfloat16_t);
            pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, Std::max(inSize, vInnerCastInSize));
            uint64_t outSize = Std::max(inSize, vInnerCastOutSize);
            pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outSize);
            pipe_->InitBuffer(tmpBuff_, (Dv_ * curDk_ + BLOCK_FLOAT_NUM) * sizeof(float));
            uint32_t buffOffset = 0;
            stateFp32Ub_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(Dv_ * curDk_), buffOffset);
            buffOffset += Ceil(Dv_ * curDk_ * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
            lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), buffOffset);
        } else {
            pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, Std::max((uint64_t)chunkSize_,
                                                                 (uint64_t)Dv_) * curDk_ * sizeof(bfloat16_t));
            uint64_t outQueueSize = Std::max((uint64_t)chunkSize_ * chunkSize_ * sizeof(bfloat16_t),
                                             (uint64_t)Dv_ * curDk_ * sizeof(bfloat16_t));
            pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outQueueSize);
            pipe_->InitBuffer(tmpBuff_, (Std::max((uint64_t)chunkSize_, (uint64_t)Dv_) * curDk_ +
                                         BLOCK_FLOAT_NUM) *sizeof(float));
            uint32_t buffOffset = 0;
            tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(Dv_ * curDk_), buffOffset);
            buffOffset += Ceil(Dv_ * curDk_ * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
            lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), buffOffset);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t coreId = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId /= AIC_AIV_1_1;
        }
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            curChunkSize_ = chunkSize_;
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if constexpr (kIsFp32) {
                        ProcessAivFp32(nvId, cId, length);
                    } else {
                        ProcessAivBf16(nvId, cId, length);
                    }
                }
                if ASCEND_IS_AIC {
                    ProcessAic(nvId, cId, length);
                }
            }
        }
    }

    __aicore__ inline void ProcessAivFp32(int64_t nvId, int64_t cId, int64_t length)
    {
        if (GetSubBlockIdx() == 0) {
            if (isFirstGroup_ && cId == 0) {
                CopyIn<float>(sTP_->stateIn[nvId * stateStride1_], Dv_, Dk_);
            } else {
                CopyIn<float>(sTP_->stateOut[nvId * Dv_ * Dk_], Dv_, Dk_);
            }
            auto stateIn = inQueue_.DeQue<float>();
            DataCopy(stateFp32Ub_, stateIn, Dv_ * curDk_);
            inQueue_.FreeTensor(stateIn);
            SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
            auto bf16State = outQueue_.AllocTensor<bfloat16_t>();
            Cast(bf16State, stateFp32Ub_, RoundMode::CAST_RINT, Dv_ * curDk_);
            outQueue_.EnQue(bf16State);
            CopyOut<bfloat16_t>(sTP_->curStateBf16[nvId * Dv_ * Dk_], Dv_, Dk_, false, curDk_);
            CalGCumExpFp32(sTP_->gCum[nvId * Sp_ + length]);
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            PipeBarrier<PIPE_MTE3>();
            auto fp32Local = outQueue_.AllocTensor<float>();
            DataCopy(fp32Local, stateFp32Ub_, Dv_ * curDk_);
            outQueue_.EnQue(fp32Local);
            CopyOut<float>(sTP_->stateOut[nvId * Dv_ * Dk_], Dv_, Dk_, false, curDk_);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x3);
        CrossCoreWaitFlag(0x2);
        if (GetSubBlockIdx() == 0) {
            PipeBarrier<PIPE_MTE3>();
            uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
            CastGmToGm<float, bfloat16_t>(sTP_->vInner[mm_offset1],
                                          sTP_->vInnerBf16[mm_offset1],
                                          curChunkSize_ * Dv_);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        CrossCoreWaitFlag(0x4);
    }

    __aicore__ inline void ProcessAivBf16(int64_t nvId, int64_t cId, int64_t length)
    {
        auto curState = (cId == 0) ? sTP_->curStateBf16[nvId * stateStride1_]
                                   : sTP_->stateOut[nvId * Dv_ * Dk_];
        auto finalState = sTP_->stateOut[nvId * Dv_ * Dk_];
        if (GetSubBlockIdx() == 0) {
            CopyIn(curState, Dv_, Dk_);
            CalGCumExpBf16(sTP_->gCum[nvId * Sp_ + length]);
        }
        CrossCoreWaitFlag(0x2);
        if (GetSubBlockIdx() == 0) {
            CopyOutState(finalState);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);
        CrossCoreWaitFlag(0x4);
    }

    __aicore__ inline void ProcessAic(int64_t nvId, int64_t cId, int64_t length)
    {
        uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
        uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
        auto stateOut = sTP_->stateOut[nvId * Dv_ * Dk_];
        GlobalTensor<bfloat16_t> stateForAic;
        if constexpr (kIsFp32) {
            CrossCoreWaitFlag(0x3);
            stateForAic = sTP_->curStateBf16[nvId * Dv_ * Dk_];
        } else {
            stateForAic = (cId == 0) ? sTP_->curStateBf16[nvId * stateStride1_]
                                     : sTP_->stateOut[nvId * Dv_ * Dk_];
        }
        CalVPrime(sTP_->kCumdecay[mm_offset0], stateForAic, sTP_->vInner[mm_offset1]);
        CalAttnInter(sTP_->qPrime[mm_offset0], stateForAic,
                     sTP_->out[nvId * Dv_ + cId * chunkSize_ * Nv_ * Dv_]);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);
        CrossCoreWaitFlag(0x5);
        if constexpr (kIsFp32) {
            CalStateNew(sTP_->vInnerBf16[mm_offset1],
                        sTP_->kg[mm_offset0], stateOut);
        } else {
            CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], stateOut);
        }
        SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
        WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);
    }

    __aicore__ inline void CalGCumExpFp32(GlobalTensor<float> gCum)
    {
        if (!gOptional_) {
            return;
        }
        DataCacheCleanAndInvalid<float,
                                 CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(gCum[curChunkSize_ - 1]);
        float gVal = gCum.GetValue(curChunkSize_ - 1);
        lastGCum_.SetValue(0, gVal);
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Exp<float, 0, true>(lastGCum_, lastGCum_, 1);
        float tmpFloat = lastGCum_.GetValue(0);
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Muls(stateFp32Ub_, stateFp32Ub_, tmpFloat, Dv_ * curDk_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalGCumExpBf16(GlobalTensor<float> gCum)
    {
        if (gOptional_) {
            DataCacheCleanAndInvalid<float,
                                     CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(gCum[curChunkSize_ - 1]);
            float tmpFloat = gCum.GetValue(curChunkSize_ - 1);
            lastGCum_.SetValue(0, tmpFloat);
            SetFlag<HardEvent::S_V>(S_V_EVENT);
            WaitFlag<HardEvent::S_V>(S_V_EVENT);
            Exp<float, 0, true>(lastGCum_, lastGCum_, 1);
        } else {
            lastGCum_.SetValue(0, 1.0f);
        }
        float tmpFloat = lastGCum_.GetValue(0);
        auto stateIn = inQueue_.DeQue<bfloat16_t>();
        auto stateOut = outQueue_.AllocTensor<bfloat16_t>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Cast(tmpBuffer1_, stateIn, RoundMode::CAST_NONE, Dv_ * curDk_);
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Muls(tmpBuffer1_, tmpBuffer1_, tmpFloat, Dv_ * curDk_);
        Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(stateOut);
        inQueue_.FreeTensor(stateIn);
    }

    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA,
                                     GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K, int32_t accum)
    {
        mm->SetOrgShape(M, N, K);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA,
                                     GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K,
                                     int64_t Ka, int64_t Kb, int32_t accum = 0)
    {
        mm->SetOrgShape(M, N, K, Ka, Kb);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> qPrime,
                                        GlobalTensor<bfloat16_t> state,
                                        GlobalTensor<bfloat16_t> out)
    {
        RunMatmul(sTP_->mm1Bf16, qPrime, false, state, true, out,
                  curChunkSize_, Dv_, Dk_, Dk_, Nv_ * Dv_);
    }

    template <typename vInnerType>
    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> kCumdecay,
                                     GlobalTensor<bfloat16_t> state,
                                     GlobalTensor<vInnerType> vPrime)
    {
        if constexpr (kIsFp32) {
            RunMatmul(sTP_->mm1Fp32, kCumdecay, false, state, true, vPrime,
                      curChunkSize_, Dv_, Dk_, 1);
        } else {
            RunMatmul(sTP_->mm1Bf16, kCumdecay, false, state, true, vPrime,
                      curChunkSize_, Dv_, Dk_, 1);
        }
    }

    template <typename stateOutType>
    __aicore__ inline void CalStateNew(GlobalTensor<bfloat16_t> vInner,
                                       GlobalTensor<bfloat16_t> kg,
                                       GlobalTensor<stateOutType> state)
    {
        if constexpr (kIsFp32) {
            RunMatmul(sTP_->mm1Fp32, vInner, true, kg, false, state,
                      Dv_, Dk_, curChunkSize_, 1);
        } else {
            RunMatmul(sTP_->mm1Bf16, vInner, true, kg, false, state,
                      Dv_, Dk_, curChunkSize_, 1);
        }
    }

    template <typename srcType, typename dstType>
    __aicore__ inline void CastGmToGm(GlobalTensor<srcType> srcGm, GlobalTensor<dstType> dstGm,
                                       int32_t count)
    {
        int32_t paddedCount = Ceil(count, BLOCK_SIZE / sizeof(dstType)) * (BLOCK_SIZE / sizeof(dstType));
        int32_t srcAlign = BLOCK_SIZE / sizeof(srcType);
        int32_t srcPadded = Ceil(count, srcAlign) * srcAlign;
        int32_t padding = srcPadded - count;

        auto srcLocal = inQueue_.AllocTensor<srcType>();
        if (paddedCount > srcPadded) {
            Duplicate<srcType>(srcLocal, static_cast<srcType>(0), paddedCount);
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
        }
        DataCopyExtParams inParams{1, static_cast<uint32_t>(count * sizeof(srcType)), 0, 0, 0};
        DataCopyPadExtParams<srcType> inPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(srcLocal, srcGm, inParams, inPadParams);
        inQueue_.EnQue(srcLocal);

        srcLocal = inQueue_.DeQue<srcType>();
        auto dstLocal = outQueue_.AllocTensor<dstType>();
        Cast(dstLocal, srcLocal, RoundMode::CAST_RINT, paddedCount);
        inQueue_.FreeTensor(srcLocal);
        outQueue_.EnQue(dstLocal);

        dstLocal = outQueue_.DeQue<dstType>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(count * sizeof(dstType)), 0, 0, 0};
        DataCopyPadExtParams<dstType> outPadParams{false, 0, 0, 0};
        DataCopyPad(dstGm, dstLocal, outParams);
        outQueue_.FreeTensor(dstLocal);
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        int padding;
        uint32_t dstStride = 0;
        if constexpr (kIsFp32 && std::is_same_v<inType, float>) {
            int32_t floatAligned = Ceil(col, BLOCK_FLOAT_NUM) * BLOCK_FLOAT_NUM;
            padding = floatAligned - col;
            if (floatAligned < curDk_) {
                uint32_t dataBlocks = static_cast<uint32_t>(floatAligned * sizeof(float) / BLOCK_SIZE);
                uint32_t rowBlocks = static_cast<uint32_t>(curDk_ * sizeof(float) / BLOCK_SIZE);
                dstStride = rowBlocks - dataBlocks;
                Duplicate<inType>(inLocal, static_cast<inType>(0), row * curDk_);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            }
        } else {
            padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        }
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   dstStride, 0};
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOutState(GlobalTensor<bfloat16_t> stateNew)
    {
        CopyOut<bfloat16_t>(stateNew, Dv_, Dk_, false);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col,
                                   bool setAtomic = false, int32_t srcCol = 0)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        if (srcCol > col) {
            uint32_t actualReadBytes = Ceil(col * sizeof(outType), BLOCK_SIZE) * BLOCK_SIZE;
            uint32_t srcRowBytes = srcCol * sizeof(outType);
            if (srcRowBytes >= actualReadBytes && (srcRowBytes - actualReadBytes) % BLOCK_SIZE == 0) {
                copyParams.srcStride = (srcRowBytes - actualReadBytes) / BLOCK_SIZE;
            } else {
                copyParams.srcStride = 0;
            }
        } else {
            copyParams.srcStride = static_cast<uint32_t>(0);
        }
        copyParams.dstStride = static_cast<uint32_t>(0);
        if (setAtomic) {
            SetAtomicAdd<outType>();
        }
        DataCopyPad(tmpGM, outLocal, copyParams);
        if (setAtomic) {
            SetAtomicNone();
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageTwoParams<stateType> *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> lastGCum_;
    LocalTensor<float> stateFp32Ub_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int64_t stateStride1_;
    bool useInitialState_;
    bool gOptional_;
    bool isFirstGroup_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_H
