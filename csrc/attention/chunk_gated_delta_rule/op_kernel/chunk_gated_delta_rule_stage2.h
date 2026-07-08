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
 * \brief float 路径 stage2：state 递推全程 float，matmul 为 bf16(A/B) × bf16(B) → float(C)。
 *        state / v_new 作 matmul 输入前先 cast 回 bf16（写入 bf16State_ / bf16VNew_）；
 *        attn_inter 以 float 写入 attnInter_（out_ 仍 bf16，由 stage3 汇总后写出）。
 *        本类为 stage2 唯一实现：state 恒 float32 输入，无 bf16 路径、无 tilingKey 分发。
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

// float 路径专用 matmul：A/B 为 bf16（isTrans 可设），C 为 float
using aT2F = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT2F = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using cT2F = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageTwoFloatMT = matmul::MatmulImpl<aT2F, bT2F, cT2F>;

struct StageTwoFloatParams {
    GlobalTensor<bfloat16_t> qPrime;     // (Nv, Sp, Dk)
    GlobalTensor<float> vInner;          // (Nv, Sp, Dv)  float v_new（CalVPrime atomic add 目标）
    GlobalTensor<float> gCum;            // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay;  // (Nv, Sp, Dk)
    GlobalTensor<float> curState;        // (Nv, Dv, Dk)  float
    GlobalTensor<float> finalState;      // (Nv, Dv, Dk)  float（CalStateNew atomic add 目标）
    GlobalTensor<bfloat16_t> kg;         // (Nv, Sp, Dk)
    GlobalTensor<float> attnInter;       // (Sp, Nv, Dv)  float（CalAttnInter 覆盖写目标，替代原 out 字段）
    GlobalTensor<bfloat16_t> bf16State;  // (Nv, Dv, Dk)  state 的 bf16 影子（CalVPrime/CalAttnInter 的 B 输入）
    GlobalTensor<bfloat16_t> bf16VNew;   // (Nv, Sp, Dv)  v_new 的 bf16 影子（CalStateNew 的 A 输入）
    GM_ADDR ws;
    StageTwoFloatMT *mm1;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage2Float {
public:
    __aicore__ inline void Init(StageTwoFloatParams *initParams, int32_t coreNum)
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
        // float 对齐粒度（BLOCK_SIZE/sizeof(float) = 8），用 BLOCK_FLOAT_NUM
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(float)) * (BLOCK_SIZE / sizeof(float));
        curChunkSize_ = chunkSize_;
        gOptional_ = sTP_->gOptional;
        InitLocalBuffers();
    }

    __aicore__ inline void InitLocalBuffers()
    {
        if ASCEND_IS_AIC {
            return;
        }
        // inQueue_ 同时承载 state CopyIn(Dv*curDk, float) 与 v_new CopyIn(chunkSize*paddedDv, float)
        uint64_t inElem = Std::max((uint64_t)Dv_ * curDk_, (uint64_t)chunkSize_ * paddedDv_);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inElem * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, Dv_ * curDk_ * sizeof(float));
        // bf16CastQueue_：存 state->bf16 / v_new->bf16 cast 结果供 CopyOut
        uint64_t bf16CastSize = Std::max((uint64_t)Dv_ * curDk_, (uint64_t)chunkSize_ * paddedDv_) * sizeof(bfloat16_t);
        pipe_->InitBuffer(bf16CastQueue_, BUFFER_NUM_ONE, bf16CastSize);
        pipe_->InitBuffer(tmpBuff_, (BLOCK_FLOAT_NUM + NUM_ONE) * sizeof(float));
        uint32_t buffOffset = 0;
        // 暂存所取的最后一位数
        lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), buffOffset);
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
                auto curState = (cId == 0) ? sTP_->curState[nvId * Dv_ * Dk_] : sTP_->finalState[nvId * Dv_ * Dk_];
                auto finalState = sTP_->finalState[nvId * Dv_ * Dk_];
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    if (GetSubBlockIdx() == 0) {
                        CopyIn(curState, Dv_, Dk_);
                        CalGCumExp(sTP_->gCum[nvId * Sp_ + length]);
                        CopyOutBf16State(sTP_->bf16State[nvId * Dv_ * Dk_]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  // 通知 AIC: bf16State_ 就绪
                    CrossCoreWaitFlag(0x2);                 // 等 AIC 读完 bf16State_ + CalVPrime 完成(v_new 就绪)
                    if (GetSubBlockIdx() == 0) {
                        CopyOutState(finalState);           // 写 state_old(float)
                        CastVNewToBf16(sTP_->vInner[nvId * Sp_ * Dv_ + length * Dv_],
                                       sTP_->bf16VNew[nvId * Sp_ * Dv_ + length * Dv_]);
                    }
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  // 通知 AIC: state_old 写好 + bf16VNew_ 就绪
                    CrossCoreWaitFlag(0x4);                 // 等 AIC CalStateNew 完成
                }
                if ASCEND_IS_AIC {
                    uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
                    uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
                    // attnInter 与 stage3 / out_ 共用 (Sp, Nv, Dv) 布局，
                    // token stride = Nv*Dv，head stride = Dv；此处按 token 起点 + head 偏移寻址。
                    uint64_t attn_inter_offset = nvId * Dv_ + length * Nv_ * Dv_;
                    CrossCoreWaitFlag(0x6);                 // 等 AIV cast bf16State_
                    CalVPrime(sTP_->kCumdecay[mm_offset0], sTP_->bf16State[nvId * Dv_ * Dk_],
                              sTP_->vInner[mm_offset1]);
                    CalAttnInter(sTP_->qPrime[mm_offset0], sTP_->bf16State[nvId * Dv_ * Dk_],
                                 sTP_->attnInter[attn_inter_offset]);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);   // 通知 AIV: bf16State_ 读完 + v_new 就绪
                    CrossCoreWaitFlag(0x5);                 // 等 AIV 写 state_old + cast bf16VNew_
                    CalStateNew(sTP_->bf16VNew[mm_offset1], sTP_->kg[mm_offset0], finalState);
                    SetFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    WaitFlag<HardEvent::FIX_MTE2>(FIX_MTE2_EVENT);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(0x4);   // 通知 AIV: state_new 完成
                }
            }
        }
    }

    __aicore__ inline void CalGCumExp(GlobalTensor<float> gCum)
    {
        if (gOptional_) {
            // 刷新cache
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
        auto stateIn = inQueue_.DeQue<float>();
        auto stateOut = outQueue_.AllocTensor<float>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        // state_old = state * exp(g_cum[-1])，全程 float（不再 cast 回 bf16）
        Muls(stateOut, stateIn, tmpFloat, Dv_ * curDk_);
        // 同步把 float state cast 为 bf16 写到 bf16State_（供 CalVPrime/CalAttnInter 的 B 输入）
        auto bf16Out = bf16CastQueue_.AllocTensor<bfloat16_t>();
        Cast(bf16Out, stateIn, RoundMode::CAST_RINT, Dv_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(stateOut);
        bf16CastQueue_.EnQue(bf16Out);
        inQueue_.FreeTensor(stateIn);
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> qPrime,
                                        GlobalTensor<bfloat16_t> state,
                                        GlobalTensor<float> attnInter)
    {
        // 写入目标 attnInter_ 与 stage3 / out_ 共用 (Sp, Nv, Dv) 布局，
        // C 矩阵 N 维度的 GM 行步长 = Nv*Dv（每行跨过一个完整 token 的所有 head）
        sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_, Dk_, Nv_ * Dv_);   // MNKaKbN
        sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_);                // SingleCoreMNK
        sTP_->mm1->SetTensorA(qPrime, false);
        sTP_->mm1->SetTensorB(state, true);
        sTP_->mm1->IterateAll(attnInter);
        sTP_->mm1->End();
    }

    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> kCumdecay,
                                     GlobalTensor<bfloat16_t> state,
                                     GlobalTensor<float> vPrime)
    {
        // v_inner += k_cumdecay @ state.transpose(0, 1) -> float vInner_（atomic add）
        sTP_->mm1->SetOrgShape(curChunkSize_, Dv_, Dk_);    // MNK
        sTP_->mm1->SetSingleShape(curChunkSize_, Dv_, Dk_); // SingleCoreMNK
        sTP_->mm1->SetTensorA(kCumdecay, false);
        sTP_->mm1->SetTensorB(state, true);
        sTP_->mm1->IterateAll(vPrime, 1);
        sTP_->mm1->End();
    }

    __aicore__ inline void CalStateNew(GlobalTensor<bfloat16_t> vInner,
                                       GlobalTensor<bfloat16_t> kg,
                                       GlobalTensor<float> state)
    {
        // state_out = v_new.transpose(0, 1) @ kg -> float finalState（atomic add）
        sTP_->mm1->SetOrgShape(Dv_, Dk_, curChunkSize_);    // MNK
        sTP_->mm1->SetSingleShape(Dv_, Dk_, curChunkSize_); // SingleCoreMNK
        sTP_->mm1->SetTensorA(vInner, true);
        sTP_->mm1->SetTensorB(kg, false);
        sTP_->mm1->IterateAll(state, 1);
        sTP_->mm1->End();
    }

    template <typename inType>
    __aicore__ inline void CopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)), // 非对齐情况需要补0
                                   static_cast<uint32_t>(0),
                                   0, 0};
        int padding = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType)) - col;
        DataCopyPadExtParams<inType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOutState(GlobalTensor<float> stateNew)
    {
        CopyOut<float>(stateNew, Dv_, Dk_, false);
    }

    // 把 CalGCumExp 中 cast 出的 bf16 state 写到 bf16State_ GM
    __aicore__ inline void CopyOutBf16State(GlobalTensor<bfloat16_t> dst)
    {
        auto outLocal = bf16CastQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(Dv_);
        copyParams.blockLen = static_cast<uint32_t>(Dk_ * sizeof(bfloat16_t));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        DataCopyPad(dst, outLocal, copyParams);
        bf16CastQueue_.FreeTensor(outLocal);
    }

    // CopyIn<float> 当前 chunk v_new(=vInner_) -> Cast bf16 -> CopyOut bf16VNew_
    __aicore__ inline void CastVNewToBf16(GlobalTensor<float> vNew, GlobalTensor<bfloat16_t> bf16VNew)
    {
        CopyIn<float>(vNew, curChunkSize_, Dv_);
        auto vIn = inQueue_.DeQue<float>();
        auto bf16Out = bf16CastQueue_.AllocTensor<bfloat16_t>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        Cast(bf16Out, vIn, RoundMode::CAST_RINT, curChunkSize_ * paddedDv_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        bf16CastQueue_.EnQue(bf16Out);
        inQueue_.FreeTensor(vIn);
        // CopyOut bf16 -> bf16VNew_
        auto outLocal = bf16CastQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        copyParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t));
        copyParams.srcStride = static_cast<uint32_t>(0);
        copyParams.dstStride = static_cast<uint32_t>(0);
        DataCopyPad(bf16VNew, outLocal, copyParams);
        bf16CastQueue_.FreeTensor(outLocal);
    }

    template <typename outType>
    __aicore__ inline void CopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col, bool setAtomic = false)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>(0);
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
    StageTwoFloatParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> bf16CastQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> lastGCum_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int32_t curDk_;
    int32_t paddedDv_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    bool gOptional_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_H