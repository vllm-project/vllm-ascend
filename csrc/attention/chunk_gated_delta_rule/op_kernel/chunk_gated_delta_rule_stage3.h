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
 * \file chunk_gated_delta_rule_stage3.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE3_H
#define CHUNK_GATED_DELTA_RULE_STAGE3_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using cT3 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageThreeMT = matmul::MatmulImpl<aT3, bT3, cT3>;

struct StageThreeParams {
    GlobalTensor<float> qkt;               // (Nv, Sp, C) fp32 输入
    GlobalTensor<float> gCumExp;           // (Nv, Sp)
    GlobalTensor<bfloat16_t> vInner;       // (Nv, Sp, Dv) v_new bf16
    GlobalTensor<float> maskTensor;
    GM_ADDR ws;
    GlobalTensor<float> attnInter;         // (Sp, Nv, Dv) fp32 中间 buffer
                                           // 调用前已由 stage2 写入 q_state(fp32);
                                           // stage3 cube 用 atomic add 把 cube_out(fp32) 累加上去,
                                           // 累加结果 = q_state + cube_out 仍为 fp32
    GlobalTensor<bfloat16_t> out;          // (Nv, Sp, Dv) 最终 bf16 输出
                                           // stage3 AIV 把 attnInter 累加结果 cast 成 bf16 后覆盖写回
    StageThreeMT *mm3;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        coreId_ = GetBlockIdx();
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ ;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        gOptional_ = sTP_->gOptional;
        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                               initParams->ws + workSpaceOffset +
                               coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));
        if ASCEND_IS_AIC {
            return;
        }
        coreId_ /= AIC_AIV_1_1;
        if (GetSubBlockIdx() == 1) {
            return;
        }
        // inQueue 用于读取 gCumExp(float) / qkt(float) / attnInter(float, q_state+cube_out 累加结果),
        // 最大尺寸取 max(chunkSize_*chunkSize_, chunkSize_*paddedDv_) * sizeof(float)
        uint64_t inQueueElems = static_cast<uint64_t>(chunkSize_) *
                                static_cast<uint64_t>(Std::max((uint64_t)chunkSize_, (uint64_t)paddedDv_));
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueElems * sizeof(float));
        // outQueue 用于写出 masked_qkt(bf16) 给 cube 做 A 矩阵, 或 cast 后的 out(bf16) 写回
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, inQueueElems * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * chunkSize_ * chunkSize_ * sizeof(float)));
        uint64_t buffOffset = 0;
        uint64_t tmpOffset = chunkSize_ * chunkSize_;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        maskBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);

        // 搬入mask
        DataCopyExtParams inParams{static_cast<uint16_t>(chunkSize_),
                                   static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(maskBuffer_, sTP_->maskTensor, inParams, copyPadParams);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    __aicore__ inline void Process()
    {
        int64_t totalChunks = Nv_ * chunkNum_;  // Nv Nc 融合
        int64_t chunksPerCore = (totalChunks + coreNum_ -1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int64_t startChunk = coreId_ * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;    // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    CalMaskedQKT(tmpGM_[coreId_ * chunkSize_ * chunkSize_], nvId, chunkPos);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                // 等待 cube mm3 完成 (AIC 端 SetFlag<PIPE_FIX>(0x3)):
                //   1) 保护下一轮 CalMaskedQKT 写 tmpGM_ 不与本轮 cube 读 tmpGM_ 冲突 (原有职责)
                //   2) 保证下面 CastAndStore 读 attnInter_ 时, cube 已经把 atomic add 结果写到 GM
                CrossCoreWaitFlag(0x3);
                // mm3 已通过 atomic add 把 cube_out(fp32) 累加到 attnInter_,
                // attnInter_ = q_state(stage2 写入) + cube_out (fp32 cube 累加)
                // AIV 这里把 attnInter_ 搬到 UB, cast 成 bf16 覆盖写回 out_
                // 对齐 chunk_o.py:137-138 的 fp32 累加 + 最终 cast 语义
                if (GetSubBlockIdx() == 0) {
                    CastAndStore(sTP_->attnInter[nvId * Dv_ + chunkPos * Nv_ * Dv_],
                                 sTP_->out[nvId * Dv_ + chunkPos * Nv_ * Dv_]);
                }
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(0x4);
                AICProcess(tmpGM_[coreId_ * chunkSize_ * chunkSize_],
                           sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                           sTP_->attnInter[nvId * Dv_ + chunkPos * Nv_ * Dv_]);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
            }
        }
    }

    __aicore__ inline void CalMaskedQKT(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        // chunkSize 大小进行自动补齐
        if (gOptional_) {
            AlignedCopyIn(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_);  // 自动补齐
            auto g_cum = inQueue_.DeQue<float>();
            const uint32_t srcShape1[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(1)};
            const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(chunkSize_)};
            const uint32_t dstShape[] = {static_cast<uint32_t>(chunkSize_),
                                         static_cast<uint32_t>(chunkSize_)};
            Broadcast<float, BROADCAST_AXIS, 1>(tmpBuffer1_, g_cum, dstShape, srcShape1);
            Broadcast<float, BROADCAST_AXIS, 0>(tmpBuffer2_, g_cum, dstShape, srcShape2);
            PipeBarrier<PIPE_V>();
            Sub(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            Mul(tmpBuffer1_, tmpBuffer1_, maskBuffer_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            Exp<float, 0, true>(tmpBuffer1_, tmpBuffer1_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(g_cum);
        } else {
            Duplicate(tmpBuffer1_, static_cast<float>(1.0f), curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
        }

        // qkt 已经是 fp32，无需 Cast
        AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<float>();
        auto scale_qkt = outQueue_.AllocTensor<bfloat16_t>();
        Muls(tmpBuffer2_, qkt, sTP_->scale, curChunkSize_ * chunkSize_);
        Mul(tmpBuffer2_, tmpBuffer2_, tmpBuffer1_, curChunkSize_ * chunkSize_);
        // mask
        Mul(tmpBuffer2_, tmpBuffer2_, maskBuffer_, curChunkSize_ * chunkSize_);
        Cast(scale_qkt, tmpBuffer2_, RoundMode::CAST_RINT, curChunkSize_ * chunkSize_);
        outQueue_.EnQue(scale_qkt);
        AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);
        inQueue_.FreeTensor(qkt);
    }

    __aicore__ inline void AICProcess(GlobalTensor<bfloat16_t> tmpGM,
                                      GlobalTensor<bfloat16_t> vInner,
                                      GlobalTensor<float> attnInter)
    {
        // masked_qkt @ v_new, 结果以 fp32 atomic add 累加到 attnInter
        // attnInter 中调用前已由 stage2 写入 q_state(fp32), 累加后 = q_state + cube_out (fp32 cube 累加)
        sTP_->mm3->SetOrgShape(curChunkSize_, Dv_, curChunkSize_, curChunkSize_, Nv_ * Dv_);    // MNK
        sTP_->mm3->SetSingleShape(curChunkSize_, Dv_, curChunkSize_); // SingleCoreMNK
        sTP_->mm3->SetTensorA(tmpGM);
        sTP_->mm3->SetTensorB(vInner);
        sTP_->mm3->IterateAll(attnInter, 1);
        sTP_->mm3->End();
    }

    // attnInter 此时已是 stage2 q_state + stage3 cube_out 的 fp32 累加结果 (cube atomic add 完成).
    // 这里把它搬到 UB, cast 成 bf16, 覆盖写回 out, 对齐 chunk_o.py:138
    //   tl.store(p_o, b_o.to(p_o.dtype.element_ty))    (fp32 -> bf16 cast 后 store)
    //
    // attnInter / out 形状为 (curChunkSize_, Dv_), 在外层 (Sp_, Nv_, Dv_) 中按 stride Nv_*Dv_ 排布.
    // 假设 Dv_ 已按 32B 对齐 (Dv_ 是 bf16 16 元素的整数倍, 同时也是 fp32 8 元素的整数倍),
    // 因此 UB 内每行无 padding, fp32 与 bf16 行布局一致, dstStride 均为 0.
    __aicore__ inline void CastAndStore(GlobalTensor<float> attnInter,
                                        GlobalTensor<bfloat16_t> out)
    {
        int32_t totalElems = static_cast<int32_t>(curChunkSize_) * static_cast<int32_t>(Dv_);

        // ---- 1) 读 fp32 attnInter (q_state + q_h 累加结果) 到 UB ----
        LocalTensor<float> inLocalFp32 = inQueue_.AllocTensor<float>();
        DataCopyExtParams inParamsFp32{static_cast<uint16_t>(curChunkSize_),
                                       static_cast<uint32_t>(Dv_ * sizeof(float)),
                                       static_cast<uint32_t>((Nv_ - 1) * Dv_ * sizeof(float)),
                                       static_cast<uint32_t>(0),
                                       0};
        DataCopyPadExtParams<float> padParamsFp32{false, 0, 0, 0};
        DataCopyPad(inLocalFp32, attnInter, inParamsFp32, padParamsFp32);
        inQueue_.EnQue(inLocalFp32);

        // ---- 2) fp32 -> bf16 cast ----
        auto inFp32 = inQueue_.DeQue<float>();
        auto outLocal = outQueue_.AllocTensor<bfloat16_t>();
        Cast(outLocal, inFp32, RoundMode::CAST_RINT, totalElems);
        outQueue_.EnQue(outLocal);
        inQueue_.FreeTensor(inFp32);

        // ---- 3) 写回 GM out (覆盖写) ----
        auto outDeq = outQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(curChunkSize_);
        outParams.blockLen = static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t));
        outParams.srcStride = static_cast<uint32_t>(0);
        outParams.dstStride = static_cast<uint32_t>((Nv_ - 1) * Dv_ * sizeof(bfloat16_t));
        DataCopyPad(out, outDeq, outParams);
        outQueue_.FreeTensor(outDeq);
    }

    template <typename inType>
    __aicore__ inline void AlignedCopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        // 非对齐拷入会自动对齐, 然后离散拷入UB
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType));
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(inType) / BLOCK_SIZE),
                                   0};
        DataCopyPadExtParams<inType> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void AlignedCopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(outType)) * (BLOCK_SIZE / sizeof(outType));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(outType) / BLOCK_SIZE);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<bfloat16_t> tmpGM_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    LocalTensor<float> maskBuffer_;
    int64_t seqLength_;
    int64_t Sp_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
    int64_t paddedDv_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int32_t curChunkSize_;
    int32_t chunkSize_;
    int32_t coreId_;
    bool gOptional_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE3_H