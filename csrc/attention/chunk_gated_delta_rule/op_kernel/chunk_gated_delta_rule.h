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
 * \file chunk_gated_delta_rule.h
 * \brief chunk 版 Gated Delta Rule 算子编排器。
 *        state(initial_state / final_state)为 float 输入，递推全程 float；
 *        stage2 matmul 为 bf16(A/B) × bf16(B) → float(C)，state / v_new 作 matmul 输入前 cast 回 bf16。
 *        stage1 中间结果已改为 float32（见 stage1 改造）；stage3 暂沿用 bf16 实现。
 */
#ifndef CHUNK_GATED_DELTA_RULE_H
#define CHUNK_GATED_DELTA_RULE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_stage1.h"
#include "chunk_gated_delta_rule_stage2.h"
#include "chunk_gated_delta_rule_stage3.h"
#include "chunk_gated_delta_rule_utils.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

struct CGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR seqlens;
    GM_ADDR gOptional;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename srcType, typename dstType>
__aicore__ inline void CopyCast(
    const GlobalTensor<srcType>& src,
    const GlobalTensor<dstType>& dst,
    TPipe* pipe,
    const int64_t totalDataCount,
    const RoundMode& roundMode)
{
    if ASCEND_IS_AIV {
        int64_t blkNum = GetBlockNum();
        int64_t blkId = GetBlockIdx();
        int64_t dataPerCore = (totalDataCount + blkNum - 1) / blkNum;
        int64_t startPos = dataPerCore * blkId;
        int64_t endPos = startPos + dataPerCore;
        if (startPos >= totalDataCount) {
            return;
        }
        if (endPos > totalDataCount) {
            endPos = totalDataCount;
        }
        TQue<QuePosition::VECIN, TQUE_DEPTH_TWO> inQueue;
        TQue<QuePosition::VECOUT, TQUE_DEPTH_TWO> outQueue;
        pipe->InitBuffer(inQueue, BUFFER_NUM_TWO, TILE_LEN * sizeof(srcType));   // use 2 buffer
        pipe->InitBuffer(outQueue, BUFFER_NUM_TWO, TILE_LEN * sizeof(dstType));  // use 2 buffer
        for (int64_t i = startPos; i < endPos; i += TILE_LEN) {
            uint32_t blockLen = i + TILE_LEN > endPos ? endPos - i : TILE_LEN;
            // copy in
            DataCopyExtParams inParams{1, static_cast<uint32_t>(blockLen * sizeof(srcType)), 0, 0, 0};
            DataCopyPadExtParams<srcType> inPadParams{false, 0, 0, 0};
            auto inLocal = inQueue.AllocTensor<srcType>();
            DataCopyPad(inLocal, src[i], inParams, inPadParams);
            inQueue.EnQue(inLocal);
            // cast
            auto stateIn = inQueue.DeQue<srcType>();
            auto stateOut = outQueue.AllocTensor<dstType>();
            Cast(stateOut, stateIn, roundMode, blockLen);
            outQueue.EnQue(stateOut);
            inQueue.FreeTensor(stateIn);
            // copy out
            auto outLocal = outQueue.DeQue<dstType>();
            DataCopyExtParams outParams{1, static_cast<uint32_t>(blockLen * sizeof(dstType)), 0, 0, 0};
            DataCopyPad(dst[i], outLocal, outParams);
            outQueue.FreeTensor(outLocal);
        }
        PipeBarrier<PIPE_V>();
    }
}


template <typename lowType, typename highType>
class CGDR {
public:
    __aicore__ inline CGDR(TPipe *pipe, const ChunkGatedDeltaRuleTilingData *tilingData)
        : stageOneOp_(s1MmBf16ToFp32_, s1MmFp32_, s1MmFp32ToBf16_)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
    };

    __aicore__ inline void InitMatmul()
    {
        if ASCEND_IS_AIC {
            s1MmBf16ToFp32_.Init(&tiling_->matmulTilingBf16ToFp32, pipe_);
            s1MmFp32_.Init(&tiling_->matmulTilingFp32ToFp32, pipe_);
            s1MmFp32ToBf16_.Init(&tiling_->matmulTilingFp32ToBf16, pipe_);
            stage3MT_.Init(&tiling_->matmulTilingFp32, pipe_);
            stage2FloatMT_.Init(&tiling_->matmulTilingStage2, pipe_);
        }
    }

    __aicore__ inline void InitMask()
    {
        if ASCEND_IS_AIV {
            uint32_t cBlockSize = tiling_->chunkSize * tiling_->chunkSize;
            pipe_->InitBuffer(tmpBuff_, cBlockSize * sizeof(float));
            auto cCFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(cBlockSize), 0);
            Duplicate<float>(cCFloat_, 0, tiling_->chunkSize);
            int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(1);
            copyParams.blockLen = static_cast<uint32_t>(tiling_->chunkSize * sizeof(float));
            copyParams.srcStride = static_cast<uint32_t>(0);
            copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(float));
            for (int i = 0; i < tiling_->chunkSize; ++i) {
                DataCopyPad(stageOneMask_[GetBlockIdx() * cBlockSize + i * tiling_->chunkSize], cCFloat_, copyParams);
                eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(eventID);
                WaitFlag<HardEvent::MTE3_S>(eventID);
                cCFloat_.SetValue(i, 1);
                eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventID);
                WaitFlag<HardEvent::S_MTE3>(eventID);
                DataCopyPad(stageThreeMask_[GetBlockIdx() * cBlockSize + i * tiling_->chunkSize], cCFloat_, copyParams);
            }
            eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            pipe_->Reset();
        }
    }

    __aicore__ inline void Init(const CGDRInitParams &initParams, GM_ADDR user)
    {
        uint64_t dataSize = tiling_->t * tiling_->nk * tiling_->dk;
        query_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.query), dataSize);
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.key), dataSize);

        dataSize = tiling_->t * tiling_->nv * tiling_->dv;
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.value), dataSize);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.attnOut), dataSize);

        dataSize = tiling_->t * tiling_->nv;
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.beta), dataSize);
        if (initParams.gOptional != nullptr) {
            g_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(initParams.gOptional), dataSize);
            gFlag_ = true;
        }

        // float 路径：state I/O 为 highType(float)
        dataSize = tiling_->b * tiling_->nv * tiling_->dv * tiling_->dk;
        initStateFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(initParams.initState), dataSize);
        finalStateFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(initParams.finalState), dataSize);

        actualSeqLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.seqlens), tiling_->b);

        uint64_t offset = 0;
        gCum_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength;

        kCumDecay_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        // vInner: stage1 改为 float32 输出，故按 highType 预留区间；
        // bf16 视图供 stage3 使用，float 视图供 stage1 输出和 stage2 输入。
        // vInner_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        vInnerFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        qPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        // attnInter: bf16 视图供 stage3 使用，float 视图供 stage2 写入
        attnInter_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        // attnInterFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        // qkt: stage1 改为 float32 输出，故按 highType 预留区间；bf16 视图供 stage3 使用
        qkt_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        // qktHigh_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength * tiling_->chunkSize;

        highState_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->b * tiling_->nv * tiling_->dv * tiling_->dk;

        bf16State_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->dv * tiling_->dk;

        bf16VNew_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        stageOneMask_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->chunkSize * tiling_->chunkSize * tiling_->aiCoreNum * TASK_RATIO;

        stageThreeMask_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->chunkSize * tiling_->chunkSize * tiling_->aiCoreNum * TASK_RATIO;

        stageWsAddr_ = user + offset;

        InitMatmul();
        InitMask();
    }

    __aicore__ inline void Process()
    {
        SyncAll<false>();
        int64_t seqStart = 0;
        int64_t seqEnd = 0;
        ChunkGroup cg;
        cg.chunkSize = tiling_->chunkSize;
        for (int64_t bid = 0; bid < tiling_->b; bid++) {
            int32_t length = actualSeqLens_.GetValue(bid);
            seqStart = seqEnd;
            seqEnd = seqStart + (int64_t)length;
            for (int64_t pos = seqStart; pos < seqEnd; pos += tiling_->maxGroupLength) {
                cg.startPos = pos;
                if (pos + tiling_->maxGroupLength > seqEnd) {
                    cg.length = seqEnd - pos;
                } else {
                    cg.length = tiling_->maxGroupLength;
                }
                // float 路径：state 为 highType(float)
                auto curState = (pos == seqStart) ?
                                initStateFloat_[bid * tiling_->nv * tiling_->dv * tiling_->dk] :
                                finalStateFloat_[bid * tiling_->nv * tiling_->dv * tiling_->dk];
                RunStage1(cg);
                SyncAll<false>();

                RunStage2Float(cg, curState,
                               finalStateFloat_[bid * tiling_->nv * tiling_->dv * tiling_->dk]);
                SyncAll<false>();

                RunStage3(cg);
                SyncAll<false>();
            }
        }
    }

private:
    __aicore__ inline void RunStage1(const ChunkGroup& cg)
    {
        GDRStageOneInitParams initStageOneParams {query_, key_, value_, beta_, g_,
                                                gCum_, kCumDecay_, vInnerFloat_, qPrime_, kg_, qkt_,
                                                stageWsAddr_, stageOneMask_, cg, gFlag_};
        stageOneOp_.Init(initStageOneParams, pipe_, tiling_);
        stageOneOp_.Process();
        pipe_->Reset();
    }

    __aicore__ inline void RunStage2Float(ChunkGroup& cg, GlobalTensor<highType> stateIn,
                                          GlobalTensor<highType> stateOut)
    {
        Stage2Float stageTwoFloatOp;
        StageTwoFloatParams initStageTwoParams{
            qPrime_, vInnerFloat_, gCum_, kCumDecay_, stateIn, stateOut, kg_,
            attnInter_, bf16State_, bf16VNew_, stageWsAddr_, &stage2FloatMT_, pipe_, &cg,
            tiling_->nv, tiling_->nk, tiling_->dv, tiling_->dk, gFlag_};
        stageTwoFloatOp.Init(&initStageTwoParams, tiling_->aiCoreNum);
        stageTwoFloatOp.Process();
        pipe_->Reset();
    }

    __aicore__ inline void RunStage3(ChunkGroup& cg)
    {
        Stage3 stageThreeOp;
        StageThreeParams initStageThreeParams{
            qkt_, gCum_, bf16VNew_,
            stageThreeMask_[int(GetBlockIdx() / 2) * tiling_->chunkSize * tiling_->chunkSize],
            stageWsAddr_,
            attnInter_,
            out_[cg.startPos * tiling_->nv * tiling_->dv],
            &stage3MT_, pipe_, &cg, tiling_->scale,
            tiling_->nv, tiling_->nk, tiling_->dv, tiling_->dk, gFlag_};
        stageThreeOp.Init(&initStageThreeParams, tiling_->aiCoreNum);
        stageThreeOp.Process();
        pipe_->Reset();
    }

private:
    TPipe *pipe_;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    GlobalTensor<lowType> query_;
    GlobalTensor<lowType> key_;
    GlobalTensor<lowType> value_;
    GlobalTensor<lowType> beta_;
    GlobalTensor<highType> g_;
    GlobalTensor<lowType> out_;
    float scale_;
    GlobalTensor<highType> finalStateFloat_;   // float 路径：state I/O 为 highType(float)
    GlobalTensor<highType> initStateFloat_;    // float 路径：state I/O 为 highType(float)
    GlobalTensor<int32_t> actualSeqLens_;

    GlobalTensor<highType> gCum_;        // (Nv, maxGroupLength)
    GlobalTensor<lowType> kCumDecay_;    // (Nv, maxGroupLength, Dk)
    GlobalTensor<lowType> vInner_;       // (Nv, maxGroupLength, Dv) bf16 视图(stage3)
    GlobalTensor<highType> vInnerFloat_; // (Nv, maxGroupLength, Dv) float 视图(stage1 输出 / stage2 输入)
    GlobalTensor<lowType> qPrime_;       // (Nv, maxGroupLength, Dk)
    GlobalTensor<highType> attnInter_;    // (maxGroupLength, Nv, Dv) bf16 影子（stage3 用）
    // GlobalTensor<highType> attnInterFloat_; // (maxGroupLength, Nv, Dv) float 视图（stage2 写入目标）
    GlobalTensor<lowType> kg_;           // (Nv, maxGroupLength, Dk)
    GlobalTensor<highType> qkt_;          // (Nv, maxGroupLength, C) bf16 视图(stage3)
    // GlobalTensor<highType> qktHigh_;     // (Nv, maxGroupLength, C) float32 视图(stage1 输出)
    GlobalTensor<highType> highState_;
    GlobalTensor<lowType> bf16State_;    // float 路径：state 的 bf16 影子（matmul B 输入）
    GlobalTensor<lowType> bf16VNew_;     // float 路径：v_new 的 bf16 影子（matmul A 输入）
    // mask矩阵
    GlobalTensor<highType> stageOneMask_;          // (Nv, maxGroupLength, C)
    GlobalTensor<highType> stageThreeMask_;          // (Nv, maxGroupLength, C)
    GM_ADDR stageWsAddr_;                 // temporary space addr for stages

    TBuf<TPosition::VECCALC> tmpBuff_;  // 构造mask矩阵

    // Matmul objects
    StageOneMmBf16ToFp32 s1MmBf16ToFp32_;   // bf16,bf16 -> fp32 (kk / qk)
    StageOneMmFp32 s1MmFp32_;               // fp32,fp32 -> fp32 (求逆 / attn@v_beta)
    StageOneMmFp32ToBf16 s1MmFp32ToBf16_;   // fp32,fp32 -> bf16 (attn@k_cumdecay)
    StageThreeMT stage3MT_;
    StageTwoFloatMT stage2FloatMT_;  // float 路径 stage2 专用（bf16 A/B + float C）

    // Stage operators
    Stage1 stageOneOp_;
    bool gFlag_ = false;
};

} // namespace ChunkGatedDeltaRule
#endif  // CHUNK_GATED_DELTA_RULE_H
