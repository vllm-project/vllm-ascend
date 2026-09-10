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
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_H
#define CHUNK_GATED_DELTA_RULE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
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


template <typename lowType, typename highType, typename stateType = lowType>
class CGDR {
public:
    static constexpr bool kStateIsFp32 = std::is_same_v<stateType, float>;
    using vInnerType = std::conditional_t<kStateIsFp32, float, lowType>;
    using StageOneMmVInner = StageOneMTT<vInnerType>;
    __aicore__ inline CGDR(TPipe *pipe, const ChunkGatedDeltaRuleTilingData *tilingData)
        : stageOneOp_(stage1MmBf16_, stage1MmVInner_)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
    };

    __aicore__ inline void InitMatmul()
    {
        if ASCEND_IS_AIC {
            stage1MmBf16_.Init(&tiling_->matmulTilingFp32, pipe_);
            if constexpr (kStateIsFp32) {
                stage1MmVInner_.Init(&tiling_->matmulTilingFp32C, pipe_);
            } else {
                stage1MmVInner_.Init(&tiling_->matmulTilingFp32, pipe_);
            }
            stage2MmBf16_.Init(&tiling_->matmulTilingFp32, pipe_);
            if constexpr (kStateIsFp32) {
                stage2MmFp32_.Init(&tiling_->matmulTilingFp32C, pipe_);
            }
            stage3Mm_.Init(&tiling_->matmulTilingFp32, pipe_);
        }
    }

    __aicore__ inline void InitMask()
    {
        if ASCEND_IS_AIV {
            // 初始化mask矩阵
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
            SetFlag<HardEvent::MTE3_MTE2>(eventID); // 这里必须同步, 否则会有提前拷出的问题
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

        dataSize = tiling_->b * tiling_->nv * tiling_->dv * tiling_->dk;
        initState_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(initParams.initState), dataSize);
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(initParams.finalState), dataSize);

        actualSeqLens_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initParams.seqlens), tiling_->b);

        uint64_t offset = 0;
        gCum_.SetGlobalBuffer(reinterpret_cast<__gm__ highType *>(user + offset));
        offset += sizeof(highType) * tiling_->nv * tiling_->maxGroupLength;

        kCumDecay_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        vInner_.SetGlobalBuffer(reinterpret_cast<__gm__ vInnerType *>(user + offset));
        offset += sizeof(vInnerType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        if constexpr (kStateIsFp32) {
            vInnerBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
            offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;
        }

        qPrime_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        attnInter_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dv;

        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->dk;

        qkt_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset));
        offset += sizeof(lowType) * tiling_->nv * tiling_->maxGroupLength * tiling_->chunkSize;

        if constexpr (kStateIsFp32) {
            uint64_t stateWkElemCnt = tiling_->nv * tiling_->dv * tiling_->dk;
            curStateBf16_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(user + offset), stateWkElemCnt);
            offset += sizeof(lowType) * stateWkElemCnt;
        }

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
                bool useInitialState = (pos == seqStart);
                auto curState = (useInitialState) ?
                                initState_[bid * tiling_->stateStride0] :
                                finalState_[bid * tiling_->nv * tiling_->dv * tiling_->dk];
                // compute this chunk group
                RunStage1(cg);
                SyncAll<false>();

                RunStage2(cg, curState,
                          finalState_[bid * tiling_->nv * tiling_->dv * tiling_->dk], useInitialState);
                SyncAll<false>();

                RunStage3(cg);
                SyncAll<false>();
            }
        }
    }

private:
    __aicore__ inline void RunStage1(const ChunkGroup& cg)
    {
        GDRStageOneInitParams<vInnerType> initStageOneParams {query_, key_, value_, beta_, g_,
                                                gCum_, kCumDecay_, vInner_, qPrime_, kg_, qkt_,
                                                stageWsAddr_, stageOneMask_, cg, gFlag_};
        stageOneOp_.Init(initStageOneParams, pipe_, tiling_);
        stageOneOp_.Process();
        pipe_->Reset();
    }

    template <typename sType>
    __aicore__ inline void RunStage2(ChunkGroup& cg, GlobalTensor<sType> stateIn,
                                     GlobalTensor<sType> stateOut, bool isFirstGroup)
    {
        Stage2<sType> stageTwoOp;
        StageTwoParams<sType> initStageTwoParams;
        initStageTwoParams.qPrime = qPrime_;
        initStageTwoParams.vInner = vInner_;
        if constexpr (std::is_same_v<sType, float>) {
            initStageTwoParams.vInnerBf16 = vInnerBf16_;
        }
        initStageTwoParams.gCum = gCum_;
        initStageTwoParams.kCumdecay = kCumDecay_;
        initStageTwoParams.kg = kg_;
        initStageTwoParams.out = out_[cg.startPos * tiling_->nv * tiling_->dv];
        initStageTwoParams.ws = stageWsAddr_;
        initStageTwoParams.mm1Bf16 = &stage2MmBf16_;
        initStageTwoParams.mm1Fp32 = &stage2MmFp32_;
        initStageTwoParams.pipe = pipe_;
        initStageTwoParams.cg = &cg;
        initStageTwoParams.Nv = tiling_->nv;
        initStageTwoParams.Nk = tiling_->nk;
        initStageTwoParams.Dv = tiling_->dv;
        initStageTwoParams.Dk = tiling_->dk;
        initStageTwoParams.stateStride1 = tiling_->stateStride1;
        initStageTwoParams.useInitialState = isFirstGroup;
        initStageTwoParams.gOptional = gFlag_;
        initStageTwoParams.isFirstGroup = isFirstGroup;
        initStageTwoParams.stateIn = stateIn;
        initStageTwoParams.stateOut = stateOut;
        if constexpr (std::is_same_v<sType, float>) {
            initStageTwoParams.curStateBf16 = curStateBf16_;
        } else {
            initStageTwoParams.curStateBf16 = stateIn;
        }
        stageTwoOp.Init(&initStageTwoParams, tiling_->aiCoreNum);
        stageTwoOp.Process();
        pipe_->Reset();
    }

    __aicore__ inline void RunStage3(ChunkGroup& cg)
    {
        if ASCEND_IS_AIC {
            stage3Mm_.Init(&tiling_->matmulTilingFp32, pipe_);
        }
        GlobalTensor<lowType> vInnerBf16;
        if constexpr (kStateIsFp32) {
            vInnerBf16 = vInnerBf16_;
        } else {
            vInnerBf16 = vInner_;
        }
        Stage3 stageThreeOp;
        StageThreeParams initStageThreeParams{
            qkt_, gCum_, vInnerBf16,
            stageThreeMask_[int(GetBlockIdx() / 2) * tiling_->chunkSize * tiling_->chunkSize],
            stageWsAddr_, out_[cg.startPos * tiling_->nv * tiling_->dv],
            &stage3Mm_, pipe_, &cg, tiling_->scale,
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
    GlobalTensor<stateType> finalState_;
    GlobalTensor<stateType> initState_;
    GlobalTensor<int32_t> actualSeqLens_;

    GlobalTensor<lowType> curStateBf16_;
    GlobalTensor<lowType> vInnerBf16_;   // (Nv, maxGroupLength, Dv) BF16 vInner for stage3 (FP32 path only)

    GlobalTensor<highType> gCum_;      // (Nv, maxGroupLength)
    GlobalTensor<lowType> kCumDecay_;     // (Nv, maxGroupLength, Dk)
    GlobalTensor<vInnerType> vInner_;     // (Nv, maxGroupLength, Dv) primary vInner
    GlobalTensor<lowType> qPrime_;        // (Nv, maxGroupLength, Dk)
    GlobalTensor<lowType> attnInter_;    // (Nv, maxGroupLength, Dv)
    GlobalTensor<lowType> kg_;           // (Nv, maxGroupLength, Dk)
    GlobalTensor<lowType> qkt_;          // (Nv, maxGroupLength, C)
    // mask矩阵
    GlobalTensor<highType> stageOneMask_;          // (Nv, maxGroupLength, C)
    GlobalTensor<highType> stageThreeMask_;          // (Nv, maxGroupLength, C)
    GM_ADDR stageWsAddr_;                 // temporary space addr for stages

    TBuf<TPosition::VECCALC> tmpBuff_;  // 构造mask矩阵

    // Matmul objects
    StageOneMTT<bfloat16_t> stage1MmBf16_;
    StageOneMmVInner stage1MmVInner_;
    StageTwoMT stage2MmBf16_;
    StageTwoMTFp32C stage2MmFp32_;
    StageThreeMT stage3Mm_;

    // Stage operators
    Stage1<kStateIsFp32> stageOneOp_;
    bool gFlag_ = false;
};

} // namespace ChunkGatedDeltaRule
#endif  // CHUNK_GATED_DELTA_RULE_H
