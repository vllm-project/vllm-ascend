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
 * \file attn_res_fwd_reload.h
 * \brief TK-RELOAD: Softmax 前每行 GM 读 1 次（同趟 sumSq→invRms→score）；
 *        Softmax 后重搬 BF16→Cast（第 2 次）；合计每行 v 搬入 2 次；无 FP32-v Workspace
 *
 * 行序（golden）：n=0..N-1 ← block_residual[n]；n=N ← prefix_sum
 */
#ifndef ATTN_RES_FWD_RELOAD_H
#define ATTN_RES_FWD_RELOAD_H

#include "kernel_operator.h"
#include "../attn_res_fwd_tiling_data.h"
#include "reduce_common.h"

namespace AttnResFwd {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM_RELOAD = 2;
constexpr uint32_t ELEM_PER_BLK_BF16 = 16;

struct AttnResFwdInitParams {
    GM_ADDR prefixSum;
    GM_ADDR blockResidual;
    GM_ADDR projWeight;
    GM_ADDR normWeight;
    GM_ADDR hiddenStates;
    GM_ADDR invRms;
    GM_ADDR probs;
    GM_ADDR addend{nullptr};
    GM_ADDR prefixOut{nullptr};
    GM_ADDR outputNorm{nullptr};
    GM_ADDR materialized{nullptr};
};

template <typename D_IN, bool FUSED_PREFIX = false, bool PREFILL_CACHE = false>
class AttnResFwdReload {
public:
    __aicore__ inline AttnResFwdReload(TPipe *pipe, const AttnResFwdTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        numTokens_ = tiling_->numTokens;
        numBlocks_ = tiling_->numBlocks;
        hiddenSize_ = tiling_->hiddenSize;
        blockCount_ = tiling_->blockCount;
        normEps_ = tiling_->normEps;
        invHiddenSize_ = tiling_->invHiddenSize;
        tokensPerCore_ = tiling_->tokensPerCore;
        needBackward_ = tiling_->needBackward != 0;
        hiddenSizeAlignBf16_ = (hiddenSize_ + ELEM_PER_BLK_BF16 - 1U) / ELEM_PER_BLK_BF16 * ELEM_PER_BLK_BF16;
        hiddenSizeAlignFp32_ = (hiddenSize_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
    }

    __aicore__ inline void Init(const AttnResFwdInitParams &params)
    {
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if (blockIdx_ >= blockNum_) {
            return;
        }

        tokenStart_ = blockIdx_ * tokensPerCore_;
        if (tokenStart_ >= numTokens_) {
            tokenNum_ = 0;
            return;
        }
        tokenNum_ = tokensPerCore_;
        if (tokenStart_ + tokenNum_ > numTokens_) {
            tokenNum_ = numTokens_ - tokenStart_;
        }

        prefixSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.prefixSum));
        blockResidualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.blockResidual));
        projWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.projWeight));
        normWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.normWeight));
        hiddenStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.hiddenStates));
        fuseAdd_ = params.addend != nullptr;
        if (fuseAdd_) {
            addendGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.addend));
            prefixOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.prefixOut));
        }
        if (tiling_->fusedChain != 0) {
            prefixOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.prefixOut));
            materializedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.materialized));
            if (params.outputNorm != nullptr) {
                outputNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.outputNorm));
            }
        }
        if (needBackward_) {
            invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.invRms));
            probsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.probs));
        }

        InitBuffers();
        if constexpr (FUSED_PREFIX) {
            if (tiling_->fusedChain == 0 || (tiling_->mix != 0 && numBlocks_ != 0)) {
                LoadScoreWeight();
            }
        } else {
            LoadScoreWeight();
        }
        if constexpr (FUSED_PREFIX && PREFILL_CACHE) {
            if (tiling_->cacheOutputNorm != 0) {
                LocalTensor<D_IN> weight = CopyInRowSync(outputNormGm_, 0);
                CopyB16Row(outputNorm_, weight);
                FreeInRow(weight);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (tokenNum_ == 0 || hiddenSize_ == 0) {
            return;
        }
        for (uint32_t localIdx = 0; localIdx < tokenNum_; ++localIdx) {
            ProcessOneToken(tokenStart_ + localIdx);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(inQue_, BUFFER_NUM_RELOAD, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(outQue_, 1, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(scoreWeightBuf_, hiddenSizeAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(vRowBuf_, hiddenSizeAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(outFp32Buf_, hiddenSizeAlignFp32_ * sizeof(float));
        metaAlign_ = (blockCount_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        pipe_->InitBuffer(vecMetaBuf_, metaAlign_ * sizeof(float));
        // 2 KiB fits within the tiling estimate's 32 KiB UB overhead.
        pipe_->InitBuffer(mathScratchBuf_, 512 * sizeof(float));
        mathScratch_ = mathScratchBuf_.Get<float>();
        pipe_->InitBuffer(metaSoftmaxBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(metaBrcBuf_, metaAlign_ * ELEM_PER_BLK_FP32 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, SCALAR_LOCAL_ELEMS * sizeof(float));

        if (fuseAdd_ || tiling_->fusedChain != 0) {
            pipe_->InitBuffer(fusedPrefixBuf_, hiddenSizeAlignBf16_ * sizeof(D_IN));
            fusedPrefix_ = fusedPrefixBuf_.Get<D_IN>();
        }

        scoreWeight_ = scoreWeightBuf_.Get<float>();
        vRow_ = vRowBuf_.Get<float>();
        outFp32_ = outFp32Buf_.Get<float>();
        vecMeta_ = vecMetaBuf_.Get<float>();
        metaSoftmax_ = metaSoftmaxBuf_.Get<float>();
        metaBrc_ = metaBrcBuf_.Get<float>();
        scalarLocal_ = scalarBuf_.Get<float>();

        if (needBackward_) {
            // inv：每行 1 标量经 invQue_；probs：Softmax 后每 token 一次搬 B 个经 probsQue_
            pipe_->InitBuffer(invQue_, 1, ELEM_PER_BLK_FP32 * sizeof(float));
            pipe_->InitBuffer(probsQue_, 1, metaAlign_ * sizeof(float));
        }
        if constexpr (FUSED_PREFIX && PREFILL_CACHE) {
            if (tiling_->cacheOutputNorm != 0) {
                pipe_->InitBuffer(outputNormBuf_, hiddenSizeAlignBf16_ * sizeof(D_IN));
                outputNorm_ = outputNormBuf_.Get<D_IN>();
            }
        }
    }

    __aicore__ inline void CopyInRowEnqueue(const GlobalTensor<D_IN> &srcGm, uint64_t srcOffset)
    {
        LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowDeQue()
    {
        return inQue_.DeQue<D_IN>();
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowSync(const GlobalTensor<D_IN> &srcGm, uint64_t srcOffset)
    {
        CopyInRowEnqueue(srcGm, srcOffset);
        return CopyInRowDeQue();
    }

    __aicore__ inline void FreeInRow(LocalTensor<D_IN> &inLocal)
    {
        inQue_.FreeTensor(inLocal);
    }

    /*! golden：n < N → residual；n == N → prefix */
    __aicore__ inline const GlobalTensor<D_IN> &GetLogicRowGm(uint32_t n) const
    {
        return (n < numBlocks_) ? blockResidualGm_ : prefixSumGm_;
    }

    __aicore__ inline uint64_t GetLogicRowOffset(uint32_t tokenIdx, uint32_t n) const
    {
        if (n < numBlocks_) {
            const uint64_t blockBase = static_cast<uint64_t>(tokenIdx) * tiling_->blockTokenStride;
            return blockBase + static_cast<uint64_t>(n) * hiddenSize_;
        }
        return static_cast<uint64_t>(tokenIdx) * hiddenSize_;
    }

    /*! Softmax 前：Cast → sumSq → invRms → score → vecMeta[metaIdx] */
    __aicore__ inline void ProcessRowScore(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx,
                                           uint32_t tokenIdx)
    {
        Cast(vRow_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
        PipeBarrier<PIPE_V>();
        // 非原地平方：outFp32=v²，保留 vRow=v
        Mul(outFp32_, vRow_, vRow_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        ReduceSumHalfInterval(scalarLocal_, outFp32_, static_cast<int32_t>(hiddenSize_));
        PipeBarrier<PIPE_V>();
        InvRmsInPlace(scalarLocal_, hiddenSize_, normEps_, metaBrc_, mathScratch_);
        if (needBackward_) {
            // inv 逐点搬 GM：Alloc→Copy→EnQue→DeQue→DataCopyPad→Free
            PipeBarrier<PIPE_V>();
            LocalTensor<float> invUb = invQue_.AllocTensor<float>();
            CopyMetaScalarToLocal(invUb, scalarLocal_);
            invQue_.EnQue(invUb);
            invUb = invQue_.DeQue<float>();
            DataCopyExtParams scalarParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            const uint64_t gmOffset =
                static_cast<uint64_t>(tokenIdx) * blockCount_ + static_cast<uint64_t>(metaIdx);
            DataCopyPad(invRmsGm_[gmOffset], invUb, scalarParams);
            WaitMte3();
            invQue_.FreeTensor(invUb);
        }
        BroadcastScalarMulTensor(vRow_, vRow_, scalarLocal_, metaSoftmax_, metaBrc_, hiddenSize_,
                                 hiddenSizeAlignFp32_);
        Mul(outFp32_, vRow_, scoreWeight_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        // Reduce directly into the compact score slot. WholeReduceSum emits
        // the scalar result without the unaligned vector copy that A3 rejects.
        ReduceSumHalfInterval(vecMeta_[metaIdx], outFp32_, static_cast<int32_t>(hiddenSize_));
        PipeBarrier<PIPE_V>();
    }

    /*! Softmax 后加权累加：Cast + MulAdd（调用方已外置 Counter mask） */
    __aicore__ inline void ProcessRowWeighted(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx)
    {
        CastRowToFp32CounterNoSetMask(vRow_, inLocal);
        PipeBarrier<PIPE_V>();
        MulAddRowByBrcBlock(outFp32_, vRow_, metaBrc_[metaIdx * ELEM_PER_BLK_FP32], hiddenSize_,
                            hiddenSizeAlignFp32_, false);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadScoreWeight()
    {
        LocalTensor<D_IN> inLocal = CopyInRowSync(projWeightGm_, 0);
        Cast(vRow_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
        PipeBarrier<PIPE_V>();
        FreeInRow(inLocal);

        inLocal = CopyInRowSync(normWeightGm_, 0);
        Cast(outFp32_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
        PipeBarrier<PIPE_V>();
        FreeInRow(inLocal);

        Mul(scoreWeight_, vRow_, outFp32_, hiddenSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePreSoftmaxScores(uint32_t tokenIdx)
    {
        if (blockCount_ == 0) {
            return;
        }
        CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
        for (uint32_t n = 0; n < blockCount_; ++n) {
            LocalTensor<D_IN> inLocal = CopyInRowDeQue();
            if (n + 1U < blockCount_) {
                CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
            }
            ProcessRowScore(inLocal, n, tokenIdx);
            FreeInRow(inLocal);
        }
    }

    __aicore__ inline void SoftmaxSmall()
    {
        SoftmaxSmallVec(vecMeta_, blockCount_, metaAlign_, scalarLocal_, metaSoftmax_, metaBrc_, mathScratch_);
        // 紧凑 prob → metaBrc[n*8]：零填充 staging 后单次 Brcb（repeat=ceil(B/8)）
        Duplicate(metaSoftmax_, 0.0f, metaAlign_);
        PipeBarrier<PIPE_V>();
        CopyCompactFloatsUb(metaSoftmax_, vecMeta_, blockCount_);
        const uint8_t brcRepeat =
            static_cast<uint8_t>((blockCount_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32);
        Brcb(metaBrc_, metaSoftmax_, brcRepeat, {1, MOV_8});
        PipeBarrier<PIPE_V>();
    }

    /*! Softmax 后：每 token 一次搬出 B 个 probs（Que 同步） */
    __aicore__ inline void WriteProbsQue(uint32_t tokenIdx)
    {
        LocalTensor<float> probsUb = probsQue_.AllocTensor<float>();
        CopyCompactFloatsUb(probsUb, vecMeta_, blockCount_);
        probsQue_.EnQue(probsUb);
        probsUb = probsQue_.DeQue<float>();
        DataCopyExtParams probsParams{1, blockCount_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const uint64_t gmOffset = static_cast<uint64_t>(tokenIdx) * blockCount_;
        DataCopyPad(probsGm_[gmOffset], probsUb, probsParams);
        WaitMte3();
        probsQue_.FreeTensor(probsUb);
    }

    __aicore__ inline void WeightedOutputReload(uint32_t tokenIdx)
    {
        Duplicate(outFp32_, 0.0f, hiddenSize_);
        PipeBarrier<PIPE_V>();
        const uint32_t rows = (fuseAdd_ || tiling_->fusedChain != 0) ? numBlocks_ : blockCount_;
        if (rows == 0) {
            ProcessRowWeighted(fusedPrefix_, numBlocks_);
            return;
        }
        CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
        // Counter 外置一次：Cast/MulAdd 均 isSetMask=false（同 Adds 前n 示例）
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(hiddenSize_);
        for (uint32_t n = 0; n < rows; ++n) {
            LocalTensor<D_IN> inLocal = CopyInRowDeQue();
            if (n + 1U < rows) {
                CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
            }
            ProcessRowWeighted(inLocal, n);
            FreeInRow(inLocal);
        }
        if (fuseAdd_ || tiling_->fusedChain != 0) {
            ProcessRowWeighted(fusedPrefix_, numBlocks_);
        }
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void WriteHiddenStates(uint32_t tokenIdx)
    {
        LocalTensor<D_IN> outLocal = outQue_.AllocTensor<D_IN>();
        Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
        if (tiling_->fusedChain != 0 && tiling_->saveMaterialized != 0) {
            outQue_.EnQue(outLocal);
            outLocal = outQue_.DeQue<D_IN>();
            DataCopyExtParams savedParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
            DataCopyPad(materializedGm_[static_cast<uint64_t>(tokenIdx) * hiddenSize_], outLocal, savedParams);
            auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(event);
            WaitFlag<HardEvent::MTE3_V>(event);
        }
        if (tiling_->fusedChain != 0 && tiling_->outputNormEps > 0.0f) {
            Cast(outFp32_, outLocal, RoundMode::CAST_NONE, hiddenSize_);
            PipeBarrier<PIPE_V>();
            Mul(vRow_, outFp32_, outFp32_, hiddenSize_);
            PipeBarrier<PIPE_V>();
            ReduceSumHalfInterval(scalarLocal_, vRow_, static_cast<int32_t>(hiddenSize_));
            InvRmsInPlace(scalarLocal_, hiddenSize_, tiling_->outputNormEps, metaBrc_, mathScratch_);
            BroadcastScalarMulTensor(outFp32_, outFp32_, scalarLocal_, metaSoftmax_, metaBrc_, hiddenSize_,
                                     hiddenSizeAlignFp32_);
            LocalTensor<D_IN> weight;
            const bool cachedWeight = PREFILL_CACHE && tiling_->cacheOutputNorm != 0;
            if (cachedWeight) {
                weight = outputNorm_;
            } else {
                weight = CopyInRowSync(outputNormGm_, 0);
            }
            Cast(vRow_, weight, RoundMode::CAST_NONE, hiddenSize_);
            PipeBarrier<PIPE_V>();
            if (!cachedWeight) {
                FreeInRow(weight);
            }
            Mul(outFp32_, outFp32_, vRow_, hiddenSize_);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
        }
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<D_IN>();
        const uint64_t outOffset = static_cast<uint64_t>(tokenIdx) * hiddenSize_;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        DataCopyPad(hiddenStatesGm_[outOffset], outLocal, copyParams);
        WaitMte3();
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void PrepareFusedPrefix(uint32_t tokenIdx)
    {
        if (!fuseAdd_ && tiling_->fusedChain == 0) {
            return;
        }
        const uint64_t offset = static_cast<uint64_t>(tokenIdx) * hiddenSize_;
        LocalTensor<D_IN> row = CopyInRowSync(prefixSumGm_, offset);
        CopyB16Row(fusedPrefix_, row);
        FreeInRow(row);
        if (fuseAdd_) {
            row = CopyInRowSync(addendGm_, offset);
            // Preserve the BF16 rounding boundary after the FP32 addition.
            Cast(vRow_, fusedPrefix_, RoundMode::CAST_NONE, hiddenSize_);
            Cast(outFp32_, row, RoundMode::CAST_NONE, hiddenSize_);
            PipeBarrier<PIPE_V>();
            Add(vRow_, vRow_, outFp32_, hiddenSize_);
            PipeBarrier<PIPE_V>();
            Cast(fusedPrefix_, vRow_, RoundMode::CAST_RINT, hiddenSize_);
            PipeBarrier<PIPE_V>();
            FreeInRow(row);
        }
        if (!fuseAdd_ && tiling_->blockWriteIdx < 0) {
            return;
        }
        LocalTensor<D_IN> out = outQue_.AllocTensor<D_IN>();
        CopyB16Row(out, fusedPrefix_);
        outQue_.EnQue(out);
        out = outQue_.DeQue<D_IN>();
        DataCopyExtParams params{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        if (fuseAdd_ || tiling_->fusedChain != 0) {
            DataCopyPad(prefixOutGm_[offset], out, params);
        }
        if (tiling_->blockWriteIdx >= 0) {
            const uint64_t bankOffset = static_cast<uint64_t>(tokenIdx) * tiling_->blockTokenStride +
                static_cast<uint64_t>(tiling_->blockWriteIdx) * hiddenSize_;
            DataCopyPad(blockResidualGm_[bankOffset], out, params);
        }
        WaitMte3();
        outQue_.FreeTensor(out);
    }

    __aicore__ inline void WaitMte3()
    {
        auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
    }

    __aicore__ inline void CopyB16Row(const LocalTensor<D_IN> &dst, const LocalTensor<D_IN> &src)
    {
        constexpr uint32_t elems = 256U / sizeof(D_IN);
        for (uint32_t offset = 0; offset < hiddenSizeAlignBf16_; offset += elems) {
            const uint32_t remaining = hiddenSizeAlignBf16_ - offset;
            const uint64_t mask = remaining < elems ? remaining : elems;
            Copy(dst[offset], src[offset], mask, 1, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOneToken(uint32_t tokenIdx)
    {
        PrepareFusedPrefix(tokenIdx);
        if (FUSED_PREFIX && (numBlocks_ == 0 || (tiling_->fusedChain != 0 && tiling_->mix == 0))) {
            Cast(outFp32_, fusedPrefix_, RoundMode::CAST_NONE, hiddenSize_);
            PipeBarrier<PIPE_V>();
            WriteHiddenStates(tokenIdx);
            return;
        }
        const uint32_t rows = (fuseAdd_ || tiling_->fusedChain != 0) ? numBlocks_ : blockCount_;
        if (rows != blockCount_) {
            ComputePreSoftmaxScoresFused(tokenIdx, rows);
        } else {
            ComputePreSoftmaxScores(tokenIdx);
        }
        SoftmaxSmall();
        if (needBackward_) {
            WriteProbsQue(tokenIdx);
        }
        WeightedOutputReload(tokenIdx);
        WriteHiddenStates(tokenIdx);
    }

    __aicore__ inline void ComputePreSoftmaxScoresFused(uint32_t tokenIdx, uint32_t rows)
    {
        if (rows == 0) {
            ProcessRowScore(fusedPrefix_, numBlocks_, tokenIdx);
            return;
        }
        CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
        for (uint32_t n = 0; n < rows; ++n) {
            LocalTensor<D_IN> inLocal = CopyInRowDeQue();
            if (n + 1U < rows) {
                CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
            }
            ProcessRowScore(inLocal, n, tokenIdx);
            FreeInRow(inLocal);
        }
        ProcessRowScore(fusedPrefix_, numBlocks_, tokenIdx);
    }

private:
    TPipe *pipe_{nullptr};
    const AttnResFwdTilingData *tiling_{nullptr};

    TBuf<TPosition::VECCALC> mathScratchBuf_;
    LocalTensor<float> mathScratch_;
    GlobalTensor<D_IN> prefixSumGm_;
    GlobalTensor<D_IN> blockResidualGm_;
    GlobalTensor<D_IN> projWeightGm_;
    GlobalTensor<D_IN> normWeightGm_;
    GlobalTensor<D_IN> hiddenStatesGm_;
    GlobalTensor<float> invRmsGm_;
    GlobalTensor<float> probsGm_;
    GlobalTensor<D_IN> addendGm_;
    GlobalTensor<D_IN> prefixOutGm_;
    GlobalTensor<D_IN> outputNormGm_;
    GlobalTensor<D_IN> materializedGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM_RELOAD> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECOUT, 1> invQue_;
    TQue<QuePosition::VECOUT, 1> probsQue_;
    TBuf<TPosition::VECCALC> scoreWeightBuf_;
    TBuf<TPosition::VECCALC> vRowBuf_;
    TBuf<TPosition::VECCALC> outFp32Buf_;
    TBuf<TPosition::VECCALC> vecMetaBuf_;
    TBuf<TPosition::VECCALC> metaSoftmaxBuf_;
    TBuf<TPosition::VECCALC> metaBrcBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> fusedPrefixBuf_;
    TBuf<TPosition::VECCALC> outputNormBuf_;

    LocalTensor<float> scoreWeight_;
    LocalTensor<float> vRow_;
    LocalTensor<float> outFp32_;
    LocalTensor<float> vecMeta_;
    LocalTensor<float> metaSoftmax_;
    LocalTensor<float> metaBrc_;
    LocalTensor<float> scalarLocal_;
    LocalTensor<D_IN> fusedPrefix_;
    LocalTensor<D_IN> outputNorm_;

    uint32_t numTokens_{0};
    uint32_t numBlocks_{0};
    uint32_t hiddenSize_{0};
    uint32_t hiddenSizeAlignBf16_{0};
    uint32_t hiddenSizeAlignFp32_{0};
    uint32_t blockCount_{0};
    uint32_t metaAlign_{0};
    uint32_t tokensPerCore_{0};
    uint32_t tokenStart_{0};
    uint32_t tokenNum_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    float normEps_{1e-5f};
    float invHiddenSize_{0.0f};
    bool needBackward_{false};
    bool fuseAdd_{false};
};

} // namespace AttnResFwd

#endif // ATTN_RES_FWD_RELOAD_H
