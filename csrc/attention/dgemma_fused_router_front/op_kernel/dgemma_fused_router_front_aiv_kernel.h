/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_aiv_kernel.h
 *  \brief Vector (AIV) stage of the router MIX op, runs FIRST:
 *  per token row t: xf=cast(x); rstd=1/sqrt(mean(xf^2)+eps);
 *  norm_scratch[t] = xf * rstd * root_size * scale[h].
 *  After ALL rows are written + fixpipe drained, signal the cube (which then GEMMs). */
#ifndef DGEMMA_FUSED_ROUTER_FRONT_AIV_KERNEL_H
#define DGEMMA_FUSED_ROUTER_FRONT_AIV_KERNEL_H

#include <kernel_operator.h>
#include "dgemma_fused_router_front_tiling_data.h"
#include "dgemma_fused_router_front_utils.h"

using namespace AscendC;
using namespace DgemmaFusedRouterFront;

template <class InDtype>
class DgemmaFusedRouterFrontAivKernel {
public:
    __aicore__ inline DgemmaFusedRouterFrontAivKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR norm_ws,
                                GM_ADDR logits_ws, GM_ADDR per_expert_scale, GM_ADDR sync_scratch,
                                GM_ADDR topk_weights, GM_ADDR topk_ids,
                                const DgemmaFusedRouterFrontTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        m_ = tiling->m; k_ = tiling->k; n_ = tiling->n; topK_ = tiling->topK;
        syncAlignFlag_ = tiling->syncReadyFlag;
        syncNormReadyFlag_ = tiling->syncDoneFlag + 1U;
        syncLogitsDoneFlag_ = tiling->syncDoneFlag + 2U;
        eps_ = tiling->epsilon; invHidden_ = tiling->invHidden; rootSize_ = tiling->rootSize;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(x), (uint64_t)m_ * k_);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(scale), (uint64_t)k_);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(norm_ws), (uint64_t)m_ * k_);
        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logits_ws), (uint64_t)m_ * n_);
        perExpertScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(per_expert_scale), (uint64_t)n_);
        syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sync_scratch), 128);
        topkWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(topk_weights), (uint64_t)m_ * topK_);
        topkIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topk_ids), (uint64_t)m_ * topK_);

        pipe_->InitBuffer(inQue_,  2, k_ * sizeof(InDtype));
        pipe_->InitBuffer(outQue_, 2, k_ * sizeof(InDtype));
        pipe_->InitBuffer(scaleQue_, 1, k_ * sizeof(InDtype));
        pipe_->InitBuffer(xfBuf_, k_ * sizeof(float));
        pipe_->InitBuffer(sqBuf_, k_ * sizeof(float));
        pipe_->InitBuffer(sfBuf_, k_ * sizeof(float));
        pipe_->InitBuffer(ssBuf_, 32);
        pipe_->InitBuffer(logitsQue_, 1, n_ * sizeof(float));
        pipe_->InitBuffer(logitsBuf_, n_ * sizeof(float));
        pipe_->InitBuffer(logitsRoundBuf_, n_ * sizeof(InDtype));
        pipe_->InitBuffer(topBuf_, 64 * sizeof(float));

        core_idx_ = get_block_idx();
        core_num_ = get_block_num();
    }

    __aicore__ inline void Process()
    {
        // Load per-dim scale once (shared across all rows), cast fp32.
        LocalTensor<InDtype> sc = scaleQue_.AllocTensor<InDtype>();
        DataCopyExtParams scp{1, (uint32_t)(k_ * sizeof(InDtype)), 0, 0, 0};
        DataCopyPadExtParams<InDtype> spp{false, 0, 0, (InDtype)0};
        DataCopyPad(sc, scaleGm_, scp, spp);
        scaleQue_.EnQue(sc);
        sc = scaleQue_.DeQue<InDtype>();
        LocalTensor<float> sf = sfBuf_.Get<float>();
        Cast(sf, sc, RoundMode::CAST_NONE, k_);
        PipeBarrier<PIPE_V>();
        scaleQue_.FreeTensor(sc);

        // Startup alignment: let AIC cores enter their wait path only after AIV
        // cores have reached this op instance. The actual data dependency uses a
        // separate norm-ready flag below, so a stale align event cannot release
        // the cube onto old norm_scratch contents.
        FFTSCrossCoreSync<PIPE_V>(FFTS_SYNC_AICORE_GROUP_MODE, syncAlignFlag_);
        FFTSCrossCoreSync<PIPE_V>(FFTS_SYNC_AICORE_GROUP_MODE, syncAlignFlag_);

        // Distribute tokens across vector cores.
        for (uint32_t t = core_idx_; t < m_; t += core_num_) {
            uint64_t off = (uint64_t)t * k_;
            LocalTensor<InDtype> xin = inQue_.AllocTensor<InDtype>();
            DataCopyExtParams cp{1, (uint32_t)(k_ * sizeof(InDtype)), 0, 0, 0};
            DataCopyPadExtParams<InDtype> pp{false, 0, 0, (InDtype)0};
            DataCopyPad(xin, xGm_[off], cp, pp);
            inQue_.EnQue(xin);
            xin = inQue_.DeQue<InDtype>();

            LocalTensor<float> xf = xfBuf_.Get<float>();
            LocalTensor<float> sq = sqBuf_.Get<float>();
            LocalTensor<float> ss = ssBuf_.Get<float>();
            Cast(xf, xin, RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            inQue_.FreeTensor(xin);
            Mul(sq, xf, xf, k_);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(ss, sq, sq, k_);
            PipeBarrier<PIPE_V>();
            Muls(ss, ss, invHidden_, 1);
            PipeBarrier<PIPE_V>();
            Adds(ss, ss, eps_, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(ss, ss, 1);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float stdv = ss.GetValue(0);
            float rstd = 1.0f / stdv;
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            Muls(xf, xf, rstd * rootSize_, k_);   // fold rstd and 1/sqrt(H)
            PipeBarrier<PIPE_V>();
            Mul(xf, xf, sf, k_);                   // per-dim scale
            PipeBarrier<PIPE_V>();

            LocalTensor<InDtype> yout = outQue_.AllocTensor<InDtype>();
            Cast(yout, xf, RoundMode::CAST_RINT, k_);
            PipeBarrier<PIPE_V>();
            outQue_.EnQue(yout);
            yout = outQue_.DeQue<InDtype>();
            DataCopyExtParams ocp{1, (uint32_t)(k_ * sizeof(InDtype)), 0, 0, 0};
            DataCopyPad(wsGm_[off], yout, ocp);
            outQue_.FreeTensor(yout);
        }

        // All normed rows written. The FFTSCrossCoreSync on PIPE_MTE3 fences the vector's
        // MTE3 stores to wsGm_ before releasing the cube's WaitEvent, so the cube's MTE2
        // reads see committed data without a full-scratch cache invalidate.
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, syncNormReadyFlag_);
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, syncNormReadyFlag_);

        WaitEvent(syncLogitsDoneFlag_);
        WaitEvent(syncLogitsDoneFlag_);
        if (core_idx_ == 0) {
            ComputeTopKScale();
        }
    }

private:
    __aicore__ inline bool BetterTop(float v, int32_t id, float best, int32_t bestId)
    {
        if (v > best) {
            return true;
        }
        if (v == best && id < bestId) {
            return true;
        }
        return false;
    }

    __aicore__ inline void WriteTopValue(LocalTensor<float> dst, uint32_t j,
                                         float v0, float v1, float v2, float v3,
                                         float v4, float v5, float v6, float v7,
                                         float maxTop)
    {
        if (j == 0) dst.SetValue(j, v0 - maxTop);
        if (j == 1) dst.SetValue(j, v1 - maxTop);
        if (j == 2) dst.SetValue(j, v2 - maxTop);
        if (j == 3) dst.SetValue(j, v3 - maxTop);
        if (j == 4) dst.SetValue(j, v4 - maxTop);
        if (j == 5) dst.SetValue(j, v5 - maxTop);
        if (j == 6) dst.SetValue(j, v6 - maxTop);
        if (j == 7) dst.SetValue(j, v7 - maxTop);
    }

    __aicore__ inline int32_t GetTopId(uint32_t j,
                                       int32_t i0, int32_t i1, int32_t i2, int32_t i3,
                                       int32_t i4, int32_t i5, int32_t i6, int32_t i7)
    {
        if (j == 0) return i0;
        if (j == 1) return i1;
        if (j == 2) return i2;
        if (j == 3) return i3;
        if (j == 4) return i4;
        if (j == 5) return i5;
        if (j == 6) return i6;
        return i7;
    }

    __aicore__ inline void ComputeTopKScale()
    {
        for (uint32_t t = 0; t < m_; ++t) {
            uint64_t rowOff = (uint64_t)t * n_;
            AscendC::GlobalTensor<float> rowGm;
            rowGm.SetGlobalBuffer(
                const_cast<__gm__ float *>(logitsGm_[rowOff].GetPhyAddr()), (uint64_t)n_);
            AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                             AscendC::DcciDst::CACHELINE_OUT>(rowGm);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<float> lin = logitsQue_.AllocTensor<float>();
            DataCopyExtParams cp{1, (uint32_t)(n_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
            DataCopyPad(lin, logitsGm_[rowOff], cp, pp);
            logitsQue_.EnQue(lin);
            lin = logitsQue_.DeQue<float>();
            LocalTensor<float> logits = lin;
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            LocalTensor<InDtype> logitsRound = logitsRoundBuf_.Get<InDtype>();
            LocalTensor<float> logitsCmp = logitsBuf_.Get<float>();
            Cast(logitsRound, logits, RoundMode::CAST_RINT, n_);
            PipeBarrier<PIPE_V>();
            Cast(logitsCmp, logitsRound, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            float v0 = -3.4028234663852886e38f;
            float v1 = -3.4028234663852886e38f;
            float v2 = -3.4028234663852886e38f;
            float v3 = -3.4028234663852886e38f;
            float v4 = -3.4028234663852886e38f;
            float v5 = -3.4028234663852886e38f;
            float v6 = -3.4028234663852886e38f;
            float v7 = -3.4028234663852886e38f;
            int32_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
            for (uint32_t e = 0; e < n_; ++e) {
                float v = logitsCmp.GetValue(e);
                int32_t ie = (int32_t)e;
                if (BetterTop(v, ie, v0, i0)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
                    v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = ie;
                } else if (BetterTop(v, ie, v1, i1)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
                    v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v; i1 = ie;
                } else if (BetterTop(v, ie, v2, i2)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
                    v3 = v2; i3 = i2; v2 = v; i2 = ie;
                } else if (BetterTop(v, ie, v3, i3)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v3; i4 = i3;
                    v3 = v; i3 = ie;
                } else if (BetterTop(v, ie, v4, i4)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v4; i5 = i4; v4 = v; i4 = ie;
                } else if (BetterTop(v, ie, v5, i5)) {
                    v7 = v6; i7 = i6; v6 = v5; i6 = i5; v5 = v; i5 = ie;
                } else if (BetterTop(v, ie, v6, i6)) {
                    v7 = v6; i7 = i6; v6 = v; i6 = ie;
                } else if (BetterTop(v, ie, v7, i7)) {
                    v7 = v; i7 = ie;
                }
            }

            float maxTop = v0;
            LocalTensor<float> topLocal = topBuf_.Get<float>();
            uint32_t topK = topK_ > 8 ? 8 : topK_;
            for (uint32_t j = 0; j < topK; ++j) {
                WriteTopValue(topLocal, j, v0, v1, v2, v3, v4, v5, v6, v7, maxTop);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            PipeBarrier<PIPE_V>();
            Exp(topLocal, topLocal, topK);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            float sum = 0.0f;
            for (uint32_t j = 0; j < topK; ++j) {
                sum += topLocal.GetValue(j);
            }
            float invSum = 1.0f / sum;
            for (uint32_t j = 0; j < topK; ++j) {
                int32_t topId = GetTopId(j, i0, i1, i2, i3, i4, i5, i6, i7);
                float s = perExpertScaleGm_.GetValue((uint32_t)topId);
                topkWeightsGm_.SetValue((uint64_t)t * topK_ + j, topLocal.GetValue(j) * invSum * s);
                topkIdsGm_.SetValue((uint64_t)t * topK_ + j, topId);
            }
            logitsQue_.FreeTensor(lin);
        }
    }

    TPipe *pipe_{nullptr};
    GlobalTensor<InDtype> xGm_, scaleGm_, wsGm_;
    GlobalTensor<float> logitsGm_;
    GlobalTensor<float> perExpertScaleGm_, topkWeightsGm_;
    GlobalTensor<int32_t> syncGm_;
    GlobalTensor<int32_t> topkIdsGm_;
    TQue<QuePosition::VECIN, 2> inQue_;
    TQue<QuePosition::VECIN, 1> scaleQue_;
    TQue<QuePosition::VECIN, 1> logitsQue_;
    TQue<QuePosition::VECOUT, 2> outQue_;
    TBuf<QuePosition::VECCALC> xfBuf_, sqBuf_, sfBuf_, ssBuf_, logitsBuf_, logitsRoundBuf_, topBuf_;
    uint32_t m_, k_, n_, topK_, core_idx_, core_num_;
    uint32_t syncAlignFlag_, syncNormReadyFlag_, syncLogitsDoneFlag_;
    float eps_, invHidden_, rootSize_;
};
#endif
