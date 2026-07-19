/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_aiv_kernel.h
 *  \brief Vector (AIV) epilogue: per m-split, read qkv from GM workspace,
 *  split q/k/v heads, RMSNorm + neox RoPE, write outputs. Overlaps cube via WaitEvent.
 *  OPT: coalesce the 32 per-head 512B GM loads into 3 per-region bulk loads
 *  (q[qSize], k[kvSize], v[kvSize]); v is loaded LAST (after q+k compute) so the cube's
 *  fixpipe->HBM writeback of the v tail has drained -> no stale/NaN. Norm weights are
 *  preloaded once (were reloaded 24x/token). */
#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_AIV_KERNEL_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_AIV_KERNEL_H

#include <kernel_operator.h>
#include "dgemma_fused_qkv_proj_norm_rope_tiling_data.h"
#include "dgemma_fused_qkv_proj_norm_rope_utils.h"

using namespace AscendC;
using namespace DgemmaFusedQkvProjNormRope;

template <class InDtype>
class DgemmaFusedQkvProjNormRopeAivKernel {
public:
    __aicore__ inline DgemmaFusedQkvProjNormRopeAivKernel() {}

    __aicore__ inline void Init(GM_ADDR qkv_ws, GM_ADDR q_weight, GM_ADDR k_weight,
                                GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR q_out, GM_ADDR k_out, GM_ADDR v_out,
                                const DgemmaFusedQkvProjNormRopeTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        m_ = tiling->m; n_ = tiling->n;
        numQHeads_ = tiling->numQHeads; numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim; rotaryDim_ = tiling->rotaryDim;
        syncDoneFlag_ = tiling->syncDoneFlag;
        syncReadyFlag_ = tiling->syncReadyFlag;
        skipEpilogue_ = (tiling->coreNum & 0x10000U) != 0;
        publishOutputs_ = (tiling->coreNum & 0x20000U) != 0;
        epsilon_ = tiling->epsilon; invHeadDim_ = tiling->invHeadDim;
        m0_ = tiling->m0; swizzlCount_ = tiling->swizzlCount;
        qSize_ = numQHeads_ * headDim_;
        kvSize_ = numKvHeads_ * headDim_;

        qkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qkv_ws), (uint64_t)m_ * n_);
        qwGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(q_weight));
        kwGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(k_weight));
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cos));
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sin));
        qOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(q_out));
        kOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(k_out));
        vOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(v_out));

        // Conservative graph-safe path: load one head at a time, matching the
        // standalone AIV-only norm/rope op. The older bulk-region UB slicing
        // path was fast in microbench but corrupted DiffusionGemma convergence
        // under graph replay.
        pipe_->InitBuffer(regQue_, 1, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(outQue_, 1, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(cosQue_, 1, rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(sinQue_, 1, rotaryDim_ * sizeof(float));
        // Persistent (non-queue) UB: preloaded norm weights + fp32 scratch.
        pipe_->InitBuffer(qwBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(kwBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(qwLdBuf_, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(kwLdBuf_, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(xfBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(sqBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(o1Buf_,  rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(o2Buf_,  rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(ssBuf_,  32);

        core_idx_ = get_block_idx();
        core_num_ = get_block_num();
    }

    __aicore__ inline void Process()
    {
        FFTSCrossCoreSync<PIPE_MTE3>(FFTS_SYNC_AICORE_GROUP_MODE, syncReadyFlag_);
        PipeBarrier<PIPE_ALL>();

        if (skipEpilogue_) {
            WaitEvent(syncDoneFlag_);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        // Preload q/k RMSNorm weights ONCE (constant across tokens). Distinct staging
        // buffers + MTE2->V ordering so the two loads cannot alias.
        qwF_ = qwBuf_.Get<float>();
        kwF_ = kwBuf_.Get<float>();
        {
            LocalTensor<InDtype> wq = qwLdBuf_.Get<InDtype>();
            LocalTensor<InDtype> wk = kwLdBuf_.Get<InDtype>();
            DataCopyExtParams wcp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
            DataCopyPadExtParams<InDtype> wpp{false, 0, 0, (InDtype)0};
            DataCopyPad(wq, qwGm_, wcp, wpp);
            DataCopyPad(wk, kwGm_, wcp, wpp);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(qwF_, wq, RoundMode::CAST_NONE, headDim_);
            Cast(kwF_, wk, RoundMode::CAST_NONE, headDim_);
            PipeBarrier<PIPE_V>();
        }

        int mPerSplit = m0_ * swizzlCount_;
        if (mPerSplit <= 0) mPerSplit = m0_;
        int splitM = AscendC::DivCeil((int)m_, mPerSplit);

        WaitEvent(syncDoneFlag_);

        for (int splitIndex = 0; splitIndex < splitM; ++splitIndex) {
            uint32_t mStart = splitIndex * mPerSplit;
            uint32_t mActual = ((uint32_t)mPerSplit > (m_ - mStart)) ? (m_ - mStart) : (uint32_t)mPerSplit;

            for (uint32_t tt = core_idx_; tt < mActual; tt += core_num_) {
                uint32_t t = mStart + tt;
                uint64_t base = (uint64_t)t * n_;

                // Coherence: invalidate this token's [n] row from cache so region loads
                // fetch the cube's committed fixpipe data (persistent scratch keeps addr
                // stable). v is loaded LAST so its fixpipe->HBM writeback has drained.
                AscendC::GlobalTensor<InDtype> rowGm;
                rowGm.SetGlobalBuffer(
                    const_cast<__gm__ InDtype *>(qkvGm_[base].GetPhyAddr()), (uint64_t)n_);
                AscendC::DataCacheCleanAndInvalid<InDtype, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                                  AscendC::DcciDst::CACHELINE_OUT>(rowGm);
                PipeBarrier<PIPE_ALL>();

                // cos/sin for this token
                LocalTensor<float> cosL = cosQue_.AllocTensor<float>();
                LocalTensor<float> sinL = sinQue_.AllocTensor<float>();
                DataCopyExtParams cp{1, (uint32_t)(rotaryDim_ * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                DataCopyPad(cosL, cosGm_[(uint64_t)t * rotaryDim_], cp, pp);
                DataCopyPad(sinL, sinGm_[(uint64_t)t * rotaryDim_], cp, pp);
                cosQue_.EnQue(cosL); sinQue_.EnQue(sinL);
                cosL = cosQue_.DeQue<float>(); sinL = sinQue_.DeQue<float>();

                // ---- q region ----
                ProcessRegion(base, qSize_, numQHeads_, qwF_, true, true,
                              (uint64_t)t * qSize_, cosL, sinL, qOutGm_);
                // ---- k region ----
                ProcessRegion(base + qSize_, kvSize_, numKvHeads_, kwF_, true, true,
                              (uint64_t)t * kvSize_, cosL, sinL, kOutGm_);
                // ---- v region (loaded LAST): RMSNorm only ----
                ProcessRegion(base + qSize_ + kvSize_, kvSize_, numKvHeads_, qwF_, false, false,
                              (uint64_t)t * kvSize_, cosL, sinL, vOutGm_);

                cosQue_.FreeTensor(cosL); sinQue_.FreeTensor(sinL);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    // Load one head at a time from qkv workspace and run RMSNorm(+rope). This
    // mirrors DgemmaFusedNormRope's proven graph-safe access pattern.
    __aicore__ inline void ProcessRegion(uint64_t gmOff, uint32_t regElems, uint32_t numHeads,
                                         const LocalTensor<float> &wF, bool hasWeight, bool doRope,
                                         uint64_t outBase, const LocalTensor<float> &cosL,
                                         const LocalTensor<float> &sinL, const GlobalTensor<InDtype> &outGm)
    {
        (void)regElems;
        DataCopyExtParams rcp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
        DataCopyPadExtParams<InDtype> rpp{false, 0, 0, (InDtype)0};
        for (uint32_t h = 0; h < numHeads; ++h) {
            LocalTensor<InDtype> reg = regQue_.AllocTensor<InDtype>();
            DataCopyPad(reg, qkvGm_[gmOff + (uint64_t)h * headDim_], rcp, rpp);
            regQue_.EnQue(reg);
            reg = regQue_.DeQue<InDtype>();
            NormRopeHead(reg, 0, wF, hasWeight, doRope,
                         cosL, sinL, outGm, outBase + (uint64_t)h * headDim_);
            regQue_.FreeTensor(reg);
        }
    }

    // RMSNorm (optional preloaded fp32 weight) + optional neox RoPE on one head,
    // reading from the in-UB region slice.
    __aicore__ inline void NormRopeHead(const LocalTensor<InDtype> &reg, uint64_t regOff,
                                        const LocalTensor<float> &wF, bool hasWeight, bool doRope,
                                        const LocalTensor<float> &cosL, const LocalTensor<float> &sinL,
                                        const GlobalTensor<InDtype> &outGm, uint64_t outoff)
    {
        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> sq = sqBuf_.Get<float>();
        LocalTensor<float> ss = ssBuf_.Get<float>();
        Cast(xf, reg[regOff], RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_ALL>();
        Mul(sq, xf, xf, headDim_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ss, sq, sq, headDim_);
        PipeBarrier<PIPE_V>();
        Muls(ss, ss, invHeadDim_, 1);
        PipeBarrier<PIPE_V>();
        Adds(ss, ss, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(ss, ss, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float stdv = ss.GetValue(0);
        float rstd = 1.0f / stdv;
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
        Muls(xf, xf, rstd, headDim_);
        PipeBarrier<PIPE_V>();

        if (hasWeight) {
            Mul(xf, xf, wF, headDim_);   // preloaded fp32 weight, no per-head load
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<InDtype> yout = outQue_.AllocTensor<InDtype>();
        if (doRope) {
            LocalTensor<float> o1 = o1Buf_.Get<float>();
            LocalTensor<float> o2 = o2Buf_.Get<float>();
            LocalTensor<float> tmp = sqBuf_.Get<float>();
            Mul(o1, xf, cosL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Mul(tmp, xf[rotaryDim_], sinL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Sub(o1, o1, tmp, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Mul(o2, xf[rotaryDim_], cosL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Mul(tmp, xf, sinL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Add(o2, o2, tmp, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Cast(yout, o1, RoundMode::CAST_RINT, rotaryDim_);
            Cast(yout[rotaryDim_], o2, RoundMode::CAST_RINT, rotaryDim_);
        } else {
            Cast(yout, xf, RoundMode::CAST_RINT, headDim_);
        }
        PipeBarrier<PIPE_V>();
        outQue_.EnQue(yout);
        yout = outQue_.DeQue<InDtype>();
        DataCopyExtParams ocp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
        DataCopyPad(outGm[outoff], yout, ocp);
        outQue_.FreeTensor(yout);
        if (publishOutputs_) {
            // Direct full-MIX feeds these returned q/k/v tensors immediately into
            // attention under graph replay. Publish each head range conservatively so
            // downstream kernels do not observe a stale AIV MTE3 write.
            PipeBarrier<PIPE_ALL>();
            AscendC::GlobalTensor<InDtype> outHeadGm;
            outHeadGm.SetGlobalBuffer(
                const_cast<__gm__ InDtype *>(outGm[outoff].GetPhyAddr()), (uint64_t)headDim_);
            AscendC::DataCacheCleanAndInvalid<InDtype, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                              AscendC::DcciDst::CACHELINE_OUT>(outHeadGm);
            PipeBarrier<PIPE_ALL>();
        } else {
            PipeBarrier<PIPE_V>();
        }
    }

    TPipe *pipe_{nullptr};
    GlobalTensor<InDtype> qkvGm_, qwGm_, kwGm_, qOutGm_, kOutGm_, vOutGm_;
    GlobalTensor<float> cosGm_, sinGm_;
    TQue<QuePosition::VECIN, 1> regQue_, cosQue_, sinQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> qwBuf_, kwBuf_, qwLdBuf_, kwLdBuf_, xfBuf_, sqBuf_, o1Buf_, o2Buf_, ssBuf_;
    LocalTensor<float> qwF_, kwF_;
    uint32_t m_, n_, numQHeads_, numKvHeads_, headDim_, rotaryDim_, qSize_, kvSize_;
    uint32_t m0_, swizzlCount_, core_idx_, core_num_;
    uint32_t syncDoneFlag_, syncReadyFlag_;
    bool skipEpilogue_{false}, publishOutputs_{false};
    float epsilon_, invHeadDim_;
};
#endif
