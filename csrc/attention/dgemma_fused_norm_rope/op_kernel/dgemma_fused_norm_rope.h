/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope.h
 *  \brief AscendC AIV kernel: fused q/k/v RMSNorm + neox RoPE for DiffusionGemma.
 *
 *  Per token row t, per head h (head_dim=D, rotary half=R=D/2):
 *    xf   = cast_fp32(x[t,h,:])
 *    ss   = sum(xf*xf); rstd = 1/sqrt(ss/D + eps)
 *    xn   = xf * rstd * gamma      (gamma=1 for v)
 *    neox RoPE (q,k only):
 *      x1=xn[:R], x2=xn[R:]
 *      o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin
 *      out = concat(o1,o2)
 *    v: out = xn (no weight, no rope)
 */
#ifndef DGEMMA_FUSED_NORM_ROPE_KERNEL_H
#define DGEMMA_FUSED_NORM_ROPE_KERNEL_H
#include "kernel_operator.h"
#include "dgemma_fused_norm_rope_tiling_data.h"
namespace DgemmaFusedNormRope {
using namespace AscendC;
constexpr uint32_t BYTES_PER_BLOCK = 32;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b) { return (a + b - 1) / b; }

template <typename InDtype>
class KernelDgemmaFusedNormRope {
public:
    __aicore__ inline KernelDgemmaFusedNormRope() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qw, GM_ADDR kw,
                                GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut,
                                const DgemmaFusedNormRopeTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        numTokens_  = tiling->numTokens;
        numQHeads_  = tiling->numQHeads;
        numKvHeads_ = tiling->numKvHeads;
        headDim_    = tiling->headDim;
        rotaryDim_  = tiling->rotaryDim;   // = headDim/2
        eps_        = tiling->epsilon;
        invHeadDim_ = tiling->invHeadDim;

        uint64_t qElems = (uint64_t)numTokens_ * numQHeads_ * headDim_;
        uint64_t kElems = (uint64_t)numTokens_ * numKvHeads_ * headDim_;
        qGm_.SetGlobalBuffer((__gm__ InDtype*)q, qElems);
        kGm_.SetGlobalBuffer((__gm__ InDtype*)k, kElems);
        vGm_.SetGlobalBuffer((__gm__ InDtype*)v, kElems);
        qwGm_.SetGlobalBuffer((__gm__ InDtype*)qw, headDim_);
        kwGm_.SetGlobalBuffer((__gm__ InDtype*)kw, headDim_);
        cosGm_.SetGlobalBuffer((__gm__ float*)cos, (uint64_t)numTokens_ * rotaryDim_);
        sinGm_.SetGlobalBuffer((__gm__ float*)sin, (uint64_t)numTokens_ * rotaryDim_);
        qOutGm_.SetGlobalBuffer((__gm__ InDtype*)qOut, qElems);
        kOutGm_.SetGlobalBuffer((__gm__ InDtype*)kOut, kElems);
        vOutGm_.SetGlobalBuffer((__gm__ InDtype*)vOut, kElems);

        // UB: one head vector at a time (headDim), plus cos/sin (rotaryDim) per token.
        pipe_->InitBuffer(inQue_,  1, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(outQue_, 1, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(wQue_,   1, headDim_ * sizeof(InDtype));
        pipe_->InitBuffer(cosQue_, 1, rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(sinQue_, 1, rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(xfBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(sqBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(gfBuf_,  headDim_ * sizeof(float));
        pipe_->InitBuffer(o1Buf_,  rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(o2Buf_,  rotaryDim_ * sizeof(float));
        pipe_->InitBuffer(ssBuf_,  BYTES_PER_BLOCK);   // reduce scratch (>=32B)
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t tokPerCore = CeilDiv<uint32_t>(numTokens_, blockNum);
        uint32_t tStart = blockIdx * tokPerCore;
        uint32_t tEnd = tStart + tokPerCore;
        if (tEnd > numTokens_) { tEnd = numTokens_; }

        // Preload q/k weights (gamma) once -> fp32.
        LocalTensor<float> qGamma = gfBuf_.Get<float>();  // reused per-call; load into stable buffers
        // We reload weights per head via wQue_ to keep buffer count low; cheap (headDim only).

        for (uint32_t t = tStart; t < tEnd; ++t) {
            // Load cos/sin for this token (shared across heads).
            LocalTensor<float> cosL = cosQue_.AllocTensor<float>();
            LocalTensor<float> sinL = sinQue_.AllocTensor<float>();
            DataCopyExtParams cp{1, (uint32_t)(rotaryDim_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
            DataCopyPad(cosL, cosGm_[(uint64_t)t * rotaryDim_], cp, pp);
            DataCopyPad(sinL, sinGm_[(uint64_t)t * rotaryDim_], cp, pp);
            cosQue_.EnQue(cosL); sinQue_.EnQue(sinL);
            cosL = cosQue_.DeQue<float>(); sinL = sinQue_.DeQue<float>();

            // Q heads: norm(weight) + rope
            for (uint32_t h = 0; h < numQHeads_; ++h) {
                uint64_t off = ((uint64_t)t * numQHeads_ + h) * headDim_;
                ProcessHead(qGm_, qOutGm_, qwGm_, off, cosL, sinL, /*hasWeight=*/true, /*doRope=*/true);
            }
            // K heads: norm(weight) + rope
            for (uint32_t h = 0; h < numKvHeads_; ++h) {
                uint64_t off = ((uint64_t)t * numKvHeads_ + h) * headDim_;
                ProcessHead(kGm_, kOutGm_, kwGm_, off, cosL, sinL, true, true);
            }
            // V heads: norm(no weight), no rope
            for (uint32_t h = 0; h < numKvHeads_; ++h) {
                uint64_t off = ((uint64_t)t * numKvHeads_ + h) * headDim_;
                ProcessHead(vGm_, vOutGm_, qwGm_, off, cosL, sinL, /*hasWeight=*/false, /*doRope=*/false);
            }
            cosQue_.FreeTensor(cosL); sinQue_.FreeTensor(sinL);
        }
    }

private:
    template <typename GmT>
    __aicore__ inline void ProcessHead(GmT &inGm, GmT &outGm, GlobalTensor<InDtype> &wGm,
                                       uint64_t off, LocalTensor<float> &cosL, LocalTensor<float> &sinL,
                                       bool hasWeight, bool doRope)
    {
        LocalTensor<InDtype> xin = inQue_.AllocTensor<InDtype>();
        DataCopyExtParams cp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
        DataCopyPadExtParams<InDtype> pp{false, 0, 0, (InDtype)0};
        DataCopyPad(xin, inGm[off], cp, pp);
        inQue_.EnQue(xin);
        xin = inQue_.DeQue<InDtype>();

        LocalTensor<float> xf = xfBuf_.Get<float>();
        LocalTensor<float> sq = sqBuf_.Get<float>();
        LocalTensor<float> ss = ssBuf_.Get<float>();
        Cast(xf, xin, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        inQue_.FreeTensor(xin);

        // sum of squares
        Mul(sq, xf, xf, headDim_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(ss, sq, sq, headDim_);
        PipeBarrier<PIPE_V>();
        // mean = sumSq * invHeadDim ; std = sqrt(mean + eps) ; rstd = 1/std  (vector Sqrt, no libm)
        Muls(ss, ss, invHeadDim_, 1);
        PipeBarrier<PIPE_V>();
        Adds(ss, ss, eps_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(ss, ss, 1);
        PipeBarrier<PIPE_V>();
        float stdv = ss.GetValue(0);
        float rstd = 1.0f / stdv;

        // xf = xf * rstd
        Muls(xf, xf, rstd, headDim_);
        PipeBarrier<PIPE_V>();

        // multiply gamma (q/k only)
        if (hasWeight) {
            LocalTensor<InDtype> w = wQue_.AllocTensor<InDtype>();
            DataCopyExtParams wcp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
            DataCopyPadExtParams<InDtype> wpp{false, 0, 0, (InDtype)0};
            DataCopyPad(w, wGm, wcp, wpp);
            wQue_.EnQue(w);
            w = wQue_.DeQue<InDtype>();
            LocalTensor<float> gf = gfBuf_.Get<float>();
            Cast(gf, w, RoundMode::CAST_NONE, headDim_);
            PipeBarrier<PIPE_V>();
            wQue_.FreeTensor(w);
            Mul(xf, xf, gf, headDim_);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<InDtype> yout = outQue_.AllocTensor<InDtype>();
        if (doRope) {
            // neox: o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin
            LocalTensor<float> o1 = o1Buf_.Get<float>();
            LocalTensor<float> o2 = o2Buf_.Get<float>();
            // x1 = xf[0:R], x2 = xf[R:2R]
            // o1 = x1*cos ; tmp = x2*sin ; o1 -= tmp
            Mul(o1, xf, cosL, rotaryDim_);
            Mul(o2, xf[rotaryDim_], sinL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Sub(o1, o1, o2, rotaryDim_);
            PipeBarrier<PIPE_V>();
            // o2 = x2*cos ; tmp2 = x1*sin ; o2 += tmp2   (reuse sq as tmp)
            LocalTensor<float> tmp = sqBuf_.Get<float>();
            Mul(o2, xf[rotaryDim_], cosL, rotaryDim_);
            Mul(tmp, xf, sinL, rotaryDim_);
            PipeBarrier<PIPE_V>();
            Add(o2, o2, tmp, rotaryDim_);
            PipeBarrier<PIPE_V>();
            // write o1 -> xf[0:R], o2 -> xf[R:], then cast to out
            Cast(yout, o1, RoundMode::CAST_RINT, rotaryDim_);
            Cast(yout[rotaryDim_], o2, RoundMode::CAST_RINT, rotaryDim_);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(yout, xf, RoundMode::CAST_RINT, headDim_);
            PipeBarrier<PIPE_V>();
        }
        outQue_.EnQue(yout);
        yout = outQue_.DeQue<InDtype>();
        DataCopyExtParams ocp{1, (uint32_t)(headDim_ * sizeof(InDtype)), 0, 0, 0};
        DataCopyPad(outGm[off], yout, ocp);
        outQue_.FreeTensor(yout);
    }

    TPipe *pipe_;
    GlobalTensor<InDtype> qGm_, kGm_, vGm_, qwGm_, kwGm_, qOutGm_, kOutGm_, vOutGm_;
    GlobalTensor<float> cosGm_, sinGm_;
    TQue<QuePosition::VECIN, 1> inQue_, wQue_, cosQue_, sinQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> xfBuf_, sqBuf_, gfBuf_, o1Buf_, o2Buf_, ssBuf_;
    uint32_t numTokens_, numQHeads_, numKvHeads_, headDim_, rotaryDim_;
    float eps_;
    float invHeadDim_;
};
} // namespace DgemmaFusedNormRope
#endif
