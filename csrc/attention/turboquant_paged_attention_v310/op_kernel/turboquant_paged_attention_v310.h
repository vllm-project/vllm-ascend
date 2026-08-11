/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * TurboQuant read path: decode attention straight out of the packed cache.
 *
 * The point of the whole exercise. Tier 0 gathered the context, dequantised it
 * into a dense fp16 tensor, and called a stock attention op -- which works, but
 * materialises O(context) per decode step and therefore OOM-killed at 6k and hit
 * an aicore timeout at 11k. Here the dequantised block never leaves UB.
 *
 * ROTATED-BASIS ATTENTION
 *   Pi is orthogonal, so
 *       <q, n_k * Pi@yhat_k>            == n_k * <Pi@q, yhat_k>
 *       sum_k p_k*n_k*Pi@yhat_k         == Pi @ (sum_k p_k*n_k*yhat_k)
 *   Rotate the query once on entry and the output once on exit; every cached
 *   key/value stays in the rotated basis and is never inverse-rotated.
 *   O(1) rotations per request instead of O(context).
 *   Verified on device at b=2/3/4: rotated == original, cosine 1.000000.
 *
 * ONLINE SOFTMAX
 *   Standard flash-attention running max/sum, so scores for a block are consumed
 *   immediately and only (acc, runningMax, runningSum) persist across blocks.
 */
#ifndef TURBOQUANT_PAGED_ATTENTION_V310_H
#define TURBOQUANT_PAGED_ATTENTION_V310_H

#include "kernel_operator.h"
#include "turboquant_paged_attention_v310_codec.h"
#include "turboquant_paged_attention_v310_tiling_data.h"

namespace TurboQuantRead {

using namespace AscendC;
using namespace TurboQuant;

constexpr int32_t kNzC0 = 16;
constexpr float kNegInf = -3.0e38f;

struct TQReadParams {
    GM_ADDR query;
    GM_ADDR keyCache;
    GM_ADDR valueCache;
    GM_ADDR keyNorms;
    GM_ADDR valueNorms;
    GM_ADDR blockTable;
    GM_ADDR seqLens;
    GM_ADDR signs;
    GM_ADDR centroids;
    GM_ADDR attnOut;
};

template <int BITS>
class TQPagedAttention {
public:
    __aicore__ inline TQPagedAttention(const TurboquantPagedAttentionV310TilingData *t) : t_(t) {}

    __aicore__ inline void Init(const TQReadParams &p, TPipe *pipe)
    {
        const uint32_t d = t_->headDim;
        qGm_.SetGlobalBuffer((__gm__ half *)p.query);
        kCacheGm_.SetGlobalBuffer((__gm__ half *)p.keyCache);
        vCacheGm_.SetGlobalBuffer((__gm__ half *)p.valueCache);
        kNormGm_.SetGlobalBuffer((__gm__ half *)p.keyNorms);
        vNormGm_.SetGlobalBuffer((__gm__ half *)p.valueNorms);
        btGm_.SetGlobalBuffer((__gm__ int32_t *)p.blockTable);
        seqGm_.SetGlobalBuffer((__gm__ int32_t *)p.seqLens);
        signGm_.SetGlobalBuffer((__gm__ float *)p.signs);
        cbGm_.SetGlobalBuffer((__gm__ float *)p.centroids);
        outGm_.SetGlobalBuffer((__gm__ half *)p.attnOut);

        pipe->InitBuffer(qBuf_, d * sizeof(float));
        pipe->InitBuffer(accBuf_, d * sizeof(float));
        pipe->InitBuffer(kvBuf_, d * sizeof(float));
        pipe->InitBuffer(tmpBuf_, d * sizeof(float));
        pipe->InitBuffer(byteBuf_, TQTraits<BITS>::PackedBytes(d));
        pipe->InitBuffer(codeBuf_, d * sizeof(int32_t));
        pipe->InitBuffer(signBuf_, d * sizeof(float));
        pipe->InitBuffer(cbBuf_, (2 * (1 << BITS)) * sizeof(float));
        pipe->InitBuffer(outBuf_, d * sizeof(half));
        pipe->InitBuffer(redBuf_, d * sizeof(float));   // ReduceSum dst (must not alias src)
        pipe->InitBuffer(wrkBuf_, d * sizeof(float));
        pipe->InitBuffer(dbgBuf_, d * sizeof(float));
        pipe->InitBuffer(unpHBuf_, (d / 2) * sizeof(half));      // planar unpack scratch
        pipe->InitBuffer(unpF0Buf_, (d / 2) * sizeof(float));
        pipe->InitBuffer(unpF1Buf_, (d / 2) * sizeof(float));
        // norm planes prefetched per BLOCK: replaces two GM scalar reads per
        // token (dependent global loads, pure latency) with one DataCopy.
        pipe->InitBuffer(kNrmBuf_, t_->blockSize * t_->numKvHeads * sizeof(half));
        pipe->InitBuffer(vNrmBuf_, t_->blockSize * t_->numKvHeads * sizeof(half));
        // block-wise softmax: one score/weight slot per token in a block
        pipe->InitBuffer(scoBuf_, t_->blockSize * sizeof(float));
        pipe->InitBuffer(wgtBuf_, t_->blockSize * sizeof(float));
        pipe->InitBuffer(rdxBuf_, t_->blockSize * sizeof(float));   // variant==10 dot dump; nothing else touches it   // ReduceSum workLocal (must not alias src/dst)

        signs_ = signBuf_.Get<float>();
        DataCopy(signs_, signGm_, d);
        cb_ = cbBuf_.Get<float>();
        if (t_->codebookMode == TQ_CB_LUT) {
            DataCopy(cb_, cbGm_, 2 * (1 << BITS));
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);   // signs_/cb_ are consumed by V ops
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }

    /*
     * Work split: one (batch, kv_head) pair per task. All query heads sharing a
     * KV head are processed together so the packed block is unpacked ONCE and
     * reused across the group -- this is what replaces Tier 0's repeat_interleave.
     */
    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t nTasks = t_->batch * t_->numKvHeads;
        const uint32_t begin = core * t_->tasksPerCore;
        uint32_t end = begin + t_->tasksPerCore;
        if (end > nTasks) {
            end = nTasks;
        }
        const uint32_t group = t_->numHeads / t_->numKvHeads;

        for (uint32_t task = begin; task < end; ++task) {
            const uint32_t b = task / t_->numKvHeads;
            const uint32_t kvh = task % t_->numKvHeads;
            for (uint32_t g = 0; g < group; ++g) {
                RunOneHead(b, kvh, kvh * group + g);
            }
        }
    }

private:
    /*
     * One (batch, query-head) attention: rotate q, stream the packed blocks with
     * an online softmax, rotate the accumulator back on exit.
     */
    __aicore__ inline void RunOneHead(uint32_t b, uint32_t kvh, uint32_t qh)
    {
        const uint32_t d = t_->headDim;
        LocalTensor<float> q = qBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> kv = kvBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<uint8_t> bytes = byteBuf_.Get<uint8_t>();
        LocalTensor<int32_t> codes = codeBuf_.Get<int32_t>();
        LocalTensor<float> red = redBuf_.Get<float>();
        LocalTensor<float> wrk = wrkBuf_.Get<float>();
        LocalTensor<float> dbg = dbgBuf_.Get<float>();
        LocalTensor<half> unpH = unpHBuf_.Get<half>();
        LocalTensor<float> unpF0 = unpF0Buf_.Get<float>();
        LocalTensor<float> unpF1 = unpF1Buf_.Get<float>();
        LocalTensor<half> kNrm = kNrmBuf_.Get<half>();
        LocalTensor<half> vNrm = vNrmBuf_.Get<half>();
        LocalTensor<float> sco = scoBuf_.Get<float>();
        LocalTensor<float> wgt = wgtBuf_.Get<float>();
        LocalTensor<float> rdx = rdxBuf_.Get<float>();

        // q -> fp32, then into the rotated basis (once per head, not per key)
        LocalTensor<half> qh16 = outBuf_.Get<half>();
        DataCopy(qh16, qGm_[(b * t_->numHeads + qh) * d], d);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);   // MTE2 -> V is NOT implicit on 310P
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        Cast(q, qh16, RoundMode::CAST_NONE, d);
        PipeBarrier<PIPE_V>();
        RotatePi(q, signs_, tmp, d, t_->invSqrtHeadDim);

        if (t_->variant == 9u) {
            // expose the rotated query: out = Pi q, compared against an
            // independently computed rotation in Python
            Adds(acc, q, 0.0f, d);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> dbg = outBuf_.Get<half>();
            Cast(dbg, acc, RoundMode::CAST_NONE, d);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
            DataCopy(outGm_[(b * t_->numHeads + qh) * d], dbg, d);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
            return;
        }
        Duplicate(acc, 0.0f, d);
        PipeBarrier<PIPE_V>();
        float runMax = kNegInf;
        float runSum = 0.0f;
        float dbgDot = -12345.0f;   // variant==5 probe: raw ReduceSum result at tk==0

        const int32_t seqLen = seqGm_.GetValue(b);
        const uint32_t nBlocks = (seqLen + t_->blockSize - 1) / t_->blockSize;

        for (uint32_t blk = 0; blk < nBlocks; ++blk) {
            const int32_t phys = btGm_.GetValue(b * t_->maxBlocksPerSeq + blk);
            if (phys < 0) {
                continue;
            }
            /*
             * Prefetch this block's norm planes. The layout is [slot, kv_head]
             * so one block's norms are blockSize*numKvHeads contiguous halves
             * starting at phys*blockSize*numKvHeads.
             */
            const uint32_t nrmCount = t_->blockSize * t_->numKvHeads;
            const uint64_t nrmBase = static_cast<uint64_t>(phys) * nrmCount;
            DataCopy(kNrm, kNormGm_[nrmBase], nrmCount);
            DataCopy(vNrm, vNormGm_[nrmBase], nrmCount);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID6);   // UB norms read by scalar below
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);

            const uint32_t tokBase = blk * t_->blockSize;
            uint32_t tokEnd = t_->blockSize;
            if (tokBase + tokEnd > static_cast<uint32_t>(seqLen)) {
                tokEnd = static_cast<uint32_t>(seqLen) - tokBase;
            }

            /*
             * BLOCK-WISE SOFTMAX (three passes over the block).
             *
             * The per-token form called ExpScalar TWICE per token -- a hand-rolled
             * range reduction, 7-term Taylor and binary-exponent loop, ~40 scalar
             * ops each. msprof showed the scalar pipe still at 0.87 after the
             * codec was vectorised, and that was the remaining consumer.
             *
             * Restructured so Exp runs ONCE per block as a vector op over all
             * tokens, leaving a single scalar exp per block for the running-max
             * correction. Per token the scalar work drops from ~80 ops to two
             * UB reads (the reduced dot, and the weight in pass 3).
             */

            // ---- pass 1: scores for every token in the block ----------------
            for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                GatherNz(kCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                if constexpr (TQPlanar<BITS>::kSupported) {
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                } else {
                    UnpackCodes<BITS>(bytes, codes, d);
                    SetFlag<HardEvent::S_V>(EVENT_ID1);
                    WaitFlag<HardEvent::S_V>(EVENT_ID1);
                }
                DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);
                Mul(tmp, q, kv, d);
                PipeBarrier<PIPE_V>();
                ReduceSum(tmp, tmp, tmp, d);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
                const float kNorm = static_cast<float>(kNrm.GetValue(tk * t_->numKvHeads + kvh));
                sco.SetValue(tk, tmp.GetValue(0) * kNorm * t_->scale);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);       // scalar wrote sco -> vector reads it
            WaitFlag<HardEvent::S_V>(EVENT_ID1);

            // pad the tail so the reductions run on a whole 32B block
            uint32_t padEnd = ((tokEnd + 7u) / 8u) * 8u;
            for (uint32_t tk = tokEnd; tk < padEnd; ++tk) {
                sco.SetValue(tk, kNegInf);            // excluded by the max, exp -> 0
            }
            if (padEnd > tokEnd) {
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }

            // ---- pass 2: vectorised softmax over the whole block ------------
            ReduceMax(rdx, sco, rdx, padEnd);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            const float blkMax = rdx.GetValue(0);
            const float newMax = (blkMax > runMax) ? blkMax : runMax;
            const float corr = (runMax == kNegInf) ? 0.0f : ExpScalar(runMax - newMax);

            Adds(sco, sco, -newMax, padEnd);
            PipeBarrier<PIPE_V>();
            Exp(wgt, sco, padEnd);                    // ONE vector exp for the block
            PipeBarrier<PIPE_V>();
            ReduceSum(rdx, wgt, rdx, padEnd);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            const float blkSum = rdx.GetValue(0);

            if (corr != 1.0f) {
                Muls(acc, acc, corr, d);
                PipeBarrier<PIPE_V>();
            }
            runSum = runSum * corr + blkSum;
            runMax = newMax;

            // ---- pass 3: accumulate w * ||v|| * yhat_v ----------------------
            for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                GatherNz(vCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                if constexpr (TQPlanar<BITS>::kSupported) {
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                } else {
                    UnpackCodes<BITS>(bytes, codes, d);
                    SetFlag<HardEvent::S_V>(EVENT_ID1);
                    WaitFlag<HardEvent::S_V>(EVENT_ID1);
                }
                DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);
                const float vNorm = static_cast<float>(vNrm.GetValue(tk * t_->numKvHeads + kvh));
                const float w = wgt.GetValue(tk);
                Muls(kv, kv, w * vNorm, d);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, kv, d);
                PipeBarrier<PIPE_V>();
            }
        }

        if (t_->variant != 6u && runSum > 0.0f) {
            Muls(acc, acc, 1.0f / runSum, d);
            PipeBarrier<PIPE_V>();
        }
        if (t_->variant == 10u || t_->variant == 11u || t_->variant == 12u) {
            LocalTensor<half> dh = outBuf_.Get<half>();
            Cast(dh, dbg, RoundMode::CAST_NONE, d);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
            DataCopy(outGm_[(b * t_->numHeads + qh) * d], dh, d);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
            return;
        }
        // Pi is self-inverse: one rotation returns the output to the original basis
        if (t_->variant != 6u) {
            RotatePi(acc, signs_, tmp, d, t_->invSqrtHeadDim);
        }

        /*
         * variant==5: overwrite the output with the RAW ReduceSum result from
         * tk==0, so Python can read what the reduction actually produced.
         * Expected: <Pi q, yhat_k>, i.e. <q, k>/||k||, order 1. A constant
         * (0, or the -12345 sentinel) localises the defect to the reduction
         * itself rather than to anything downstream.
         */
        if (t_->variant == 5u) {
            Duplicate(acc, dbgDot, d);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<half> out = outBuf_.Get<half>();
        Cast(out, acc, RoundMode::CAST_NONE, d);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID4);   // V -> MTE3 is NOT implicit
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
        DataCopy(outGm_[(b * t_->numHeads + qh) * d], out, d);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);   // guards `out`/`qh16` reuse next head
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }

    /*
     * Gather one vector's packed halves out of (numBlocks, C1, blockSize, 16).
     * Mirror of the write path's ScatterNz: a head occupies packedHalves/16
     * contiguous 16-element groups at a fixed stride. 16 halves = 32B, so these
     * are aligned copies -- 310P has no DataCopyPad.
     */
    __aicore__ inline void GatherNz(const GlobalTensor<half> &cache, const LocalTensor<uint8_t> &bytes,
                                    uint32_t h, uint32_t blk, uint32_t off)
    {
        const uint32_t ph = t_->packedHalves;
        const uint32_t groups = ph / kNzC0;
        LocalTensor<half> asHalf = bytes.ReinterpretCast<half>();
        /*
         * The per-group loop below used to issue `groups` separate DataCopys,
         * each recomputing a 64-bit address (two 64-bit multiplies). GatherNz
         * runs TWICE per token (K in pass 1, V in pass 3), so at d=256/b=4
         * (groups==4) that was ~16 64-bit multiplies per token on the SCALAR
         * pipe -- the third scalar consumer, hidden first behind the scalar
         * unpack and then behind ExpScalar.
         *
         * The group index only ever shifts the address by a CONSTANT stride:
         *     src_g = base + g * blockSize * kNzC0
         * so all `groups` bursts collapse into one strided DataCopy.
         *   burst      = kNzC0 halves = 32B = 1 block   -> blockLen 1
         *   consecutive g are blockSize blocks apart    -> gap blockSize-1
         *   dst is contiguous in UB                     -> dstStride 0
         * srcStride is the GAP in 32B units, exclusive of the burst
         * (kernel_operator_data_copy_impl: srcStride310 = srcStride*32 + burstLength).
         */
        const uint32_t c1Base = (h * ph) / kNzC0;
        const uint64_t src =
            ((static_cast<uint64_t>(blk) * t_->c1 + c1Base) * t_->blockSize + off) * kNzC0;
        const DataCopyParams nzParams{static_cast<uint16_t>(groups), 1,
                                      static_cast<uint16_t>(t_->blockSize - 1), 0};
        DataCopy(asHalf, cache[src], nzParams);
        /*
         * The consumer of `bytes` depends on the code path:
         *   scalar UnpackCodes  -> GetValue on the SCALAR pipe  => MTE2 -> S
         *   UnpackCodesVec      -> Cast on the VECTOR pipe      => MTE2 -> V
         * MTE2->V is explicitly NOT implicitly drained on 310P, so switching the
         * unpack to vector ops silently removed the only guarantee that the DMA
         * had landed. Symptom: codes correct for token 0 (nothing in flight) but
         * attention degrading over a full context, and worse at b=2 (4 planes,
         * more vector ops, wider race window) than b=4 (2 planes).
         */
        if constexpr (TQPlanar<BITS>::kSupported) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }

    /*
     * Scalar exp for the online-softmax weights.
     *
     * Deliberately NOT the vector Exp(): pushing one scalar through a UB tensor
     * round-trip per key would cost far more than it computes, and it forced the
     * helper non-const. This is a plain series/range-reduction evaluation:
     * exp(x) = 2^k * exp(r), with r in [-ln2/2, ln2/2] and a 7-term Taylor
     * expansion -- accurate to ~1e-7 over the range softmax produces (x <= 0
     * after max-subtraction, and anything below -80 flushes to zero anyway).
     */
    __aicore__ inline float ExpScalar(float x)
    {
        if (t_->variant == 13u) {
            return 1.0f;   // ATTRIBUTION PROBE ONLY -- wrong results by design
        }
        if (x < -80.0f) {
            return 0.0f;
        }
        if (x > 80.0f) {
            x = 80.0f;
        }
        constexpr float kLn2 = 0.6931471805599453f;
        constexpr float kInvLn2 = 1.4426950408889634f;
        const int k = static_cast<int>((x * kInvLn2) + ((x >= 0.0f) ? 0.5f : -0.5f));
        const float r = x - static_cast<float>(k) * kLn2;
        float term = 1.0f;
        float sum = 1.0f;
#pragma unroll
        for (int i = 1; i <= 7; ++i) {
            term *= r / static_cast<float>(i);
            sum += term;
        }
        // 2^k by exponent construction; k is small here (|x| <= 80 => |k| <= 116)
        float scale = 1.0f;
        int n = (k < 0) ? -k : k;
        float base = (k < 0) ? 0.5f : 2.0f;
        while (n > 0) {
            if (n & 1) {
                scale *= base;
            }
            base *= base;
            n >>= 1;
        }
        return sum * scale;
    }

    const TurboquantPagedAttentionV310TilingData *t_;
    GlobalTensor<half> qGm_, kCacheGm_, vCacheGm_, kNormGm_, vNormGm_, outGm_;
    GlobalTensor<int32_t> btGm_, seqGm_;
    GlobalTensor<float> signGm_, cbGm_;
    TBuf<TPosition::VECCALC> qBuf_, accBuf_, kvBuf_, tmpBuf_, byteBuf_, codeBuf_, signBuf_, cbBuf_,
        outBuf_, redBuf_, wrkBuf_, dbgBuf_, unpHBuf_, unpF0Buf_, unpF1Buf_, kNrmBuf_, vNrmBuf_,
        scoBuf_, wgtBuf_, rdxBuf_;
    LocalTensor<float> signs_, cb_;
};

}  // namespace TurboQuantRead

#endif  // TURBOQUANT_PAGED_ATTENTION_V310_H
