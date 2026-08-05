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

        signs_ = signBuf_.Get<float>();
        DataCopy(signs_, signGm_, d);
        cb_ = cbBuf_.Get<float>();
        if (t_->codebookMode == TQ_CB_LUT) {
            DataCopy(cb_, cbGm_, 2 * (1 << BITS));
        }
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

        // q -> fp32, then into the rotated basis (once per head, not per key)
        LocalTensor<half> qh16 = outBuf_.Get<half>();
        DataCopy(qh16, qGm_[(b * t_->numHeads + qh) * d], d);
        PipeBarrier<PIPE_ALL>();
        Cast(q, qh16, RoundMode::CAST_NONE, d);
        PipeBarrier<PIPE_V>();
        RotatePi(q, signs_, tmp, d, t_->invSqrtHeadDim);

        Duplicate(acc, 0.0f, d);
        PipeBarrier<PIPE_V>();
        float runMax = kNegInf;
        float runSum = 0.0f;

        const int32_t seqLen = seqGm_.GetValue(b);
        const uint32_t nBlocks = (seqLen + t_->blockSize - 1) / t_->blockSize;

        for (uint32_t blk = 0; blk < nBlocks; ++blk) {
            const int32_t phys = btGm_.GetValue(b * t_->maxBlocksPerSeq + blk);
            if (phys < 0) {
                continue;
            }
            const uint32_t tokBase = blk * t_->blockSize;
            uint32_t tokEnd = t_->blockSize;
            if (tokBase + tokEnd > static_cast<uint32_t>(seqLen)) {
                tokEnd = static_cast<uint32_t>(seqLen) - tokBase;
            }

            for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                const uint32_t slot = static_cast<uint32_t>(phys) * t_->blockSize + tk;

                // ---- score: <Pi q, yhat_k> * ||k|| * scale -------------------
                GatherNz(kCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                UnpackCodes<BITS>(bytes, codes, d);
                PipeBarrier<PIPE_V>();
                DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);

                Mul(tmp, q, kv, d);
                PipeBarrier<PIPE_V>();
                ReduceSum(tmp, tmp, tmp, d);
                PipeBarrier<PIPE_V>();
                const float kNorm = static_cast<float>(
                    kNormGm_.GetValue(static_cast<uint64_t>(slot) * t_->numKvHeads + kvh));
                const float score = tmp.GetValue(0) * kNorm * t_->scale;

                // ---- online softmax update ---------------------------------
                const float newMax = (score > runMax) ? score : runMax;
                const float corr = (runMax == kNegInf) ? 0.0f : ExpScalar(runMax - newMax);
                const float w = ExpScalar(score - newMax);
                if (corr != 1.0f) {
                    Muls(acc, acc, corr, d);
                    PipeBarrier<PIPE_V>();
                }
                runSum = runSum * corr + w;
                runMax = newMax;

                // ---- accumulate w * ||v|| * yhat_v (still rotated) ----------
                GatherNz(vCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                UnpackCodes<BITS>(bytes, codes, d);
                PipeBarrier<PIPE_V>();
                DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);
                const float vNorm = static_cast<float>(
                    vNormGm_.GetValue(static_cast<uint64_t>(slot) * t_->numKvHeads + kvh));
                Muls(kv, kv, w * vNorm, d);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, kv, d);
                PipeBarrier<PIPE_V>();
            }
        }

        if (runSum > 0.0f) {
            Muls(acc, acc, 1.0f / runSum, d);
            PipeBarrier<PIPE_V>();
        }
        // Pi is self-inverse: one rotation returns the output to the original basis
        RotatePi(acc, signs_, tmp, d, t_->invSqrtHeadDim);

        LocalTensor<half> out = outBuf_.Get<half>();
        Cast(out, acc, RoundMode::CAST_NONE, d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm_[(b * t_->numHeads + qh) * d], out, d);
        PipeBarrier<PIPE_ALL>();
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
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t c1 = (h * ph) / kNzC0 + g;
            const uint64_t src =
                ((static_cast<uint64_t>(blk) * t_->c1 + c1) * t_->blockSize + off) * kNzC0;
            DataCopy(asHalf[g * kNzC0], cache[src], kNzC0);
        }
        PipeBarrier<PIPE_ALL>();
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
        outBuf_;
    LocalTensor<float> signs_, cb_;
};

}  // namespace TurboQuantRead

#endif  // TURBOQUANT_PAGED_ATTENTION_V310_H
