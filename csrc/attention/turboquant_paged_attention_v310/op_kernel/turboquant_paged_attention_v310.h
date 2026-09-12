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
/*
 * Tokens batched into a single set of vector ops. The scalar bound is
 * instruction ISSUE + PipeBarrier (~20 cycles each, measured), not scalar
 * arithmetic, so the lever is fewer/larger vector ops rather than fewer
 * scalar statements. 8 keeps the tile buffers ~36KB of the 256KB UB.
 */
constexpr uint32_t kTile = 8;
/*
 * Query heads sharing one KV head that are decoded together. Process() used to
 * call RunOneHead per query head and each call re-gathered and re-unpacked the
 * WHOLE KV cache, so at group=4 both the MTE2 traffic and the dominant
 * unpack/dequant vector work were 4x redundant (25.2 MB of GM traffic for a
 * 6.29 MB cache). Larger groups are handled in chunks of this size.
 */
constexpr uint32_t kMaxGroup = 4;

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

        /*
         * TILED BUFFERS. msprof (b=2 vs b=4, identical element-work, 2x the
         * instruction count) measured ~20 cycles per vector instruction of pure
         * issue + PipeBarrier overhead. Each op covered only d=256 fp32 = 4
         * vector repeats, so ~20 cycles of drain wrapped ~4 cycles of work.
         * Batching kTile tokens per vector op amortises that over kTile x the
         * data; element-work is unchanged, so vec_time is the floor.
         *
         * Sized for kTile tokens. UnpackCodesVec is len-generic, so the tile
         * simply passes len = kTile*d. Scratch is `lanes` = kTile*d/kPlanes,
         * maximal when kPlanes is smallest (2, at BITS==4) -> kTile*d/2.
         */
        pipe->InitBuffer(qBuf_, kMaxGroup * d * sizeof(float));
        pipe->InitBuffer(accBuf_, kMaxGroup * d * sizeof(float));
        pipe->InitBuffer(kvBuf_, kTile * d * sizeof(float));   // ONE shared K/V tile
        pipe->InitBuffer(prodBuf_, kTile * d * sizeof(float));  // q*kv per query head
        pipe->InitBuffer(tmpBuf_, d * sizeof(float));
        pipe->InitBuffer(byteBuf_, kTile * TQTraits<BITS>::PackedBytes(d));
        pipe->InitBuffer(codeBuf_, kTile * d * sizeof(int32_t));
        // q permuted to plane-major, one copy per query head in the group
        pipe->InitBuffer(qbBuf_, kMaxGroup * kTile * d * sizeof(float));
        pipe->InitBuffer(signBuf_, d * sizeof(float));
        pipe->InitBuffer(cbBuf_, (2 * (1 << BITS)) * sizeof(float));
        pipe->InitBuffer(outBuf_, d * sizeof(half));
        pipe->InitBuffer(redBuf_, d * sizeof(float));   // ReduceSum dst (must not alias src)
        pipe->InitBuffer(wrkBuf_, d * sizeof(float));
        pipe->InitBuffer(dbgBuf_, d * sizeof(float));
        pipe->InitBuffer(unpHBuf_, (kTile * d / 2) * sizeof(half));      // planar unpack scratch
        pipe->InitBuffer(unpF0Buf_, (kTile * d / 2) * sizeof(float));
        pipe->InitBuffer(unpF1Buf_, (kTile * d / 2) * sizeof(float));
        // norm planes prefetched per BLOCK: replaces two GM scalar reads per
        // token (dependent global loads, pure latency) with one DataCopy.
        // norm plane is [num_slots, kNzC0] halves: each slot owns a whole 32B
        // block (see the write kernel). 4KB each at blockSize=128.
        pipe->InitBuffer(kNrmBuf_, t_->blockSize * kNzC0 * sizeof(half));
        pipe->InitBuffer(vNrmBuf_, t_->blockSize * kNzC0 * sizeof(half));
        /*
         * Block-wise softmax: one score/weight slot per token in a block, but
         * rounded UP to a whole tile. Both the tile reduction
         * (WholeReduceSum writes kTile slots from t0) and the kNegInf pad run to
         * ceil(tokEnd/8)*8, either of which writes past blockSize the moment
         * blockSize is not a multiple of kTile. Dormant at blockSize=128.
         */
        const uint32_t scoSlots = ((t_->blockSize + kTile - 1) / kTile) * kTile;
        pipe->InitBuffer(scoBuf_, kMaxGroup * scoSlots * sizeof(float));
        pipe->InitBuffer(wgtBuf_, kMaxGroup * scoSlots * sizeof(float));
        // ReduceMax/ReduceSum workLocal over padEnd (<= scoSlots); must not alias src/dst
        pipe->InitBuffer(rdxBuf_, scoSlots * sizeof(float));
        /*
         * Per-BLOCK norm columns, hoisted out of the per-token inner loop.
         * kNrm/vNrm are [slot][kv_head] halves, so head kvh's column is strided
         * by numKvHeads and cannot be loaded contiguously. Building it once per
         * block costs 2 scalar ops per token; leaving it inline cost
         * (1 + nq*2) scalar ops per token -- 72 per tile at nq=4, against only
         * ~36 vector ops per tile. It is invariant across the whole block AND
         * across every query head in the group, so it is built once and reused.
         */
        pipe->InitBuffer(kColBuf_, scoSlots * sizeof(float));
        pipe->InitBuffer(vColBuf_, scoSlots * sizeof(float));
        pipe->InitBuffer(wvBuf_, kMaxGroup * scoSlots * sizeof(float));

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
            /*
             * Production path: decode the whole GQA group together so each K/V
             * tile is gathered and unpacked ONCE instead of `group` times.
             * Groups larger than kMaxGroup are split into chunks, each of which
             * still amortises the shared work across its members.
             */
            if constexpr (TQPlanar<BITS>::kSupported) {
                if (t_->variant == 0u) {
                    for (uint32_t g0 = 0; g0 < group; g0 += kMaxGroup) {
                        uint32_t nq = group - g0;
                        if (nq > kMaxGroup) {
                            nq = kMaxGroup;
                        }
                        RunKvGroup(b, kvh, kvh * group + g0, nq);
                    }
                    continue;
                }
            }
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
    /*
     * GQA-SHARED DECODE. Every query head sharing KV head `kvh` is decoded in one
     * pass, so each K/V tile is gathered, unpacked and dequantised ONCE and then
     * dotted against all `nq` queries.
     *
     * RunOneHead (below) is still correct but re-reads the whole cache per query
     * head: at group=4 that made the MTE2 traffic and the dominant unpack/dequant
     * vector work 4x redundant -- 25.17 MB of GM traffic against a 6.29 MB cache,
     * 3.17 GB/s achieved on a ~205 GB/s part. Only the SHARED work is amortised
     * here; the dots and the Axpy still scale with the group, as they must.
     *
     * Used for the production path only (variant==0) and only where the planar
     * codec applies; every debug variant and BITS==3 keep the per-head path, so
     * the b=3 control and all probes are untouched.
     */
    __aicore__ inline void RunKvGroup(uint32_t b, uint32_t kvh, uint32_t qh0, uint32_t nq)
    {
        using P = TQPlanar<BITS>;
        const uint32_t d = t_->headDim;
        const uint32_t lanes = kTile * d / P::kPlanes;
        const uint32_t lpt = d / P::kPlanes;
        const uint32_t scoSlots = ((t_->blockSize + kTile - 1) / kTile) * kTile;

        LocalTensor<float> q = qBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> kv = kvBuf_.Get<float>();
        LocalTensor<float> prod = prodBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<uint8_t> bytes = byteBuf_.Get<uint8_t>();
        LocalTensor<int32_t> codes = codeBuf_.Get<int32_t>();
        LocalTensor<half> unpH = unpHBuf_.Get<half>();
        LocalTensor<float> unpF0 = unpF0Buf_.Get<float>();
        LocalTensor<float> unpF1 = unpF1Buf_.Get<float>();
        LocalTensor<half> kNrm = kNrmBuf_.Get<half>();
        LocalTensor<half> vNrm = vNrmBuf_.Get<half>();
        LocalTensor<float> sco = scoBuf_.Get<float>();
        LocalTensor<float> wgt = wgtBuf_.Get<float>();
        LocalTensor<float> rdx = rdxBuf_.Get<float>();
        LocalTensor<float> qb = qbBuf_.Get<float>();
        LocalTensor<float> kCol = kColBuf_.Get<float>();
        LocalTensor<float> vCol = vColBuf_.Get<float>();
        LocalTensor<float> wv = wvBuf_.Get<float>();
        LocalTensor<half> qh16 = outBuf_.Get<half>();

        float runMax[kMaxGroup];
        float runSum[kMaxGroup];

        // ---- per-query setup: rotate q, permute into the tiled code layout ----
        for (uint32_t g = 0; g < nq; ++g) {
            DataCopy(qh16, qGm_[(b * t_->numHeads + qh0 + g) * d], d);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);   // MTE2 -> V is NOT implicit on 310P
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            Cast(q[g * d], qh16, RoundMode::CAST_NONE, d);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);   // next g overwrites qh16
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
            RotatePi(q[g * d], signs_, tmp, d, t_->invSqrtHeadDim);
            for (uint32_t pl = 0; pl < P::kPlanes; ++pl) {
                for (uint32_t t = 0; t < kTile; ++t) {
                    Adds(qb[g * kTile * d + pl * lanes + t * lpt], q[g * d + pl * lpt], 0.0f, lpt);
                    PipeBarrier<PIPE_V>();
                }
            }
            runMax[g] = kNegInf;
            runSum[g] = 0.0f;
        }
        Duplicate(acc, 0.0f, nq * d);
        PipeBarrier<PIPE_V>();

        const int32_t seqLen = seqGm_.GetValue(b);
        const uint32_t nBlocks = (seqLen + t_->blockSize - 1) / t_->blockSize;

        for (uint32_t blk = 0; blk < nBlocks; ++blk) {
            const int32_t phys = btGm_.GetValue(b * t_->maxBlocksPerSeq + blk);
            if (phys < 0) {
                continue;
            }
            const uint32_t nrmCount = t_->blockSize * kNzC0;
            const uint64_t nrmBase = static_cast<uint64_t>(phys) * nrmCount;
            DataCopy(kNrm, kNormGm_[nrmBase], nrmCount);
            DataCopy(vNrm, vNormGm_[nrmBase], nrmCount);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);

            const uint32_t tokBase = blk * t_->blockSize;
            uint32_t tokEnd = t_->blockSize;
            if (tokBase + tokEnd > static_cast<uint32_t>(seqLen)) {
                tokEnd = static_cast<uint32_t>(seqLen) - tokBase;
            }

            /*
             * Hoist the strided norm columns out of the inner loops: 2 scalar
             * ops per token here, once per BLOCK, replacing (1 + nq*2) per token
             * per tile. t_->scale is folded into kCol so the score scaling later
             * is a single vector Mul per query.
             */
            for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                const uint32_t ni = tk * kNzC0 + kvh;
                kCol.SetValue(tk, static_cast<float>(kNrm.GetValue(ni)) * t_->scale);
                vCol.SetValue(tk, static_cast<float>(vNrm.GetValue(ni)));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);

            // ---- pass 1: gather K ONCE per tile, dot against every query ----
            for (uint32_t t0 = 0; t0 < tokEnd; t0 += kTile) {
                GatherNzTile(kCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), t0, kTile);
                if (t_->codebookMode == TQ_CB_UNIFORM) {
                    // fused: skips the int32 round trip, ~25% less element-work
                    UnpackDequantVec<BITS>(bytes, kv, unpH, unpF0, unpF1, kTile * d,
                                           t_->invSqrtHeadDim);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                } else {
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, kTile * d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    DequantizeVec<BITS>(codes, kv, cb_, kTile * d, t_->invSqrtHeadDim,
                                        t_->codebookMode);
                }
                for (uint32_t g = 0; g < nq; ++g) {
                    // kv must SURVIVE for the next query, so the product goes to prod
                    Mul(prod, kv, qb[g * kTile * d], kTile * d);
                    PipeBarrier<PIPE_V>();
                    for (uint32_t pl = 1; pl < P::kPlanes; ++pl) {
                        Add(prod, prod, prod[pl * lanes], lanes);
                        PipeBarrier<PIPE_V>();
                    }
                    LocalTensor<float> rsrc = prod;
                    if (lpt > 64) {
                        const uint8_t sRep = static_cast<uint8_t>(lpt / 8);
                        Add(prod[kTile * lpt], prod, prod[64], 64, kTile,
                            BinaryRepeatParams{1, 1, 1, 8, sRep, sRep});
                        PipeBarrier<PIPE_V>();
                        rsrc = prod[kTile * lpt];
                    }
                    WholeReduceSum(sco[g * scoSlots + t0], rsrc, 64, kTile, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                }
                // raw dots land in sco; the kNorm*scale factor is applied as ONE
                // vector Mul per query below, not per token here
            }
            for (uint32_t g = 0; g < nq; ++g) {
                Mul(sco[g * scoSlots], sco[g * scoSlots], kCol, tokEnd);
                PipeBarrier<PIPE_V>();
            }

            const uint32_t padEnd = ((tokEnd + 7u) / 8u) * 8u;
            if (padEnd > tokEnd) {
                for (uint32_t g = 0; g < nq; ++g) {
                    for (uint32_t tk = tokEnd; tk < padEnd; ++tk) {
                        sco.SetValue(g * scoSlots + tk, kNegInf);
                    }
                }
                SetFlag<HardEvent::S_V>(EVENT_ID1);
                WaitFlag<HardEvent::S_V>(EVENT_ID1);
            }

            // ---- pass 2: independent online softmax per query ----
            for (uint32_t g = 0; g < nq; ++g) {
                ReduceMax(rdx, sco[g * scoSlots], rdx, padEnd);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
                const float blkMax = rdx.GetValue(0);
                const float newMax = (blkMax > runMax[g]) ? blkMax : runMax[g];
                const float corr = (runMax[g] == kNegInf) ? 0.0f : ExpScalar(runMax[g] - newMax);
                Adds(sco[g * scoSlots], sco[g * scoSlots], -newMax, padEnd);
                PipeBarrier<PIPE_V>();
                Exp(wgt[g * scoSlots], sco[g * scoSlots], padEnd);
                PipeBarrier<PIPE_V>();
                ReduceSum(rdx, wgt[g * scoSlots], rdx, padEnd);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
                const float blkSum = rdx.GetValue(0);
                if (corr != 1.0f) {
                    Muls(acc[g * d], acc[g * d], corr, d);
                    PipeBarrier<PIPE_V>();
                }
                runSum[g] = runSum[g] * corr + blkSum;
                runMax[g] = newMax;
            }

            // fold ||v|| into the weights: one vector Mul per query, so pass 3
            // reads ONE scalar per (token, query) instead of two
            for (uint32_t g = 0; g < nq; ++g) {
                Mul(wv[g * scoSlots], wgt[g * scoSlots], vCol, tokEnd);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);

            // ---- pass 3: gather V ONCE per tile, scatter into every acc ----
            for (uint32_t t0 = 0; t0 < tokEnd; t0 += kTile) {
                GatherNzTile(vCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), t0, kTile);
                if (t_->codebookMode == TQ_CB_UNIFORM) {
                    // fused: skips the int32 round trip, ~25% less element-work
                    UnpackDequantVec<BITS>(bytes, kv, unpH, unpF0, unpF1, kTile * d,
                                           t_->invSqrtHeadDim);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                } else {
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, kTile * d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    DequantizeVec<BITS>(codes, kv, cb_, kTile * d, t_->invSqrtHeadDim,
                                        t_->codebookMode);
                }
                for (uint32_t t = 0; t < kTile; ++t) {
                    const uint32_t tk = t0 + t;
                    if (tk >= tokEnd) {
                        break;
                    }
                    for (uint32_t g = 0; g < nq; ++g) {
                        const float w = wv.GetValue(g * scoSlots + tk);
                        for (uint32_t pl = 0; pl < P::kPlanes; ++pl) {
                            Axpy(acc[g * d + pl * lpt], kv[pl * lanes + t * lpt], w, lpt);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
        }

        // ---- finalise: normalise, rotate back, store, one query at a time ----
        LocalTensor<half> out = outBuf_.Get<half>();
        for (uint32_t g = 0; g < nq; ++g) {
            if (runSum[g] > 0.0f) {
                Muls(acc[g * d], acc[g * d], 1.0f / runSum[g], d);
                PipeBarrier<PIPE_V>();
            }
            // Pi is self-inverse: one rotation returns the output to the original basis
            RotatePi(acc[g * d], signs_, tmp, d, t_->invSqrtHeadDim);
            Cast(out, acc[g * d], RoundMode::CAST_NONE, d);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
            DataCopy(outGm_[(b * t_->numHeads + qh0 + g) * d], out, d);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);   // next g overwrites `out`
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        }
    }

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
        LocalTensor<float> qb = qbBuf_.Get<float>();

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
        /*
         * Permute q into the tiled code layout, ONCE per head (not per token).
         * The tile puts token t's plane-pl lanes at [pl*lanes + t*lpt], so q's
         * plane-pl slice must be replicated kTile times inside each plane:
         *     qb[pl*lanes + t*lpt + j] = q[pl*lpt + j]
         * For a SINGLE token this permutation is the identity, which is why the
         * per-token path could multiply by q directly.
         */
        if constexpr (TQPlanar<BITS>::kSupported) {
            using P = TQPlanar<BITS>;
            const uint32_t lanes = kTile * d / P::kPlanes;
            const uint32_t lpt = d / P::kPlanes;
            for (uint32_t pl = 0; pl < P::kPlanes; ++pl) {
                for (uint32_t t = 0; t < kTile; ++t) {
                    Adds(qb[pl * lanes + t * lpt], q[pl * lpt], 0.0f, lpt);
                    PipeBarrier<PIPE_V>();
                }
            }
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
            const uint32_t nrmCount = t_->blockSize * kNzC0;
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
            if constexpr (TQPlanar<BITS>::kSupported) {
                /*
                 * TILED. UnpackCodesVec is plane-major and len-generic, so
                 * len = kTile*d over token-contiguous bytes gives
                 *     codes[pl*lanes + t*lpt + j]  <->  token t, dim pl*lpt+j
                 * with lanes = kTile*d/kPlanes and lpt = lanes/kTile.
                 * qb_ is q permuted into that same layout ONCE per head, so the
                 * elementwise Mul below is a plain kTile*d op.
                 *
                 * Tiles always read a full kTile even in the sequence tail --
                 * blockSize is a multiple of kTile so the reads stay inside the
                 * allocated block, and the surplus tokens get kNegInf scores
                 * below and are killed by the softmax.
                 */
                using P = TQPlanar<BITS>;
                const uint32_t lanes = kTile * d / P::kPlanes;
                const uint32_t lpt = d / P::kPlanes;           // lanes per token
                for (uint32_t t0 = 0; t0 < tokEnd; t0 += kTile) {
                    GatherNzTile(kCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), t0, kTile);
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, kTile * d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    DequantizeVec<BITS>(codes, kv, cb_, kTile * d, t_->invSqrtHeadDim,
                                        t_->codebookMode);
                    Mul(kv, kv, qb, kTile * d);
                    PipeBarrier<PIPE_V>();
                    // fold the planes: dot = sum over ALL dims = sum_pl sum_j
                    for (uint32_t pl = 1; pl < P::kPlanes; ++pl) {
                        Add(kv, kv, kv[pl * lanes], lanes);
                        PipeBarrier<PIPE_V>();
                    }
                    /*
                     * Each token's lpt partials are CONTIGUOUS at kv[t*lpt], so
                     * the whole tile reduces with ONE WholeReduceSum writing the
                     * kTile dots straight into sco -- replacing kTile ReduceSums,
                     * kTile barriers and kTile V_S flag pairs.
                     *
                     * fp32 caps a repeat at 64 elements, so lpt=128 (BITS==4) is
                     * first halved with a single strided Add; lpt=64 (BITS==2)
                     * reduces directly. Scratch lives above the folded region:
                     * kv holds kTile*d but only kTile*lpt = kTile*d/kPlanes is
                     * live after the plane fold.
                     */
                    LocalTensor<float> rsrc = kv;
                    if (lpt > 64) {
                        const uint8_t sRep = static_cast<uint8_t>(lpt / 8);   // lpt fp32 in 32B blocks
                        Add(kv[kTile * lpt], kv, kv[64], 64, kTile,
                            BinaryRepeatParams{1, 1, 1, 8, sRep, sRep});
                        PipeBarrier<PIPE_V>();
                        rsrc = kv[kTile * lpt];
                    }
                    WholeReduceSum(sco[t0], rsrc, 64, kTile, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    WaitFlag<HardEvent::V_S>(EVENT_ID2);
                    for (uint32_t t = 0; t < kTile; ++t) {
                        const uint32_t tk = t0 + t;
                        if (tk < tokEnd) {
                            const float kNorm =
                                static_cast<float>(kNrm.GetValue(tk * kNzC0 + kvh));
                            sco.SetValue(tk, sco.GetValue(tk) * kNorm * t_->scale);
                        }
                    }
                    SetFlag<HardEvent::S_V>(EVENT_ID1);   // scalar rewrote sco -> next tile's V ops
                    WaitFlag<HardEvent::S_V>(EVENT_ID1);
                }
            } else {
                for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                    GatherNz(kCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                    UnpackCodes<BITS>(bytes, codes, d);
                    SetFlag<HardEvent::S_V>(EVENT_ID1);
                    WaitFlag<HardEvent::S_V>(EVENT_ID1);
                    DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);
                    Mul(tmp, q, kv, d);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(tmp, tmp, tmp, d);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    WaitFlag<HardEvent::V_S>(EVENT_ID2);
                    const float kNorm = static_cast<float>(kNrm.GetValue(tk * kNzC0 + kvh));
                    sco.SetValue(tk, tmp.GetValue(0) * kNorm * t_->scale);
                }
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
            if constexpr (TQPlanar<BITS>::kSupported) {
                /*
                 * The gather/unpack/dequant batches exactly as in pass 1, but the
                 * accumulation itself does NOT batch away: acc[dim] += w_t*v[t][dim]
                 * needs a per-token scalar. In the tiled layout token t's values
                 * for plane pl sit at kv[pl*lanes + t*lpt], while acc is dim-ordered
                 * (dim = pl*lpt + j), so one Axpy per plane scatters them home.
                 * That is kPlanes ops/token -- the same count as the old
                 * Muls+Add, so nothing regresses while the unpack gets amortised.
                 */
                using P = TQPlanar<BITS>;
                const uint32_t lanes = kTile * d / P::kPlanes;
                const uint32_t lpt = d / P::kPlanes;
                for (uint32_t t0 = 0; t0 < tokEnd; t0 += kTile) {
                    GatherNzTile(vCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), t0, kTile);
                    UnpackCodesVec<BITS>(bytes, codes, unpH, unpF0, unpF1, kTile * d);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                    DequantizeVec<BITS>(codes, kv, cb_, kTile * d, t_->invSqrtHeadDim,
                                        t_->codebookMode);
                    for (uint32_t t = 0; t < kTile; ++t) {
                        const uint32_t tk = t0 + t;
                        if (tk >= tokEnd) {
                            break;
                        }
                        const float vNorm =
                            static_cast<float>(vNrm.GetValue(tk * kNzC0 + kvh));
                        const float w = wgt.GetValue(tk);
                        const float wv = w * vNorm;
                        for (uint32_t pl = 0; pl < P::kPlanes; ++pl) {
                            Axpy(acc[pl * lpt], kv[pl * lanes + t * lpt], wv, lpt);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                }
            } else {
                for (uint32_t tk = 0; tk < tokEnd; ++tk) {
                    GatherNz(vCacheGm_, bytes, kvh, static_cast<uint32_t>(phys), tk);
                    UnpackCodes<BITS>(bytes, codes, d);
                    SetFlag<HardEvent::S_V>(EVENT_ID1);
                    WaitFlag<HardEvent::S_V>(EVENT_ID1);
                    DequantizeVec<BITS>(codes, kv, cb_, d, t_->invSqrtHeadDim, t_->codebookMode);
                    const float vNorm = static_cast<float>(vNrm.GetValue(tk * kNzC0 + kvh));
                    const float w = wgt.GetValue(tk);
                    Muls(kv, kv, w * vNorm, d);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, kv, d);
                    PipeBarrier<PIPE_V>();
                }
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
     * Gather kTile consecutive tokens in one shot, laid out TOKEN-CONTIGUOUS so
     * the batched unpack sees exactly the byte order the single-token path saw.
     *
     * Within one c1 plane consecutive tokens ARE contiguous in GM (off varies
     * fastest), so each group is one burst of nTok blocks. Writing with
     * dstStride = groups-1 interleaves the groups back together:
     *     dst(g, t) = g*kNzC0 + t*groups*kNzC0
     * i.e. token t's `groups` chunks land adjacent, giving [t][g][kNzC0].
     *
     * This issues `groups` DataCopys per TILE instead of per token.
     */
    __aicore__ inline void GatherNzTile(const GlobalTensor<half> &cache, const LocalTensor<uint8_t> &bytes,
                                        uint32_t h, uint32_t blk, uint32_t off0, uint32_t nTok)
    {
        const uint32_t ph = t_->packedHalves;
        const uint32_t groups = ph / kNzC0;
        LocalTensor<half> asHalf = bytes.ReinterpretCast<half>();
        /*
         * Never read past the block. The caller always asks for a full kTile so
         * the unpack length (and therefore the qb permutation) stays constant,
         * but if blockSize is not a multiple of kTile the surplus slots would
         * run into the next c1 plane -- or off the end of the cache on the last
         * block. Those slots are discarded by the tk >= tokEnd guard, so leaving
         * them stale is fine; reading out of bounds is not.
         */
        const uint32_t avail = t_->blockSize - off0;
        if (nTok > avail) {
            nTok = avail;
        }
        const uint32_t c1Base = (h * ph) / kNzC0;
        const uint64_t base =
            ((static_cast<uint64_t>(blk) * t_->c1 + c1Base) * t_->blockSize + off0) * kNzC0;
        const uint64_t planeStride = static_cast<uint64_t>(t_->blockSize) * kNzC0;
        const DataCopyParams p{static_cast<uint16_t>(nTok), 1, 0, static_cast<uint16_t>(groups - 1)};
        for (uint32_t g = 0; g < groups; ++g) {
            DataCopy(asHalf[g * kNzC0], cache[base + g * planeStride], p);
        }
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
        scoBuf_, wgtBuf_, rdxBuf_, qbBuf_, prodBuf_, kColBuf_, vColBuf_, wvBuf_;
    LocalTensor<float> signs_, cb_;
};

}  // namespace TurboQuantRead

#endif  // TURBOQUANT_PAGED_ATTENTION_V310_H
