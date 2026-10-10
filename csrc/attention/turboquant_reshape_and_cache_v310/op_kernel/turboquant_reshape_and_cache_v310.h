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
 * TurboQuant write path: rotate -> norm -> quantize -> pack -> scatter to the
 * FRACTAL_NZ paged cache, plus a separate fp16 norm plane.
 *
 * Norms live OUTSIDE the packed plane deliberately: 96 packed bytes + a 2-byte
 * norm is 98, not a multiple of the 32B NZ tile, and padding to 128 would drop
 * compression 5.22x -> 4.0x. A separate plane costs ~2% and keeps 5.22x.
 */
#ifndef TURBOQUANT_RESHAPE_AND_CACHE_V310_H
#define TURBOQUANT_RESHAPE_AND_CACHE_V310_H

#include "kernel_operator.h"
#include "turboquant_reshape_and_cache_v310_codec.h"
#include "turboquant_reshape_and_cache_v310_tiling_data.h"

namespace TurboQuantWrite {

using namespace AscendC;
using namespace TurboQuant;

constexpr int32_t kNzC0 = 16;        // fp16 NZ tile on 310P
constexpr int32_t kBufferNum = 1;    // single-buffered; the vector work dominates
/*
 * Norm staging block: 16 halves = 32B, the smallest aligned GM write. A token's
 * numKvHeads norms always fall inside ONE such block (index = slot*numKvHeads + h,
 * numKvHeads <= 16), so one atomic-add DataCopy per plane per token replaces the
 * racy per-head scalar stores.
 */
constexpr int32_t kNrmBlk = 16;

struct TQWriteParams {
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR keyCache;
    GM_ADDR valueCache;
    GM_ADDR slotMapping;
    GM_ADDR keyNorms;
    GM_ADDR valueNorms;
    GM_ADDR signs;        // D vector, +-1.0f, length headDim
    GM_ADDR centroids;    // Lloyd-Max table (LEVELS) + midpoints (LEVELS-1); unused when uniform
};

template <int BITS>
class TQReshapeAndCache {
public:
    __aicore__ inline TQReshapeAndCache(const TurboquantReshapeAndCacheV310TilingData *t) : t_(t) {}

    __aicore__ inline void Init(const TQWriteParams &p, TPipe *pipe)
    {
        const uint32_t d = t_->headDim;
        keyGm_.SetGlobalBuffer((__gm__ half *)p.key);
        valGm_.SetGlobalBuffer((__gm__ half *)p.value);
        kCacheGm_.SetGlobalBuffer((__gm__ half *)p.keyCache);
        vCacheGm_.SetGlobalBuffer((__gm__ half *)p.valueCache);
        slotGm_.SetGlobalBuffer((__gm__ int32_t *)p.slotMapping);
        kNormGm_.SetGlobalBuffer((__gm__ half *)p.keyNorms);
        vNormGm_.SetGlobalBuffer((__gm__ half *)p.valueNorms);
        signGm_.SetGlobalBuffer((__gm__ float *)p.signs);
        cbGm_.SetGlobalBuffer((__gm__ float *)p.centroids);

        pipe->InitBuffer(inQue_, kBufferNum, d * sizeof(half));
        pipe->InitBuffer(vecBuf_, d * sizeof(float));
        pipe->InitBuffer(kNrmStgBuf_, kNrmBlk * sizeof(half));   // atomic norm staging
        pipe->InitBuffer(vNrmStgBuf_, kNrmBlk * sizeof(half));
        pipe->InitBuffer(tmpBuf_, d * sizeof(float));
        pipe->InitBuffer(codeBuf_, d * sizeof(int32_t));
        pipe->InitBuffer(byteBuf_, TQTraits<BITS>::PackedBytes(d));
        pipe->InitBuffer(signBuf_, d * sizeof(float));
        pipe->InitBuffer(cbBuf_, (2 * (1 << BITS)) * sizeof(float));
        // planar-pack scratch. These were DECLARED and USED but never
        // InitBuffer'd, so Get<float>() handed back aliasing tensors: pack then
        // computed c[128]*32 instead of c[0] + c[128]*16, and c[0] was lost.
        pipe->InitBuffer(pkHBuf_, (d / 2) * sizeof(half));
        pipe->InitBuffer(pkF0Buf_, (d / 2) * sizeof(float));
        pipe->InitBuffer(pkF1Buf_, (d / 2) * sizeof(float));

        signs_ = signBuf_.Get<float>();
        DataCopy(signs_, signGm_, d);
        if (t_->codebookMode == TQ_CB_LUT) {
            cb_ = cbBuf_.Get<float>();
            DataCopy(cb_, cbGm_, 2 * (1 << BITS));
        }
        invSqrtLen_ = t_->invSqrtHeadDim;   // host-computed: AscendC Sqrt is tensor-only
        sqrtLen_ = t_->sqrtHeadDim;
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t begin = core * t_->tokensPerCore;
        const uint32_t end = (begin + t_->tokensPerCore < t_->numTokens) ? (begin + t_->tokensPerCore)
                                                                        : t_->numTokens;
        for (uint32_t tok = begin; tok < end; ++tok) {
            const int32_t slot = slotGm_.GetValue(tok);
            if (slot < 0) {
                continue;  // padded / masked token
            }
            /*
             * NORM WRITE -- one whole 32B block per slot.
             *
             * The plane is [num_slots, kNzC0] halves, so slot `slot` OWNS the
             * block at slot*kNzC0 and only the first numKvHeads lanes carry
             * data. That buys three things at once:
             *
             *  1. No cache-line race. The old layout packed numKvHeads halves =
             *     8 BYTES per slot, so 8 slots shared a 64B line and a scalar
             *     2-byte store was a read-modify-write that lost concurrent
             *     updates (measured: 184/512 norms wrong under permuted slots).
             *     A 32B aligned store is safe -- the packed payload is exactly
             *     32B per (block,c1,slot) and was bit-identical under
             *     contiguous, gapped and permuted slot layouts.
             *  2. Idempotent rewrite, so SLOT REUSE overwrites instead of
             *     accumulating. The previous fix used atomic-add, which is only
             *     correct while the plane is freshly zeroed on every call.
             *  3. The plane can therefore be a PERSISTENT caller-owned tensor.
             *     It has to be: serving writes one token per decode step, and
             *     allocating inside the op discarded all history (measured
             *     cosine 0.139670 -- see talk/tq_norm_persistence.py).
             */
            LocalTensor<half> kStage = kNrmStgBuf_.Get<half>();
            LocalTensor<half> vStage = vNrmStgBuf_.Get<half>();
            Duplicate(kStage, static_cast<half>(0.0f), kNrmBlk);
            Duplicate(vStage, static_cast<half>(0.0f), kNrmBlk);
            PipeBarrier<PIPE_V>();
            const uint64_t nBase = static_cast<uint64_t>(slot) * kNrmBlk;
            for (uint32_t h = 0; h < t_->numKvHeads; ++h) {
                const float kn = HandleVector(keyGm_, kCacheGm_, tok, h, slot);
                const float vn = HandleVector(valGm_, vCacheGm_, tok, h, slot);
                kStage.SetValue(h, static_cast<half>(kn));
                vStage.SetValue(h, static_cast<half>(vn));
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
            DataCopy(kNormGm_[nBase], kStage, kNrmBlk);
            DataCopy(vNormGm_[nBase], vStage, kNrmBlk);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID4);   // next token re-Duplicates the stage
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
        }
    }

private:
    /*
     * One (token, head) vector: load -> norm -> normalise -> Pi -> quantize ->
     * pack -> scatter. The norm is stored separately, so the packed plane keeps
     * its exact NZ tile alignment.
     */
    __aicore__ inline float HandleVector(const GlobalTensor<half> &src, const GlobalTensor<half> &cache,
                                         uint32_t tok, uint32_t h, int32_t slot)
    {
        const uint32_t d = t_->headDim;
        LocalTensor<half> in = inQue_.AllocTensor<half>();
        DataCopy(in, src[(tok * t_->numKvHeads + h) * d], d);
        inQue_.EnQue(in);
        in = inQue_.DeQue<half>();

        LocalTensor<float> v = vecBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        Cast(v, in, RoundMode::CAST_NONE, d);
        PipeBarrier<PIPE_V>();
        inQue_.FreeTensor(in);

        // ||x||, then normalise. Pi is orthogonal so the order (normalise then
        // rotate, or rotate then normalise) is free; we normalise first so the
        // quantizer always sees a unit vector.
        Mul(tmp, v, v, d);
        PipeBarrier<PIPE_V>();
        ReduceSum(tmp, tmp, tmp, d);
        PipeBarrier<PIPE_V>();
        Sqrt(tmp, tmp, 1);                  // tensor form; no scalar overload exists
        PipeBarrier<PIPE_V>();
        float nrm = tmp.GetValue(0);
        if (nrm < 1e-12f) {
            nrm = 1e-12f;
        }
        Muls(v, v, 1.0f / nrm, d);
        PipeBarrier<PIPE_V>();

        RotatePi(v, signs_, tmp, d, invSqrtLen_);

        LocalTensor<int32_t> codes = codeBuf_.Get<int32_t>();
        QuantizeVec<BITS>(v, codes, cb_[1 << BITS], tmp, d, sqrtLen_, t_->codebookMode);

        LocalTensor<uint8_t> bytes = byteBuf_.Get<uint8_t>();
        if constexpr (TQPlanar<BITS>::kSupported) {
            LocalTensor<half> pkH = pkHBuf_.Get<half>();
            LocalTensor<float> pkF0 = pkF0Buf_.Get<float>();
            LocalTensor<float> pkF1 = pkF1Buf_.Get<float>();
            PackCodesVec<BITS>(codes, bytes, pkH, pkF0, pkF1, d);   // vector path
            /*
             * V -> MTE3. The vector pack writes `bytes` with Cast on the V pipe
             * and ScatterNz then reads it with an MTE3 DataCopy; V->MTE3 is NOT
             * implicitly drained on 310P. The old scalar PackCodes wrote via
             * SetValue on the S pipe and S->MTE3 IS drained, so this edge did
             * not exist before the vectorisation.
             * Symptom without it: token 0 packs correctly (nothing in flight)
             * while later tokens land corrupted, and b=2 (4 planes, longer pack)
             * is hit far harder than b=4 (2 planes).
             */
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        } else {
            PackCodes<BITS>(codes, bytes, d);                 // scalar fallback (BITS=3)
        }
        PipeBarrier<PIPE_V>();

        ScatterNz(cache, bytes, h, slot);
        return nrm;   // caller batches the norms into one atomic 32B block
    }

    /*
     * Scatter packed halves into (numBlocks, C1, blockSize, 16) fp16.
     * Feature index f = h*packedHalves + j maps to [c1 = f/16, off, c0 = f%16],
     * so a head's run is packedHalves/16 contiguous 16-element groups, each
     * landing at a fixed stride. 16 halves = 32B: aligned, no DataCopyPad
     * (which 310P does not support).
     */
    __aicore__ inline void ScatterNz(const GlobalTensor<half> &cache, const LocalTensor<uint8_t> &bytes,
                                     uint32_t h, int32_t slot)
    {
        const uint32_t blockSize = t_->blockSize;
        const uint32_t blk = static_cast<uint32_t>(slot) / blockSize;
        const uint32_t off = static_cast<uint32_t>(slot) % blockSize;
        const uint32_t ph = t_->packedHalves;
        const uint32_t groups = ph / kNzC0;
        LocalTensor<half> asHalf = bytes.ReinterpretCast<half>();
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t c1 = (h * ph) / kNzC0 + g;
            const uint64_t dst = ((static_cast<uint64_t>(blk) * t_->c1 + c1) * blockSize + off) * kNzC0;
            DataCopy(cache[dst], asHalf[g * kNzC0], kNzC0);
        }
    }

    const TurboquantReshapeAndCacheV310TilingData *t_;
    GlobalTensor<half> keyGm_, valGm_, kCacheGm_, vCacheGm_, kNormGm_, vNormGm_;
    GlobalTensor<int32_t> slotGm_;
    GlobalTensor<float> signGm_, cbGm_;
    TQue<QuePosition::VECIN, kBufferNum> inQue_;
    TBuf<TPosition::VECCALC> kNrmStgBuf_, vNrmStgBuf_,
        vecBuf_, tmpBuf_, codeBuf_, byteBuf_, signBuf_, cbBuf_,
        pkHBuf_, pkF0Buf_, pkF1Buf_;
    LocalTensor<float> signs_, cb_;
    float invSqrtLen_{1.0f};
    float sqrtLen_{1.0f};
};

}  // namespace TurboQuantWrite

#endif  // TURBOQUANT_RESHAPE_AND_CACHE_V310_H
