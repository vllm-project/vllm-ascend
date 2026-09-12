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
 * \file quant_batch_matmul_v3_x_int8_init.h
 * \brief INT8 kernel Init() implementation: UB layout and tiling-data unpack.
 * \author Feodor Pisnitchenko
 *
 * Included by quant_batch_matmul_v3_x_int8.h AFTER the class
 * declaration; do not include directly.
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_INT8_INIT_H
#define QUANT_BATCH_MATMUL_V3_X_INT8_INIT_H

namespace QBM {

template <int SCALE_TYPE>
__aicore__ inline void QBMInt8Compute<SCALE_TYPE>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y, GM_ADDR workspace,
    const QBMParams* params, TPipe* tPipe)
    {
        pipe_ = tPipe;
        const QBMParams* p = params;

        // Core grid + problem shape from tiling.
        coreIdx_ = GetBlockIdx();
        coreNum_ = p->usedCoreNum;
        MCoreNum_ = (p->MCoreNum > 0) ? p->MCoreNum : 1;
        NCoreNum_ = (p->NCoreNum > 0) ? p->NCoreNum : 1;
        batch_ = p->batch;
        M_ = p->M;
        K_ = p->K;
        N_ = p->N;

        // Inner tile + K-pass count.
        baseM_ = p->maxOneTurnToken;
        baseN_ = p->maxWeightColOneTurn;
        hasPertoken_ = p->hasPertoken;
        Q_ = p->quantGroupNum;
        if (Q_ == 0) Q_ = 1;
        // kTail / kPadded are read early because Kq (and downstream
        // kPasses / kSize) routes through K_padded for Q==1 so the last
        // K-pass is K0-aligned. For Q>1 the host already requires
        // K % (Q*K0) == 0, so kTail is always 0 and kPadded == K.
        kTail_ = p->kTail;
        kPadded_ = (kTail_ != 0) ? (K_ + (32U - kTail_)) : K_;
        Kq_ = (Q_ == 1) ? kPadded_ : (K_ / Q_);
        baseK_ = p->baseK;
        if (baseK_ == 0 || baseK_ > Kq_) baseK_ = Kq_;
        kPasses_ = (Kq_ + baseK_ - 1) / baseK_;

        // wL1 chunk = wChunkKPasses_ * baseK * baseN bytes (one MTE2
        // refills it, the kp loop reads kp slices via MTE1).
        wChunkKPasses_ = p->wChunkKPasses;
        if (wChunkKPasses_ == 0) wChunkKPasses_ = 1;
        if (wChunkKPasses_ > kPasses_) wChunkKPasses_ = kPasses_;
        wL1ChunkBytes_ = wChunkKPasses_ * baseK_ * baseN_;

        // Scale path selection. scaleCoalesce only applies on the u64
        // path (Q==1 loads [N] once per mTile, Q>1 loads Q*baseN per cb).
        scaleType_ = p->scaleType;
        isPerTensor_ = p->isPerTensor;
        scaleCoalesce_ = p->scaleCoalesce;
        if (scaleCoalesce_ && isPerTensor_) {
            scaleCoalesce_ = 0;
        }
        // fp32 coalesce supports Q == 1 only; Q > 1 fp32 falls back to
        // per-channel staging (the staging buffer would need to hold
        // Q*N floats, which exceeds the UB budget).
        if (scaleCoalesce_ && scaleType_ != SCALE_UINT64 && Q_ > 1) {
            scaleCoalesce_ = 0;
        }

        // L2 super-tile geometry. mTileCntL2_ == 1 && nTileCntL2_ == 1
        // means no split -- the kernel falls back to the existing
        // cooperative-batch / contiguous-vmIdx outer iteration for the
        // entire problem (one super-tile covering all base blocks).
        mTileCntL2_ = (p->mTileCntL2 > 0) ? p->mTileCntL2 : 1U;
        nTileCntL2_ = (p->nTileCntL2 > 0) ? p->nTileCntL2 : 1U;
        mTileBlock_ = (p->mTileBlock > 0) ? p->mTileBlock : 1U;
        nTileBlock_ = (p->nTileBlock > 0) ? p->nTileBlock : 1U;

        // L0C ping-pong: alternate L0C halves per cb iteration so Mmad of
        // cb_{i+1} on half b can run in parallel with VDEQ16 of cb_i on
        // half a. dbL0c_ == 0 falls back to single-buffer (half always 0).
        dbL0c_ = p->dbL0c;

        // Pertoken coalesce: when 1, the kernel fires ONE MTE2 covering
        // all B*M floats at Process entry; per-mTile use indexes into the
        // resident buffer at offset (batchIdx*M + mStart). Falls back to
        // per-mTile MTE2 when 0.
        pertokenCoalesce_ = p->pertokenCoalesce;

        // Bias: when 1, bias[N] int32 is loaded once at Process entry into
        // ubBias_, then BroadCastVecToMM (UB->L0C) broadcasts the per-cb
        // slice into L0C before each cb's first Mmad. Mmad runs with
        // init_c=false so X@W accumulates onto bias. VDEQ16 then
        // dequantizes (bias + X@W) * scale in one fp16 truncation.
        hasBias_ = p->hasBias;

        // Inner-tile Phase D: rows per mu slice in the cb-loop tail (0 = disabled).
        ubCalcM_ = p->ubCalcM;

        // CacheMode for SetL2CacheHint: NORMAL by default, PERSISTENT for
        // shared-weight workloads where the weight fits L2, DISABLE for
        // single-work-item-per-core (no reuse) workloads.
        l2HintMode_ = p->l2HintMode;
        CacheMode l2Mode = CacheMode::CACHE_MODE_NORMAL;
        if (l2HintMode_ == 0U) {
            l2Mode = CacheMode::CACHE_MODE_DISABLE;
        } else if (l2HintMode_ == 4U) {
            l2Mode = CacheMode::CACHE_MODE_PERSISTENT;
        }

        // Phase X chunking. tail is "paired" when tail > chunk and
        // splits into ping + pong.
        phaseXChunk_ = p->phaseXChunk;
        pingPong_ = (phaseXChunk_ < K_);
        phaseXFullPairs_ = p->phaseXFullPairs;
        phaseXTail_ = p->phaseXTail;
        tailPingSize_ = (phaseXTail_ > 0) ? Min(phaseXChunk_, phaseXTail_) : 0;
        tailPongSize_ = (phaseXTail_ > phaseXChunk_) ? (phaseXTail_ - phaseXChunk_) : 0;

        // Precomputed strides + counts. Blocked-ZN fires when
        // N1 > BLOCKED_ZN_NG (otherwise srcStride overflows uint16).
        N1_ = N_ / CUBE_N0;
        weightBlockN1_ = (N1_ > qbmv3::BLOCKED_ZN_NG) ? qbmv3::BLOCKED_ZN_NG : 0;
        // K1q_ is K1_total per Q-channel: since Kq_ above is already
        // K_padded for Q==1 (and K-aligned Kq for Q>1), this naturally
        // includes the partial last K0-group when kTail > 0.
        K1q_ = Kq_ / CUBE_K0_INT8;
        baseNG_ = baseN_ / CUBE_N0;
        baseKFracs_ = baseK_ / CUBE_K0_INT8;
        MK_ = static_cast<uint64_t>(M_) * K_;     // x1 GM stride: K_actual
        MN_ = static_cast<uint64_t>(M_) * N_;

        // GM batch / channel strides. Weight on device is COMPACT (= K * N
        // bytes per batch, no K-tail padding). Per-batch stride = K * N;
        // per-channel stride = (K / Q) * N. Q > 1 requires K % (Q*K0) == 0
        // so K_actual = K_padded in that branch; Q == 1 may have a partial
        // last K-fractal, handled separately in PrefetchWeightToL1.
        x2FracTotal_ = static_cast<uint64_t>(K_) * N_;
        wGmChStride_ = static_cast<uint64_t>(K_ / Q_) * N_;

        // xL1 / L0A use K_padded for per-row strides so the zero-padded
        // K-tail fits. Phase X MTE2 source stride uses K_actual (x1 GM
        // layout); only the L1/L0 destination side is padded.
        l0aTurnStride_ = CUBE_M0 * kPadded_;
        l0aKpStride_ = baseKFracs_ * CUBE_K0_INT8 * CUBE_M0;
        l1TurnStride_ = CUBE_M0 * kPadded_;

        x1Gm_.SetGlobalBuffer((__gm__ int8_t*)x1);
        x1Gm_.SetL2CacheHint(l2Mode);
        x2Gm_.SetGlobalBuffer((__gm__ int8_t*)x2);
        x2Gm_.SetL2CacheHint(l2Mode);
        yGm_.SetGlobalBuffer((__gm__ half*)y);
        yGm_.SetL2CacheHint(l2Mode);
        if (scaleType_ == SCALE_UINT64) {
            scaleGm_.SetGlobalBuffer((__gm__ uint64_t*)scale);
            scaleGm_.SetL2CacheHint(l2Mode);
        } else {
            scaleGmFp32_.SetGlobalBuffer((__gm__ float*)scale);
            scaleGmFp32_.SetL2CacheHint(l2Mode);
        }
        // Treat hasPertoken=true with a null pointer as no-pertoken so later
        // reads do not dereference an unset GlobalBuffer.
        if (hasPertoken_ && pertokenScale == nullptr) {
            hasPertoken_ = 0;
        }
        if (hasPertoken_) {
            pertokenGm_.SetGlobalBuffer((__gm__ float*)pertokenScale);
            pertokenGm_.SetL2CacheHint(l2Mode);
        }
        // Treat hasBias=true with a null pointer as no-bias so later
        // reads do not dereference an unset GlobalBuffer.
        if (hasBias_ && bias == nullptr) {
            hasBias_ = 0;
        }
        if (hasBias_) {
            biasGm_.SetGlobalBuffer((__gm__ int32_t*)bias);
            biasGm_.SetL2CacheHint(l2Mode);
        }

        // L0 buffers
        pipe_->InitBuffer(l0aBuf_, qbmv3::L0A_TOTAL_BYTES);
        pipe_->InitBuffer(l0bBuf_, qbmv3::L0B_TOTAL_BYTES);
        pipe_->InitBuffer(l0cBuf_, qbmv3::L0C_TOTAL_BYTES);

        // L1: xL1 (ZZ fractal), wL1 ping/pong (ZN fractal).
        // wL1 chunk = wChunkKPasses_ * baseK * baseN bytes; one MTE2
        // refills it, then the inner kp loop reads kp slices via MTE1.
        uint32_t mAligned = AlignUp(baseM_, CUBE_M0);
        // xL1 holds activation in ZZ format with K_padded bytes per
        // M-fractal so the partial last K0-group fits.
        uint32_t xL1Size = mAligned * kPadded_;
        pipe_->InitBuffer(xL1Buf_, xL1Size);
        pipe_->InitBuffer(wL1Ping_, wL1ChunkBytes_);
        pipe_->InitBuffer(wL1Pong_, wL1ChunkBytes_);

        // UB layout (UB_BANK_PAD between regions to avoid bank conflicts):
        //   Persistent: outPing/Pong [M_a*Nb*2 fp16], pertokenLT [M_a*4 fp32],
        //     scaleU64Ping/Pong [Nb*8 u64], scatterIdxLT [Nb*4 i32].
        //   Workspace at offWs (Phase X and Phase D aliased here):
        //     Phase X: xPing [M_a*chunk] + PAD + xPong [M_a*chunk].
        //     Phase D: dequantScratch [M_a*Nb*2 fp16] + PAD + scaleFp32Staging [Nb*4].
        //     wsSize = max(2*chunk+PAD, singleFoSize+PAD+baseN*4).
        //   The Phase X MTE3 scatter drains before Phase D writes the
        //   aliased range, so the alias is safe.
        constexpr uint64_t UB_BANK_PAD = qbmv3::UB_BANK_PAD;

        uint64_t singleFoSize = static_cast<uint64_t>(mAligned) * baseN_ * sizeof(half);

        uint64_t offFoPong = singleFoSize + UB_BANK_PAD;
        uint64_t offPt = offFoPong + singleFoSize + UB_BANK_PAD;
        // Pertoken buffer sized for either the per-mTile slice (mAligned
        // floats) or the full B*M coalesced load when pertokenCoalesce_.
        uint32_t ptSize = pertokenCoalesce_
            ? AlignUp(batch_ * M_ * 4U, 32U)
            : AlignUp(mAligned * 4U, 32U);

        uint64_t offScalePing = offPt + ptSize + UB_BANK_PAD;
        // Coalesced scale buffer size depends on the path:
        //   Q == 1 coalesce -> full [N] uint64 scale, loaded once per mTile.
        //   Q  > 1 coalesce -> Q * baseN uint64s, loaded once per cb.
        //   non-coalesce   -> baseN * 8 ping (+ pong when per-channel).
        uint32_t scaleBaseSize = baseN_ * 8;
        uint32_t scalePingSize;
        if (scaleCoalesce_) {
            scalePingSize = (Q_ == 1) ? (N_ * 8) : (Q_ * baseN_ * 8);
        } else {
            scalePingSize = scaleBaseSize;
        }
        // Pertensor scale is filled once per mTile into scaleU64PingLT_ and
        // reused across every (cb, ch); the pong slot is pure dead space then.
        // Coalesce is per-cb and also uses only the ping slot.
        bool allocScalePong = !isPerTensor_ && !scaleCoalesce_;
        uint32_t scalePongSize = allocScalePong ? scaleBaseSize : 0;
        uint64_t offScalePong = offScalePing + scalePingSize + UB_BANK_PAD;
        uint64_t offAfterScale = allocScalePong
            ? (offScalePong + scalePongSize + UB_BANK_PAD)
            : (offScalePing + scalePingSize + UB_BANK_PAD);

        // scatterIdxLT_ is only needed on the per-channel float32 path
        // (used by Scatter to pack float32 into uint64 slots).
        bool needScatterIdx = (!isPerTensor_) && (scaleType_ != SCALE_UINT64);
        uint32_t scatterIdxSize = baseN_ * sizeof(uint32_t);
        uint64_t offScatterIdx = offAfterScale;

        uint64_t offAfterScatter = needScatterIdx
            ? (offScatterIdx + scatterIdxSize + UB_BANK_PAD)
            : offAfterScale;

        // ubBias_ persistent region: loaded once at Process entry via a
        // single MTE2 of N*4 bytes, then sliced per-cb during the
        // BroadCastVecToMM (UB->L0C) write that precedes each cb's first
        // Mmad.
        uint32_t biasSize = hasBias_
            ? AlignUp(static_cast<uint32_t>(N_ * sizeof(int32_t)), 32U)
            : 0U;
        uint64_t offBias = offAfterScatter;
        uint64_t offWs = hasBias_
            ? (offBias + biasSize + UB_BANK_PAD)
            : offAfterScatter;
        // Phase X size: xPingLT_ + PAD + xPongLT_ (both sized chunkBufSize).
        // Phase X: ping/pong buffers each hold mAligned * phaseXChunk_ bytes.
        // When pingPong_ == false, only xPingLT_ is used (no pong+pad needed).
        uint32_t chunkBufSize = AlignUp(
            static_cast<uint32_t>(mAligned * phaseXChunk_), 32U);
        uint64_t phaseXSize = pingPong_
            ? (static_cast<uint64_t>(chunkBufSize) * 2 + UB_BANK_PAD)
            : static_cast<uint64_t>(chunkBufSize);
        // Phase D: dequantScratch [singleFo] + scaleFp32Staging [singleFo
        // upper bound] + ptBroadcast [singleFo] when Q>1 + hasPertoken.
        // ptBroadcast holds pre-broadcast pertoken values for the whole
        // mTile so the cb loop can issue ONE Mul instead of mSize Muls +
        // mSize GetValue scalar reads per cb.
        const bool needPtBroadcast = hasPertoken_ && Q_ > 1;
        // Slot 1 (scaleFp32StagingLT_) is baseN*4 or N*4 bytes, not
        // singleFo. The tight phaseDSize formula here must match the
        // ptBroadcast offset computed below.
        const uint32_t fp32StagingElems =
            (scaleCoalesce_ && scaleType_ != SCALE_UINT64) ? N_ : baseN_;
        const uint64_t fp32StagingBytes =
            static_cast<uint64_t>(fp32StagingElems) * sizeof(float);
        const uint64_t fp32StagingAligned =
            (fp32StagingBytes + 31) & ~31ULL;
        uint64_t phaseDSize;
        if (needPtBroadcast) {
            phaseDSize = singleFoSize + UB_BANK_PAD +
                         fp32StagingAligned + UB_BANK_PAD +
                         singleFoSize + UB_BANK_PAD;
        } else {
            phaseDSize = singleFoSize + UB_BANK_PAD +
                         fp32StagingAligned + UB_BANK_PAD;
        }
        uint64_t wsSize = (phaseXSize > phaseDSize) ? phaseXSize : phaseDSize;

        pipe_->InitBuffer(ubBuf_, static_cast<uint32_t>(offWs + wsSize));

        // Persistent UB tensors
        outPing_ = ubBuf_.GetWithOffset<half>(
            mAligned * baseN_, 0);
        outPong_ = ubBuf_.GetWithOffset<half>(
            mAligned * baseN_, static_cast<int64_t>(offFoPong));
        pertokenLT_ = ubBuf_.GetWithOffset<float>(
            ptSize / sizeof(float), static_cast<int64_t>(offPt));
        // Ping slot size mirrors scalePingSize: baseN (non-coalesce),
        // Q * baseN (Q>1 coalesce), or N (Q=1 coalesce).
        uint32_t scalePingElems;
        if (scaleCoalesce_) {
            scalePingElems = (Q_ == 1) ? N_ : (Q_ * baseN_);
        } else {
            scalePingElems = baseN_;
        }
        scaleU64PingLT_ = ubBuf_.GetWithOffset<uint64_t>(
            scalePingElems, static_cast<int64_t>(offScalePing));
        // Pong slot: only allocated on the per-channel uint64 + fp32 path
        // with ping/pong (not coalesced, not per-tensor).
        if (allocScalePong) {
            scaleU64PongLT_ = ubBuf_.GetWithOffset<uint64_t>(
                baseN_, static_cast<int64_t>(offScalePong));
        }
        if (needScatterIdx) {
            scatterIdxLT_ = ubBuf_.GetWithOffset<uint32_t>(
                baseN_, static_cast<int64_t>(offScatterIdx));
        }
        if (hasBias_) {
            ubBias_ = ubBuf_.GetWithOffset<int32_t>(
                N_, static_cast<int64_t>(offBias));
        }

        // Phase X workspace tensors (freed after Phase X completes)
        xPingLT_ = ubBuf_.GetWithOffset<int8_t>(
            chunkBufSize, static_cast<int64_t>(offWs));
        if (pingPong_) {
            xPongLT_ = ubBuf_.GetWithOffset<int8_t>(
                chunkBufSize, static_cast<int64_t>(offWs + chunkBufSize + UB_BANK_PAD));
        }

        // Phase D workspace tensors (aliased with Phase X workspace)
        // dequantScratchLT_: VDEQ16 ch>0 accumulate + pertoken pre-broadcast (dual use, no conflict)
        dequantScratchLT_ = ubBuf_.GetWithOffset<half>(
            mAligned * baseN_, static_cast<int64_t>(offWs));
        // scaleFp32StagingLT_: float32 scale staging (separate from
        // dequantScratchLT_). When the fp32 coalesce path is active the
        // buffer must hold the full N floats so a single MTE2 covers the
        // whole row of the scale; otherwise it just holds one cb (baseN).
        uint64_t offScaleStaging = offWs + singleFoSize + UB_BANK_PAD;
        // fp32StagingElems was declared at the top of InitBuffer for
        // the tight Phase D budget; the staging buffer lives here.
        scaleFp32StagingLT_ = ubBuf_.GetWithOffset<float>(
            fp32StagingElems, static_cast<int64_t>(offScaleStaging));
        // ptBroadcastLT_: Q>1 + hasPertoken pre-broadcast destination,
        // sized like dequantScratchLT_ so the cb pertoken Mul can take
        // it as the RHS of a single element-wise Mul over
        // (mAligned * baseN) elements. Placed tightly after the actual
        // scaleFp32StagingLT_ extent so high-baseM cells fit UB. The
        // host's Phase D budget must mirror this offset.
        if (needPtBroadcast) {
            const uint64_t fp32StagingBytes =
                static_cast<uint64_t>(fp32StagingElems) * sizeof(float);
            const uint64_t fp32StagingAligned =
                (fp32StagingBytes + 31) & ~31ULL;  // 32-byte align
            uint64_t offPtBroadcast =
                offWs + singleFoSize + UB_BANK_PAD +
                fp32StagingAligned + UB_BANK_PAD;
            ptBroadcastLT_ = ubBuf_.GetWithOffset<half>(
                mAligned * baseN_, static_cast<int64_t>(offPtBroadcast));
        }
    }

}  // namespace QBM

#endif  // QUANT_BATCH_MATMUL_V3_X_INT8_INIT_H
