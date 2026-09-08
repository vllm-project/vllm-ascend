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
 * \file quant_batch_matmul_v3_x_int8_process.h
 * \brief INT8 kernel Process() implementation: Phase X / Phase D pipeline loop.
 * \author Feodor Pisnitchenko
 *
 * Included by quant_batch_matmul_v3_x_int8.h AFTER the class
 * declaration; do not include directly.
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_INT8_PROCESS_H
#define QUANT_BATCH_MATMUL_V3_X_INT8_PROCESS_H

namespace QBM {

template <int SCALE_TYPE>
__aicore__ inline void QBMInt8Compute<SCALE_TYPE>::Process()
    {
        if (coreIdx_ >= coreNum_) return;

        uint32_t totalMTiles = (M_ + baseM_ - 1) / baseM_;
        uint32_t totalColBatch = (N_ + baseN_ - 1) / baseN_;
        uint32_t virtualM = batch_ * totalMTiles;

        // Grid ownership. coopBatch (mTiles_in_supertile >= MCoreNum): all
        // cores share batchIdx, split mTiles -> x2 reads coalesce. Else
        // fall back to contiguous vmIdx split (no idle cores; no L2 reuse).
        // Per-core ranges are recomputed per L2 super-tile (see Process
        // outer loops); only the M/N axis core indices are constant here.
        uint32_t mCoreIdx = coreIdx_ / NCoreNum_;
        uint32_t nCoreIdx = coreIdx_ % NCoreNum_;

        // === PRIME: seed pipelined event counters ===
        // L0 ping/pong: ID0 = L0 half 0, ID1 = L0 half 1. Both start free.
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);     // CopyOut buffer initially free
        SetFlag<HardEvent::V_M>(EVENT_ID0);        // L0C half 0 initially free
        if (dbL0c_) {
            SetFlag<HardEvent::V_M>(EVENT_ID1);    // L0C half 1 initially free
        }
        if (!isPerTensor_ && scaleType_ != SCALE_UINT64) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2); // fp32 staging initially free
        }
        if (scaleCoalesce_) {
            // Coalesced scale buffer initially free so the first cb's MTE2
            // doesn't wait. Paired with per-cb SetFlag at loop tail and the
            // drain WaitFlag at kernel end.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        }
        // wL1Ping_ and wL1Pong_ initially free
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);  // wL1Ping_
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);  // wL1Pong_
        if (hasPertoken_ && !pertokenCoalesce_) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID3); // pertokenLT_ initially free (per-mTile path)
        }
        // dequantScratchLT_ aliases xPingLT_/xPongLT_ at offWs. V's writes
        // to dequantScratchLT_ in the cb-loop must drain before Phase X
        // MTE2 of the next vmIdx overwrites that UB region. Init set so
        // the first vmIdx's Phase X does not block.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);

        // Invariant across all mTile and colBatch iterations
        LocalTensor<int8_t> xL1 = xL1Buf_.Get<int8_t>();
        LocalTensor<int8_t> l0a = l0aBuf_.Get<int8_t>();
        LocalTensor<int8_t> l0b = l0bBuf_.Get<int8_t>();
        LocalTensor<int32_t> l0c = l0cBuf_.Get<int32_t>();
        half scatterOne = static_cast<half>(1.0f);
        uint32_t scatterDstRS = baseNG_;

        // Per-channel fp32 scale: build idx[i] = 8*i so Scatter places each
        // fp32 word in the even int32 slot of a u64 (odd slot = VDEQ16 marker).
        if (scaleType_ != SCALE_UINT64 && !isPerTensor_) {
            LocalTensor<int32_t> idxI32 = scatterIdxLT_.ReinterpretCast<int32_t>();
            // idxI32[i] = i  (linear ramp, Muls below scales by 8)
            CreateVecIndex<int32_t>(idxI32, static_cast<int32_t>(0), baseN_);
            // idxI32[i] = i * 8  (byte offset of even int32 slot i)
            Muls<int32_t>(idxI32, idxI32, static_cast<int32_t>(8), baseN_);
            PipeBarrier<PIPE_V>();
        }

        // Pertoken coalesce: single MTE2 covers all B*M floats once;
        // every mTile reads from the persistent buffer at offset
        // (batchIdx*M + mStart). Skips per-mTile MTE2 setup entirely.
        if (hasPertoken_ && pertokenCoalesce_) {
            // pertokenLT[0..B*M-1] <- pt[0..B*M-1]  (one MTE2, full pertoken vector)
            DataCopy<float>(pertokenLT_, pertokenGm_,
                            AlignUp(batch_ * M_, 8U));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        }

        // Bias coalesce: single MTE2 covers all N int32 bias values.
        // Per-cb broadcast (BroadCastVecToMM) slices into ubBias_[nStart:]
        // without re-touching GM. Drains the MTE2_V on EVENT_ID2 before
        // the cb loop's first BroadCastVecToMM consumes the buffer.
        if (hasBias_) {
            // ubBias[0..N-1] <- bias[0..N-1]  (one MTE2, full bias vector)
            DataCopy<int32_t>(ubBias_, biasGm_, AlignUp(N_, 8U));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        }

        // L2 super-tile outer loops. mTileCntL2_ == nTileCntL2_ == 1 means
        // the entire problem fits one super-tile; the bounds collapse to
        // the original iteration with totalMTiles / totalColBatch / virtualM.
        // Otherwise each (mTSuperIdx, nTSuperIdx) covers an L2-resident
        // sub-rectangle of the full work space.
        for (uint32_t mTSuperIdx = 0; mTSuperIdx < mTileCntL2_; mTSuperIdx++) {
        for (uint32_t nTSuperIdx = 0; nTSuperIdx < nTileCntL2_; nTSuperIdx++) {
        uint32_t mTileLo = mTSuperIdx * mTileBlock_;
        uint32_t mTileHi = (mTileLo + mTileBlock_ < totalMTiles)
                              ? (mTileLo + mTileBlock_) : totalMTiles;
        uint32_t mTileLenSuper = mTileHi - mTileLo;
        uint32_t cbLo = nTSuperIdx * nTileBlock_;
        uint32_t cbHi = (cbLo + nTileBlock_ < totalColBatch)
                           ? (cbLo + nTileBlock_) : totalColBatch;
        uint32_t cbLenSuper = cbHi - cbLo;

        // Per-core ranges within this super-tile. coopBatch decision uses
        // the super-tile's own mTile count (not the full problem) so a
        // narrow super-tile does not get spuriously labelled coop.
        bool coopBatch = (mTileLenSuper >= MCoreNum_);
        uint32_t myMTileStart = 0, myMTileEnd = 0;
        uint32_t myVmStart = 0, myVmEnd = 0;
        if (coopBatch) {
            uint32_t mTileBase = mTileLenSuper / MCoreNum_;
            uint32_t mTileRem  = mTileLenSuper % MCoreNum_;
            myMTileStart = mTileLo +
                mCoreIdx * mTileBase + Min(mCoreIdx, mTileRem);
            myMTileEnd   = myMTileStart + mTileBase +
                           (mCoreIdx < mTileRem ? 1 : 0);
        } else {
            uint32_t virtualMSuper = batch_ * mTileLenSuper;
            uint32_t mBase = virtualMSuper / MCoreNum_;
            uint32_t mRem  = virtualMSuper % MCoreNum_;
            myVmStart = mCoreIdx * mBase + Min(mCoreIdx, mRem);
            myVmEnd   = myVmStart + mBase + (mCoreIdx < mRem ? 1 : 0);
        }
        uint32_t nBase = cbLenSuper / NCoreNum_;
        uint32_t nRem  = cbLenSuper % NCoreNum_;
        uint32_t cbStart = cbLo + nCoreIdx * nBase + Min(nCoreIdx, nRem);
        uint32_t cbEnd   = cbStart + nBase + (nCoreIdx < nRem ? 1 : 0);

        // If this core has no cb work in the current super-tile (tail
        // super-tile with cbLenSuper < NCoreNum can leave higher
        // nCoreIdx with empty range), skip the per-mTile setup that
        // would otherwise leak MTE2_MTE1 / V_MTE2 tokens.
        if (cbStart >= cbEnd) {
            continue;
        }

        // Coop-batch: outer = batch (all cores on same batchIdx -> coalesce).
        // Fallback: contiguous vmIdx split (each core walks unique vm work).
        uint32_t cbRange = cbEnd - cbStart;
        uint32_t batchOuter = coopBatch ? batch_ : 1;
        uint32_t batchOuterStart = 0;
        for (uint32_t batchOuterIdx = batchOuterStart;
             batchOuterIdx < batchOuter; batchOuterIdx++) {
        uint32_t innerStart = coopBatch ? myMTileStart : myVmStart;
        uint32_t innerEnd   = coopBatch ? myMTileEnd   : myVmEnd;
        for (uint32_t innerIdx = innerStart; innerIdx < innerEnd; innerIdx++) {
            uint32_t batchIdx = coopBatch ? batchOuterIdx
                                          : (innerIdx / mTileLenSuper);
            uint32_t mTileIdx = coopBatch ? innerIdx
                                          : (mTileLo + (innerIdx % mTileLenSuper));
            uint32_t mStart = mTileIdx * baseM_;
            uint32_t mSize = Min(baseM_, M_ - mStart);
            uint32_t mAligned = AlignUp(mSize, CUBE_M0);

            // GM base offsets: X1[b,0,0], X2[b,0,0,0,0], Y[b,0,0]
            uint64_t x1BatchOff = batchIdx * MK_;
            uint64_t x2BatchOff = batchIdx * x2FracTotal_;
            uint64_t yBatchOff = batchIdx * MN_;


            // pertokenLT[r] <- pt[b*M + mStart + r]   (once per mTile in
            // non-coalesce mode). When pertokenCoalesce_, the buffer was
            // populated once by a single MTE2 at Process entry and we
            // skip per-mTile MTE2 entirely; downstream uses index into
            // pertokenLT_ at offset (batchIdx*M + mStart).
            uint32_t ptOff = batchIdx * M_ + mStart;
            if (hasPertoken_ && !pertokenCoalesce_) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
                DataCopy<float>(pertokenLT_, pertokenGm_[ptOff],
                                AlignUp(mSize, 8U));
                SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            }

            // Phase X: xL1 <- X1[b, mStart..+mSize, 0..K] in ZZ fractal.
            // K split into phaseXChunk_-byte chunks, double-buffered
            // through xPing / xPong (MTE2 || V pad || MTE3 scatter).
            // tailPingSize_ + tailPongSize_ cover leftover K bytes.
            uint64_t xGmBase = x1BatchOff +
                static_cast<uint64_t>(mStart) * K_;

            // Gate Phase X MTE2 on the previous vmIdx's V-pipe writes to
            // dequantScratchLT_, which aliases xPingLT_ at offWs.
            // PipeBarrier<PIPE_V> drains V locally but does not order V
            // before the next vmIdx's MTE2.
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);

            // Prime EVENT_ID7 before first scatter (no prior MTE3 to set it)
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            // Prime MTE3_MTE2(0): xPingLT_/xPongLT_ initially safe for MTE2
            // Only needed when ping/pong is active (fullPairs > 0 or tail uses both buffers)
            if (pingPong_) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }

            // Full ping/pong pairs over the K axis.
            for (uint32_t fp = 0; fp < phaseXFullPairs_; fp++) {
                uint32_t kOff = fp * 2 * phaseXChunk_;

                // Gate MTE2 on prev pong scatter (xPing/xPong overwrite).
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                // xPing[r, k] <- X1[b, mStart+r, kOff..kOff+chunk)
                X1ChunkMte2(xPingLT_, xGmBase + kOff, mSize, phaseXChunk_);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
                if (mSize < mAligned) {
                    // xPing[mSize..mAligned, *] <- 0  (pad trailing rows)
                    uint32_t padStart = mSize * phaseXChunk_;
                    uint32_t padCount = (mAligned - mSize) * phaseXChunk_;
                    Duplicate<int16_t>(xPingLT_.ReinterpretCast<int16_t>()[padStart / 2],
                                       static_cast<int16_t>(0), padCount / 2);
                    PipeBarrier<PIPE_V>();
                }
                // xPong[r, k] <- X1[b, mStart+r, kOff+chunk..+2*chunk)
                X1ChunkMte2(xPongLT_, xGmBase + kOff + phaseXChunk_, mSize, phaseXChunk_);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                // xL1[mAligned, kOff..kOff+chunk) <- xPing[r, k] reshuffled into ZZ fractal
                ScatterZZToL1(mAligned, kOff, phaseXChunk_, xPingLT_);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
                if (mSize < mAligned) {
                    uint32_t padStart = mSize * phaseXChunk_;
                    uint32_t padCount = (mAligned - mSize) * phaseXChunk_;
                    // xPong[mSize..mAligned, *] <- 0  (pad trailing rows)
                    Duplicate<int16_t>(xPongLT_.ReinterpretCast<int16_t>()[padStart / 2],
                                       static_cast<int16_t>(0), padCount / 2);
                    PipeBarrier<PIPE_V>();
                }
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                // xL1[mAligned, kOff+chunk..kOff+2*chunk) <- xPong[r, k] reshuffled into ZZ fractal
                ScatterZZToL1(mAligned, kOff + phaseXChunk_, phaseXChunk_, xPongLT_);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // UB buffers safe for next MTE2 load
            }

            // --- Tail: 0, 1, or 2 sub-chunks (tailPingSize_ + tailPongSize_) ---
            if (tailPingSize_ > 0) {
                uint32_t tailKOff = phaseXFullPairs_ * 2 * phaseXChunk_;
                // Per-row UB stride and scatter chunk are aligned up to K0
                // so the trailing partial K0-group (when kTail > 0) is a
                // full 32-byte block in UB and L1. Bytes past kTail in
                // each row come from x1 GM overread (next-row data or
                // past-end) but get multiplied by zero-padded weight in
                // Mmad, so they do not affect the result.
                const uint32_t tailPingScatter =
                    ((tailPingSize_ + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) * UB_BLOCK_SIZE;
                const uint32_t tailPongScatter =
                    ((tailPongSize_ + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) * UB_BLOCK_SIZE;

                // Gate: MTE2 waits for last pong scatter before overwriting xPingLT_.
                // Only needed when ping/pong is active -- single-buffer path has
                // no prior MTE3_MTE2 token to consume (would deadlock).
                if (pingPong_) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                // xPing[r, k] <- X1[b, mStart+r, tailKOff..tailKOff+tailPing)
                X1ChunkMte2(xPingLT_, xGmBase + tailKOff, mSize, tailPingSize_);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID5);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
                if (mSize < mAligned) {
                    uint32_t padStart = mSize * tailPingScatter;
                    uint32_t padCount = (mAligned - mSize) * tailPingScatter;
                    // xPing[mSize..mAligned, *] <- 0  (pad trailing rows of tail ping)
                    Duplicate<int16_t>(xPingLT_.ReinterpretCast<int16_t>()[padStart / 2],
                                       static_cast<int16_t>(0), padCount / 2);
                    PipeBarrier<PIPE_V>();
                }
                // V-pipe zero the K-tail bytes per row in xPingLT_ (when
                // tail chunk's last K0-group is the K-actual boundary).
                // Mmad sums L0A K-tail (zero) * L0B K-tail (garbage from
                // overread / unloaded partial K-fractal) = 0.
                if (kTail_ > 0 && tailPongSize_ == 0) {
                    LocalTensor<int16_t> ub16 = xPingLT_.ReinterpretCast<int16_t>();
                    uint32_t rowStrideElems = tailPingScatter / sizeof(int16_t);
                    uint32_t tailOffElems   = tailPingSize_  / sizeof(int16_t);
                    uint32_t tailLenElems   =
                        (tailPingScatter - tailPingSize_) / sizeof(int16_t);
                    for (uint32_t r = 0; r < mAligned; r++) {
                        // xPing[r, tailOff..tailOff+tailLen) <- 0  (zero K-tail of last K0-group, per row)
                        Duplicate<int16_t>(
                            ub16[r * rowStrideElems + tailOffElems],
                            static_cast<int16_t>(0),
                            tailLenElems);
                    }
                    PipeBarrier<PIPE_V>();
                }
                if (tailPongSize_ > 0) {
                    // xPong[r, k] <- X1[b, mStart+r, tailKOff+tailPing..tailKOff+tail)
                    X1ChunkMte2(xPongLT_, xGmBase + tailKOff + tailPingSize_,
                                mSize, tailPongSize_);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
                }
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                // xL1[mAligned, tailKOff..tailKOff+tailPingScatter) <- xPing[r, k] reshuffled into ZZ fractal
                ScatterZZToL1(mAligned, tailKOff, tailPingScatter, xPingLT_);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

                if (tailPongSize_ > 0) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
                    if (mSize < mAligned) {
                        uint32_t padStart = mSize * tailPongScatter;
                        uint32_t padCount = (mAligned - mSize) * tailPongScatter;
                        // xPong[mSize..mAligned, *] <- 0  (pad trailing rows of tail pong)
                        Duplicate<int16_t>(xPongLT_.ReinterpretCast<int16_t>()[padStart / 2],
                                           static_cast<int16_t>(0), padCount / 2);
                        PipeBarrier<PIPE_V>();
                    }
                    // K-tail lives in the LAST chunk (= pong when present).
                    if (kTail_ > 0) {
                        LocalTensor<int16_t> ub16 = xPongLT_.ReinterpretCast<int16_t>();
                        uint32_t rowStrideElems = tailPongScatter / sizeof(int16_t);
                        uint32_t tailOffElems   = tailPongSize_  / sizeof(int16_t);
                        uint32_t tailLenElems   =
                            (tailPongScatter - tailPongSize_) / sizeof(int16_t);
                        for (uint32_t r = 0; r < mAligned; r++) {
                            // xPong[r, tailOff..tailOff+tailLen) <- 0  (zero K-tail of last K0-group, per row)
                            Duplicate<int16_t>(
                                ub16[r * rowStrideElems + tailOffElems],
                                static_cast<int16_t>(0),
                                tailLenElems);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    // xL1[mAligned, tailKOff+tailPingScatter..+tailPong) <- xPong[r, k] reshuffled into ZZ fractal
                    ScatterZZToL1(mAligned, tailKOff + tailPingScatter, tailPongScatter, xPongLT_);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
                }
            }

            // MTE3 done writing xL1 -> MTE1 can read xL1 safely
            SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);

            if (pingPong_ && tailPingSize_ == 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // drain last pong scatter token
            }

            // Pertoken pre-broadcast lets the cb loop run a single
            // element-wise Mul against the [mAligned, baseN] scratch
            // instead of mSize per-row Muls + mSize S-pipe GetValues
            // per cb. Q==1 broadcasts into dequantScratchLT_ (free for
            // Q==1 because VDEQ16 writes finalOutputLT_ directly when
            // ch==0). Q>1 broadcasts into a dedicated ptBroadcastLT_
            // slot -- dequantScratchLT_ is needed across every ch>0
            // for the VDEQ16 accumulate.
            if (hasPertoken_) {
                if (!pertokenCoalesce_) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
                }
                // Coalesce mode: index globally; per-mTile mode: index local.
                uint32_t ptBase = pertokenCoalesce_ ? ptOff : 0u;
                LocalTensor<half> ptDst =
                    (Q_ == 1) ? dequantScratchLT_ : ptBroadcastLT_;
                for (uint32_t m = 0; m < mSize; m++) {
                    half pts = static_cast<half>(pertokenLT_.GetValue(ptBase + m));
                    // ptDst[m, 0..baseN-1] <- pt[ptBase+m]  (broadcast scalar across baseN cols)
                    Duplicate<half>(ptDst[m * baseN_], pts, baseN_);
                }
                PipeBarrier<PIPE_V>();
            }

            // mTile-dependent colBatch-loop invariants
            uint32_t mTurns = mAligned / CUBE_M0;
            uint32_t scatterMRows = mSize;
            uint32_t srcNZStride = mAligned * CUBE_N0;
            // Mul covers full ND row stride (baseN_); junk past nSize is harmless
            // because CopyOut uses ngCount/srcStride to skip it.
            uint32_t pertokenMulCount = mSize * baseN_;

            // Inner-tile Phase D (full-buffer case): when ubCalcM_ > 0
            // and Q_ == 1 and mAligned > ubCalcM_, wrap the cb-loop
            // tail (VDEQ16 + scatter + pertoken Mul + CopyOut) in an
            // mu inner loop draining L0C in M slices. Q_ > 1 not
            // supported (cross-ch accumulator carry needs full
            // finalOutputLT_).
            const bool useInnerTilePhaseD =
                (Q_ == 1) && (ubCalcM_ > 0) && (ubCalcM_ < mAligned);
            const uint32_t mUbLoops = useInnerTilePhaseD
                ? ((mAligned + ubCalcM_ - 1) / ubCalcM_)
                : 1u;

            // MTE2 prime: load the first chunk of (ch=0, cb=cbStart) into ping.
            uint32_t cbFirstNSize = Min(baseN_, N_ - cbStart * baseN_);
            uint32_t primeKSize = Min(wChunkKPasses_ * baseK_, Kq_);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);  // wL1Ping_ free
            PrefetchWeightToL1(0, cbStart, /*kpBaseK=*/0, primeKSize,
                               x2BatchOff, cbFirstNSize, /*usePing=*/true);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
            // MTE1 must wait for Phase X MTE3 scatter to finish before reading xL1
            WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);

            // Running weight-buffer index: 0 = ping, 1 = pong. Flipped on
            // every MTE2. Starts at 0 -- the prime above filled ping.
            uint32_t wBufIdx = 0;

            if (isPerTensor_) {
                // Per-tensor scale: build VDEQ16 u64 buffer once per mTile.
                //   scaleU64Ping[2j  ] = scaleBits             (low32 fp32 bits)
                //   scaleU64Ping[2j+1] = 0x4000                (high32 marker)
                int32_t scaleBits;
                if (scaleType_ == SCALE_UINT64) {
                    scaleBits = static_cast<int32_t>(scaleGm_.GetValue(0));
                } else {
                    float val = scaleGmFp32_.GetValue(0);
                    scaleBits = *reinterpret_cast<int32_t*>(&val);
                }
                LocalTensor<int32_t> scaleI32 =
                    scaleU64PingLT_.ReinterpretCast<int32_t>();
                // Fill all i32 slots with marker, then overwrite even
                // slots with scaleBits using an even-mask Duplicate.
                Duplicate<int32_t>(scaleI32,
                                   static_cast<int32_t>(qbmv3::VDEQ16_MARKER), baseN_ * 2);
                uint32_t reps = (baseN_ * 2 + 7) / 8;
                uint64_t evenMask[2] = {0x55ULL, 0x0ULL};
                Duplicate<int32_t>(scaleI32, scaleBits, evenMask,
                                   static_cast<uint8_t>(reps), 1, 1);
                PipeBarrier<PIPE_V>();
            }

            // ColBatch loop:
            //   Y_tile[m, n] = pt[m] * \sum_q fp16(scale[q,n] * (X1 @ X2)[m,n])
            // Per-cb iteration. NCoreNum=1 makes all cores walk the same
            // cbStart..cbEnd so concurrent reads of cb_k coalesce in L2.
            for (uint32_t iter = 0; iter < cbRange; iter++) {
                uint32_t cb = cbStart + iter;
                uint32_t nxIter = iter + 1;
                bool hasNextIter = (nxIter < cbRange);
                uint32_t nextIterCb = hasNextIter ? (cbStart + nxIter) : cb;
                bool isFirstIter = (iter == 0);

                // L0C ping-pong: when dbL0c_, alternate halves per cb so
                // Mmad of cb_{i+1} on half b runs in parallel with VDEQ16
                // of cb_i on half a. dbL0c_ == 0 collapses to half 0 only.
                uint32_t l0cHalf = dbL0c_ ? (iter & 1u) : 0u;
                uint32_t l0cOff = l0cHalf *
                    static_cast<uint32_t>(baseM_) * baseN_;
                event_t l0cVMEvent = l0cHalf ? EVENT_ID1 : EVENT_ID0;
                event_t l0cMVEvent = l0cHalf ? EVENT_ID4 : EVENT_ID3;
                uint32_t nStart = cb * baseN_;
                uint32_t nSize = Min(baseN_, N_ - nStart);
                uint32_t ngCount = nSize / CUBE_N0;

                WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);       // prev CopyOut done
                finalOutputLT_ = (cb & 1) ? outPong_ : outPing_;

                // Coalesced scale MTE2 cadence:
                //   Q>1: once per cb (loads Q scales for this cb).
                //   Q==1: once per mTile (loads full [N], reused).
                // Gated by V_MTE2(ID7) back-edge so prior VDEQ16 finished.
                if (scaleCoalesce_) {
                    bool isFirstCbOfMTile = isFirstIter;
                    bool fireScaleMTE2 = (Q_ > 1) || isFirstCbOfMTile;
                    if (fireScaleMTE2) {
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
                        if (Q_ > 1) {
                            // scaleU64Ping[q, n] <- scale[q, nStart..+nSize]
                            //   q in [0, Q), strided in 32-B blocks (4 u64).
                            DataCopyParams scP = {
                                static_cast<uint16_t>(Q_),
                                static_cast<uint16_t>(nSize / 4),
                                static_cast<uint16_t>((N_ - nSize) / 4),
                                static_cast<uint16_t>((baseN_ - nSize) / 4)};
                            DataCopy<uint64_t>(scaleU64PingLT_,
                                                scaleGm_[nStart], scP);
                        } else if (scaleType_ == SCALE_UINT64) {
                            // scaleU64Ping[n] <- scale[0, n]   (full [N])
                            DataCopy<uint64_t>(scaleU64PingLT_,
                                                scaleGm_, N_);
                        } else {
                            // fp32 Q==1 coalesce: single MTE2 of full
                            // [N] fp32s into staging once per mTile; per
                            // cb the V-pipe encode reads from
                            // staging[nStart..nStart+nSize].
                            // scaleFp32Staging[0..N-1] <- scale[0..N-1]  (Q==1 fp32 coalesce, once per mTile)
                            DataCopy<float>(scaleFp32StagingLT_,
                                             scaleGmFp32_, N_);
                        }
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    }
                }

                // VDEQ16 params (blockMode/deqScale invariant, deqTensorAddr set per-ch)
                DataCopyEnhancedParams paramL0c2Ub;
                paramL0c2Ub.blockMode = BlockMode::BLOCK_MODE_MATRIX;
                paramL0c2Ub.deqScale = DeqScale::VDEQ16;

                for (uint32_t ch = 0; ch < Q_; ch++) {
                    // Coalesce mode: single ping buffer holds all Q scales
                    // back-to-back, one per channel at offset ch * baseN.
                    // Non-coalesce: classic ping/pong by (iter*Q+ch) parity.
                    LocalTensor<uint64_t> curScaleLT;
                    if (scaleCoalesce_) {
                        // Q>1: channel's scale starts at ch * baseN.
                        // Q==1: single scale tensor [N]; this cb's slice at
                        //       nStart (the whole [N] was loaded per-mTile).
                        uint32_t off = (Q_ == 1) ? nStart : (ch * baseN_);
                        curScaleLT = scaleU64PingLT_[off];
                    } else {
                        bool scalePing = (((iter * Q_) + ch) & 1) == 0;
                        curScaleLT = (isPerTensor_ || scalePing)
                            ? scaleU64PingLT_ : scaleU64PongLT_;
                    }
                    paramL0c2Ub.deqTensorAddr =
                        reinterpret_cast<uint64_t>(curScaleLT.GetPhyAddr());

                    // K-pass loop:  L0C = sum_{kp} L0A[m, kp] * L0B[kp, n].
                    // Outer iterates wChunkKPasses_ consecutive kPasses, each
                    // chunk = one ping/pong wL1 slot (one MTE2). Inner reads
                    // baseK*baseN slices of that chunk via MTE1.
                    uint32_t xChBase = ch * CUBE_M0 * Kq_;
                    uint32_t kpBaseKBytes = 0;
                    uint32_t kpFracOff = 0;
                    uint32_t chunkStartKp = 0;
                    while (chunkStartKp < kPasses_) {
                        uint32_t chunkKps = Min(wChunkKPasses_,
                                                kPasses_ - chunkStartKp);

                        LocalTensor<int8_t> curWL1 = (wBufIdx == 0)
                            ? wL1Ping_.Get<int8_t>() : wL1Pong_.Get<int8_t>();
                        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);  // chunk ready

                        // Pre-loop next-chunk MTE2 (overlaps with Cube/MTE1).
                        // Order: next chunk -> next ch -> next cb. Only when
                        // wChunkKPasses_ > 1; post-loop emit covers the rest.
                        bool nxPrefetchFired = false;
                        if (wChunkKPasses_ > 1) {
                            uint32_t nxChunkStartKpPre = chunkStartKp + chunkKps;
                            bool hasNextPre = false;
                            uint32_t nxChPre = 0, nxCbPre = 0;
                            uint32_t nxKpBaseKPre = 0, nxNSizePre = 0;
                            uint32_t nxChunkKpsPre = 0;
                            if (nxChunkStartKpPre < kPasses_) {
                                hasNextPre = true;
                                nxChPre = ch; nxCbPre = cb;
                                nxKpBaseKPre = nxChunkStartKpPre * baseK_;
                                nxNSizePre = nSize;
                                nxChunkKpsPre = Min(wChunkKPasses_,
                                    kPasses_ - nxChunkStartKpPre);
                            } else if (ch + 1 < Q_) {
                                hasNextPre = true;
                                nxChPre = ch + 1; nxCbPre = cb;
                                nxKpBaseKPre = 0; nxNSizePre = nSize;
                                nxChunkKpsPre = Min(wChunkKPasses_, kPasses_);
                            } else if (hasNextIter) {
                                hasNextPre = true;
                                nxChPre = 0; nxCbPre = nextIterCb;
                                nxKpBaseKPre = 0;
                                nxNSizePre = Min(baseN_,
                                    N_ - nextIterCb * baseN_);
                                nxChunkKpsPre = Min(wChunkKPasses_, kPasses_);
                            }
                            if (hasNextPre) {
                                uint32_t nxBuf = 1u - wBufIdx;
                                uint32_t nxKSize = nxChunkKpsPre * baseK_;
                                if (nxKpBaseKPre + nxKSize > Kq_) {
                                    nxKSize = Kq_ - nxKpBaseKPre;
                                }
                                WaitFlag<HardEvent::MTE1_MTE2>(
                                    (nxBuf == 0) ? EVENT_ID0 : EVENT_ID1);
                                PrefetchWeightToL1(nxChPre, nxCbPre,
                                    nxKpBaseKPre, nxKSize,
                                    x2BatchOff, nxNSizePre,
                                    /*usePing=*/(nxBuf == 0));
                                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                                nxPrefetchFired = true;
                            }
                        }

                        for (uint32_t kpInChunk = 0; kpInChunk < chunkKps; kpInChunk++) {
                            uint32_t kp = chunkStartKp + kpInChunk;
                            uint32_t kSize = Min(baseK_, Kq_ - kpBaseKBytes);
                            uint32_t kf = kSize / CUBE_K0_INT8;
                            uint32_t totalFracs = kf * ngCount;
                            if (totalFracs == 0) totalFracs = 1;

                            // Scale MTE2 once per (cb, ch), gated on kp==0.
                            // Coalesce mode pulls the uint64 scale load out
                            // of this kp loop (one MTE2 per cb before ch
                            // loop), so skip the per-ch load there.
                            if (kp == 0 && !scaleCoalesce_) {
                                if (!isPerTensor_ && scaleType_ != SCALE_UINT64) {
                                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);  // fp32 staging free
                                }
                                if (!isPerTensor_) {
                                    uint64_t scaleOff = (Q_ == 1) ? nStart
                                        : static_cast<uint64_t>(ch) * N_ + nStart;
                                    if (scaleType_ == SCALE_UINT64) {
                                        // curScale[0..nSize-1] <- scale[scaleOff..scaleOff+nSize)  (per-cb u64 path)
                                        DataCopy<uint64_t>(curScaleLT, scaleGm_[scaleOff], nSize);
                                    } else {
                                        // scaleFp32Staging[0..nSize-1] <- scale[scaleOff..scaleOff+nSize)  (per-cb fp32 path)
                                        DataCopy<float>(scaleFp32StagingLT_, scaleGmFp32_[scaleOff], nSize);
                                        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
                                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
                                    }
                                }
                            }

                            // L0 ping/pong: half 0 for even kp, half 1 for odd kp.
                            uint32_t l0Idx = kp & 1u;
                            uint32_t l0aOff = l0Idx ? qbmv3::L0A_HALF_BYTES : 0u;
                            uint32_t l0bOff = l0Idx ? qbmv3::L0B_HALF_BYTES : 0u;
                            event_t l0Event = l0Idx ? EVENT_ID1 : EVENT_ID0;

                            // L0A[m, k_kp] <- xL1[m, k_kp]   (ZZ -> L0A, per mTurn)
                            // L0B[k_kp, n]  <- wL1[k_kp, n]   (ZN -> L0B, packed
                            //                                  kf*ngCount fractals)
                            WaitFlag<HardEvent::M_MTE1>(l0Event);
                            for (uint32_t t = 0; t < mTurns; t++)
                                LoadData(l0a[l0aOff + t * CUBE_M0 * kSize],
                                         xL1[xChBase + t * l0aTurnStride_ + kpFracOff],
                                         {0, static_cast<uint8_t>(kf), 1, 0, 0, false, 0});
                            // wL1 row stride = nSize (not baseN_; partial last cb).
                            uint32_t wL1Off = kpInChunk * baseK_ * nSize;
                            LoadData(l0b[l0bOff], curWL1[wL1Off],
                                     {0, static_cast<uint8_t>(totalFracs), 1, 0, 0, false, 0});
                            SetFlag<HardEvent::MTE1_M>(l0Event);

                            // L0C[m,n] = (init ? bias[n] : L0C[m,n])
                            //          + \sum_{k} L0A[m,k] * L0B[k,n]
                            // With hasBias_, V-pipe BroadCastVecToMM loads
                            // bias[n] across M into L0C, then Mmad accumulates
                            // X@W with init_c=false. Otherwise init_c=(kp==0)
                            // zeros L0C before Mmad as usual.
                            WaitFlag<HardEvent::MTE1_M>(l0Event);
                            if (kp == 0) {
                                WaitFlag<HardEvent::V_M>(l0cVMEvent);  // L0C half free
                                if (hasBias_ && ch == 0) {
                                    // L0C NZ layout [N1, M1, M0, N0] int32.
                                    // Each BroadCastVecToMM call writes one
                                    // (M0=16, N0=16) tile from a 16-elt UB
                                    // slice (blockCount=blockLen=1, no gap).
                                    const uint32_t m1Cnt = mAligned / CUBE_M0;
                                    const uint32_t n1Cnt = nSize / CUBE_N0;
                                    for (uint32_t i = 0; i < n1Cnt; i++) {
                                        for (uint32_t j = 0; j < m1Cnt; j++) {
                                            // L0C[n1=i, m1=j, m0=0..M0, n0=0..N0]
                                            //   <- bias[nStart + i*N0 + n0]
                                            //   (broadcast 16 fp32 bias values across M0 rows of one NZ tile)
                                            BroadCastVecToMM(
                                                l0c[l0cOff +
                                                    i * mAligned * CUBE_N0 +
                                                    j * CUBE_M0 * CUBE_N0],
                                                ubBias_[nStart + i * CUBE_N0],
                                                1, 1, 0, 0);
                                        }
                                    }
                                    SetFlag<HardEvent::V_M>(EVENT_ID2);
                                    WaitFlag<HardEvent::V_M>(EVENT_ID2);
                                }
                            }
                            PipeBarrier<PIPE_M>();  // serialize back-to-back Mmads on L0C
                            bool mmadInit = (kp == 0) && !(hasBias_ && ch == 0);
                            Mmad(l0c[l0cOff], l0a[l0aOff], l0b[l0bOff],
                                 {static_cast<uint16_t>(mAligned),
                                  static_cast<uint16_t>(nSize),
                                  static_cast<uint16_t>(kSize),
                                  0, false, mmadInit});
                            SetFlag<HardEvent::M_MTE1>(l0Event);

                            kpBaseKBytes += baseK_;
                            kpFracOff += l0aKpStride_;
                        }

                        // End of chunk -- wL1 buffer fully consumed, MTE2 may refill.
                        SetFlag<HardEvent::MTE1_MTE2>((wBufIdx == 0) ? EVENT_ID0 : EVENT_ID1);

                        // Post-loop prefetch is the fallback for
                        // wChunkKPasses_ == 1 (no overlap window) and tail
                        // cases where the pre-loop emit did not fire.
                        if (!nxPrefetchFired) {
                            // Order: next chunk in this ch, then first chunk
                            // of next ch, then first chunk of next cb.
                            uint32_t nxChunkStartKp = chunkStartKp + chunkKps;
                            bool hasNext = false;
                            uint32_t nxCh = 0, nxCb = 0;
                            uint32_t nxKpBaseK = 0, nxNSize = 0;
                            uint32_t nxChunkKps = 0;
                            if (nxChunkStartKp < kPasses_) {
                                hasNext = true;
                                nxCh = ch; nxCb = cb;
                                nxKpBaseK = nxChunkStartKp * baseK_;
                                nxNSize = nSize;
                                nxChunkKps = Min(wChunkKPasses_, kPasses_ - nxChunkStartKp);
                            } else if (ch + 1 < Q_) {
                                hasNext = true;
                                nxCh = ch + 1; nxCb = cb;
                                nxKpBaseK = 0; nxNSize = nSize;
                                nxChunkKps = Min(wChunkKPasses_, kPasses_);
                            } else if (hasNextIter) {
                                hasNext = true;
                                nxCh = 0; nxCb = nextIterCb;
                                nxKpBaseK = 0;
                                nxNSize = Min(baseN_, N_ - nextIterCb * baseN_);
                                nxChunkKps = Min(wChunkKPasses_, kPasses_);
                            }
                            if (hasNext) {
                                uint32_t nxBuf = 1u - wBufIdx;
                                uint32_t nxKSize = nxChunkKps * baseK_;
                                if (nxKpBaseK + nxKSize > Kq_) {
                                    nxKSize = Kq_ - nxKpBaseK;
                                }
                                WaitFlag<HardEvent::MTE1_MTE2>(
                                    (nxBuf == 0) ? EVENT_ID0 : EVENT_ID1);
                                PrefetchWeightToL1(nxCh, nxCb, nxKpBaseK, nxKSize,
                                                   x2BatchOff, nxNSize,
                                                   /*usePing=*/(nxBuf == 0));
                                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                            }
                        }

                        wBufIdx = 1u - wBufIdx;
                        chunkStartKp += chunkKps;
                    }

                    // --- Float32->uint64 encode || Cube Mmad in parallel ---
                    if (!isPerTensor_ && scaleType_ != SCALE_UINT64) {
                        // When fp32 coalesce fires (Q==1 only), the
                        // staging buffer holds [N] fp32s; this cb's slice
                        // starts at offset nStart.
                        const uint32_t fp32SrcOff =
                            (scaleCoalesce_) ? nStart : 0u;
                        LocalTensor<int32_t> scaleI32 = curScaleLT.ReinterpretCast<int32_t>();
                        LocalTensor<uint32_t> srcU32 = scaleFp32StagingLT_[fp32SrcOff].ReinterpretCast<uint32_t>();
                        // scaleI32[0..2*nSize-1] <- VDEQ16_MARKER  (Scatter below overwrites even slots with fp32 bits)
                        Duplicate<int32_t>(scaleI32, static_cast<int32_t>(qbmv3::VDEQ16_MARKER), nSize * 2);
                        LocalTensor<uint32_t> scaleU32 = curScaleLT.ReinterpretCast<uint32_t>();
                        // scaleU32[idx[i]/4] <- srcU32[i]  for i in [0, nSize)
                        //   idx = even-int32-slot byte offsets (i*8 from CreateVecIndex+Muls);
                        //   packs each fp32 word into the lo32 of u64 slot i.
                        Scatter<uint32_t>(scaleU32, srcU32, scatterIdxLT_, 0, nSize);
                        PipeBarrier<PIPE_V>();
                    }

                    // Scale is ready -- signal V before VDEQ16. Skipped for
                    // pertensor (no per-ch scale MTE2) and for coalesce mode
                    // (the SetFlag happens once per cb, outside the ch loop).
                    if (!isPerTensor_ && !scaleCoalesce_) {
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    }

                    // VDEQ16: vdeqDst[m, n] = fp16(L0C[m, n] * scale[ch, n])
                    SetFlag<HardEvent::M_V>(l0cMVEvent);
                    WaitFlag<HardEvent::M_V>(l0cMVEvent);
                    // Consume the MTE2_V(ID4) "scale ready" flag exactly once
                    // per scale MTE2 fire (mode-dependent cadence).
                    if (!isPerTensor_ && !scaleCoalesce_) {
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    } else if (scaleCoalesce_ && Q_ > 1 && ch == 0) {
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    } else if (scaleCoalesce_ && Q_ == 1 && isFirstIter) {
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
                    }

                    // Inner-tile Phase D: defer VDEQ16 to per-mu
                    // partial drains after the ch loop. Q == 1 means a
                    // single ch=0 iteration so the deferred VDEQ16
                    // reads a fully populated L0C.
                    if (!useInnerTilePhaseD) {
                        LocalTensor<half> vdeqDst = (ch == 0) ? finalOutputLT_ : dequantScratchLT_;
                        DataCopy(vdeqDst, l0c[l0cOff],
                                 {static_cast<uint16_t>(ngCount), static_cast<uint16_t>(mTurns), 0, 0},
                                 paramL0c2Ub);
                        PipeBarrier<PIPE_V>();
                    }
                    // fp32 staging-free signal. Only fires when
                    // !scaleCoalesce_; in the coalesce path the cross-cb
                    // V->MTE2 dependency is already covered by V_MTE2(ID7),
                    // so firing here would leak tokens with no matching wait.
                    if (!isPerTensor_ && scaleType_ != SCALE_UINT64 && !scaleCoalesce_) {
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);  // fp32 staging free
                    }

                    // Channel accumulate (ch > 0):
                    //   accumDst[m, n] = finalOutput[m, n] + dequantScratch[m, n]
                    // Non-inplace: accumDst != finalOutput by (cb+ch) parity
                    // because V-pipe Add is undefined when src and dst alias.
                    if (ch > 0) {
                        LocalTensor<half> accumDst = ((cb + ch) & 1)
                            ? outPong_ : outPing_;
                        Add(accumDst, finalOutputLT_, dequantScratchLT_,
                            mAligned * nSize);
                        finalOutputLT_ = accumDst;
                        PipeBarrier<PIPE_V>();
                    }

                    // Inner-tile Phase D: the per-mu loop below fires
                    // V_M(l0cVMEvent) once after the last partial
                    // VDEQ16 (L0C only free after all mu slices
                    // drained).
                    if (!useInnerTilePhaseD) {
                        SetFlag<HardEvent::V_M>(l0cVMEvent);
                    }
                }

                if (useInnerTilePhaseD) {
                    // Per-mu partial VDEQ16 + scatter + pertoken Mul +
                    // CopyOut. Buffers stay mAligned*baseN sized
                    // (full-buffer case).
                    LocalTensor<half> copyOutBuf = ((cb + Q_) & 1)
                        ? outPong_ : outPing_;
                    const uint64_t gmOffBase = yBatchOff +
                        static_cast<uint64_t>(mStart) * N_ + nStart;
                    constexpr uint32_t MAX_REP = 255;

                    for (uint32_t mu = 0; mu < mUbLoops; mu++) {
                        const uint32_t mStart_u = mu * ubCalcM_;
                        const uint32_t mSize_u = (mu + 1 < mUbLoops)
                            ? ubCalcM_ : (mAligned - mStart_u);
                        const uint32_t mTurns_u = mSize_u / CUBE_M0;
                        const uint32_t scatterRows_u =
                            (mStart_u >= scatterMRows)
                                ? 0u
                                : Min(scatterMRows - mStart_u, mSize_u);

                        // Partial VDEQ16: Q==1 so vdeqDst = finalOutputLT_.
                        // Case B keeps full mAligned*baseN buffer; the mu
                        // slice lands at row mStart_u of each N1 sub-block
                        // (dstStride skips the unused rows of this N1).
                        // finalOutput[mStart_u..mStart_u+mSize_u, :]
                        //   <- fp16(L0C[mStart_u..mStart_u+mSize_u, :] * scale[0, :])
                        DataCopy(finalOutputLT_[mStart_u * CUBE_N0],
                                 l0c[l0cOff + mStart_u * CUBE_N0],
                                 {static_cast<uint16_t>(ngCount),
                                  static_cast<uint16_t>(mTurns_u),
                                  static_cast<uint16_t>(mTurns - mTurns_u),
                                  static_cast<uint16_t>(mAligned - mSize_u)},
                                 paramL0c2Ub);
                        PipeBarrier<PIPE_V>();

                        if (scatterRows_u == 0) {
                            continue;
                        }

                        // Partial NZ -> ND scatter (Muls scalar=1).
                        uint32_t rowsLeft = scatterRows_u;
                        uint32_t mOff = mStart_u;
                        while (rowsLeft > 0) {
                            uint32_t chunk = (rowsLeft > MAX_REP) ? MAX_REP : rowsLeft;
                            for (uint32_t ng = 0; ng < ngCount; ng++)
                                // copyOutBuf[m, ng*N0..(ng+1)*N0] <- finalOutput[ng, m, 0..N0] * 1.0
                                //   for m in [mOff, mOff+chunk), one Muls call per ng
                                Muls(copyOutBuf[mOff * baseN_ + ng * CUBE_N0],
                                     finalOutputLT_[ng * srcNZStride + mOff * CUBE_N0],
                                     scatterOne, CUBE_N0, static_cast<uint8_t>(chunk),
                                     {1, 1, static_cast<uint8_t>(scatterDstRS), 1});
                            rowsLeft -= chunk;
                            mOff += chunk;
                        }

                        // Partial pertoken Mul (Q==1 path uses
                        // dequantScratchLT_ which holds the pre-broadcast
                        // pertoken[m] tiled across baseN columns).  Writes
                        // IN-PLACE to copyOutBuf so the next mu's partial
                        // VDEQ16 -- which writes finalOutputLT_ in NZ
                        // stride overlapping this mu's ND-slice region -- does not
                        // race with this mu's still-in-flight MTE3 CopyOut.
                        if (hasPertoken_) {
                            PipeBarrier<PIPE_V>();
                            // copyOutBuf[m, n] *= ptBroadcast[m, n]
                            //   for m in [mStart_u, mStart_u+scatterRows_u), n in [0, baseN)
                            Mul(copyOutBuf[mStart_u * baseN_],
                                copyOutBuf[mStart_u * baseN_],
                                dequantScratchLT_[mStart_u * baseN_],
                                scatterRows_u * baseN_);
                        }
                        PipeBarrier<PIPE_V>();

                        // Partial CopyOut.  Always reads copyOutBuf (the
                        // pertoken Mul above wrote in-place into it).  The
                        // cb-level MTE3_V back-edge (below the if/else
                        // block) fires once after the LAST mu MTE3 op
                        // since MTE3 is in-order.
                        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                        DataCopyParams outP_mu = {
                            static_cast<uint16_t>(scatterRows_u),
                            static_cast<uint16_t>(ngCount),
                            static_cast<uint16_t>(baseNG_ - ngCount),
                            static_cast<uint16_t>(N1_ - ngCount)};
                        // yGm[b, mStart+mStart_u..+scatterRows_u, nStart..nStart+nSize]
                        //   <- copyOutBuf[mStart_u..mStart_u+scatterRows_u, 0..nSize]  (MTE3 UB->GM)
                        DataCopy(yGm_[gmOffBase +
                                      static_cast<uint64_t>(mStart_u) * N_],
                                 copyOutBuf[mStart_u * baseN_], outP_mu);
                    }
                    SetFlag<HardEvent::V_M>(l0cVMEvent);  // L0C free
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
                } else {

                // NZ -> ND scatter (Muls scalar=1, multi-row repeat):
                //   copyOutBuf[m, n] = finalOutput[ng = n/N0, m, n%N0]
                // copyOutBuf picked from the opposite ping/pong slot via
                // (cb+Q)&1 to ensure no alias with finalOutput.
                LocalTensor<half> copyOutBuf = ((cb + Q_) & 1)
                    ? outPong_ : outPing_;
                // Chunk in 255-row blocks -- Muls repeatTimes is uint8_t and
                // wraps to 0 at mSize=256.
                constexpr uint32_t MAX_REP = 255;
                uint32_t rowsLeft = scatterMRows;
                uint32_t mOff = 0;
                while (rowsLeft > 0) {
                    uint32_t chunk = (rowsLeft > MAX_REP) ? MAX_REP : rowsLeft;
                    for (uint32_t ng = 0; ng < ngCount; ng++)
                        Muls(copyOutBuf[mOff * baseN_ + ng * CUBE_N0],
                             finalOutputLT_[ng * srcNZStride + mOff * CUBE_N0],
                             scatterOne, CUBE_N0, static_cast<uint8_t>(chunk),
                             {1, 1, static_cast<uint8_t>(scatterDstRS), 1});
                    rowsLeft -= chunk;
                    mOff += chunk;
                }

                // Pertoken: finalOutput[m, n] = copyOutBuf[m, n] * pt[m].
                // Both Q==1 and Q>1 use a single element-wise Mul
                // against the pre-broadcast scratch (mTile-local, reused
                // across every cb in the mTile).
                if (hasPertoken_) {
                    PipeBarrier<PIPE_V>();
                    LocalTensor<half> ptSrc =
                        (Q_ == 1) ? dequantScratchLT_ : ptBroadcastLT_;
                    Mul(finalOutputLT_, copyOutBuf, ptSrc, pertokenMulCount);
                }
                PipeBarrier<PIPE_V>();

                // CopyOut: Y[b, mStart..mEnd, nStart..+nSize] <- UB(ND).
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);   // V-pipe done writing UB
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);  // MTE3 safe to read UB
                uint64_t gmOff = yBatchOff +
                    static_cast<uint64_t>(mStart) * N_ + nStart;
                DataCopyParams outP = {
                    static_cast<uint16_t>(mSize),
                    static_cast<uint16_t>(ngCount),
                    static_cast<uint16_t>(baseNG_ - ngCount),
                    static_cast<uint16_t>(N1_ - ngCount)};
                // Pertoken path writes to finalOutputLT_; no-pertoken path
                // leaves scatter result in copyOutBuf. The (cb+Q) parity above
                // guarantees finalOutputLT_ and copyOutBuf are always distinct.
                LocalTensor<half> coSrc = hasPertoken_ ? finalOutputLT_ : copyOutBuf;
                DataCopy(yGm_[gmOff], coSrc, outP);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
                }  // close else (single-shot Phase D)

                // Release the coalesced scale buffer so the next MTE2 can
                // refill it once every VDEQ16 on it has completed. Frequency:
                //   Q > 1  -> one MTE2 per cb, one release per cb.
                //   Q == 1 -> one MTE2 per mTile, one release per mTile
                //            (signalled on the last cb only).
                if (scaleCoalesce_ && (Q_ > 1 || !hasNextIter)) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                }
            }

            // Pertoken back-edge: one per mTile to keep semaphore balanced.
            // With coalesce, the per-mTile MTE2 is skipped so the back-edge
            // is unnecessary (no V_MTE2 token to refresh).
            if (hasPertoken_ && !pertokenCoalesce_) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
            }
            // V->MTE2 back-edge for the dequantScratchLT_/xPingLT_ alias.
            // Set after V's last write to dequantScratchLT_ in this vmIdx
            // (last ch's VDEQ16 in the last cb). The next vmIdx's Phase X
            // MTE2 waits on it before overwriting xPingLT_.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        }  // close outer batch loop
        }  // close nTSuperIdx loop
        }  // close mTSuperIdx loop

        // === DRAIN: consume outstanding events ===
        // Drain both L0 ping/pong M_MTE1 tokens (primed before the loop).
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        WaitFlag<HardEvent::V_M>(EVENT_ID0);
        if (dbL0c_) {
            WaitFlag<HardEvent::V_M>(EVENT_ID1);
        }
        if (!isPerTensor_ && scaleType_ != SCALE_UINT64) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2); // fp32 staging
        }
        // Drain the 2 init MTE1_MTE2 tokens (last cb sets one that has no prefetch)
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        if (hasPertoken_ && !pertokenCoalesce_) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        }
        // Drain the V_MTE2(EVENT_ID5) flag for the dequantScratch/xPing
        // alias. Init sets 1; per-vmIdx the body sets 1 and waits 1; the
        // initial set is the surplus drained here.
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        if (scaleCoalesce_) {
            // Drain the init prime: init set 1, loop set N cbs, loop waited N
            // cbs, so one surplus set remains to be drained here.
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        }
    }

}  // namespace QBM

#endif  // QUANT_BATCH_MATMUL_V3_X_INT8_PROCESS_H
