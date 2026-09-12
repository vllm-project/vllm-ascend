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
 * \file quant_batch_matmul_v3_x_int8_helpers.h
 * \brief INT8 kernel helpers: X1ChunkMte2 / ScatterZZToL1 / PrefetchWeightToL1 implementations.
 * \author Feodor Pisnitchenko
 *
 * Included by quant_batch_matmul_v3_x_int8.h AFTER the class
 * declaration; do not include directly.
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_INT8_HELPERS_H
#define QUANT_BATCH_MATMUL_V3_X_INT8_HELPERS_H

namespace QBM {

    // MTE2 read: ubDst[r, k] <- X1[b, mStart+r, gmKOff + k]
    //   r in [0, mSize), k in [0, chunkBytes).
    // Per-row UB stride = AlignUp(chunkBytes, 32) so the burst is
    // K0-block-aligned. When K is K0-aligned the canonical multi-burst
    // form is used; otherwise the kernel falls back to a per-row
    // single-burst loop because 310P MTE2 expresses srcStride only in
    // 32-byte units -- non-aligned (K - chunkBytes) silently truncates
    // and drifts between rows.
template <int SCALE_TYPE>
__aicore__ inline void QBMInt8Compute<SCALE_TYPE>::X1ChunkMte2(LocalTensor<int8_t> ubDst, uint64_t gmSrcOff,
                                                                uint32_t mSize, uint32_t chunkBytes)
    {
        if (kTail_ == 0) {
            const uint16_t burstLen = static_cast<uint16_t>(chunkBytes / UB_BLOCK_SIZE);
            const uint16_t srcStride = static_cast<uint16_t>((K_ - chunkBytes) / UB_BLOCK_SIZE);
            // ubDst[r, 0..chunkBytes) <- x1Gm[gmSrcOff + r*K_, 0..chunkBytes)  (multi-burst MTE2, mSize rows)
            DataCopy(ubDst, x1Gm_[gmSrcOff],
                     {static_cast<uint16_t>(mSize), burstLen, srcStride, 0});
        } else {
            const uint16_t alignedBurst =
                static_cast<uint16_t>((chunkBytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE);
            const uint32_t alignedBytes = alignedBurst * UB_BLOCK_SIZE;
            for (uint32_t r = 0; r < mSize; r++) {
                // ubDst[r, 0..alignedBytes) <- x1Gm[gmSrcOff + r*K_, 0..alignedBytes)  (single-burst per row)
                DataCopy(ubDst[r * alignedBytes],
                         x1Gm_[gmSrcOff + static_cast<uint64_t>(r) * K_],
                         {1, alignedBurst, 0, 0});
            }
        }
    }

    // MTE3 scatter UB(ND) -> L1(ZZ).
    //   xL1[turn, row, kFrac, kByte] <- ubSrc[turn*M0+row, kFrac*K0+kByte]
    // ZZ = [mAligned/M0 turns][M0=16 rows][K/K0=32 fracs][K0=32 bytes].
    // chunkSize = K-bytes scattered this call; kByteOff = K offset in L1.
template <int SCALE_TYPE>
__aicore__ inline void QBMInt8Compute<SCALE_TYPE>::ScatterZZToL1(uint32_t mAligned,
                                                                  uint32_t kByteOff,
                                                                  uint32_t chunkSize,
                                                                  LocalTensor<int8_t> ubSrc)
    {
        LocalTensor<int8_t> xL1 = xL1Buf_.Get<int8_t>();
        constexpr uint32_t zzDstStride = CUBE_M0 - 1;
        uint16_t chunkBlocks = static_cast<uint16_t>(chunkSize / UB_BLOCK_SIZE);
        DataCopyParams zzP = {
            chunkBlocks, 1, 0, static_cast<uint16_t>(zzDstStride)};
        uint32_t kFracOff = kByteOff * CUBE_M0;
        uint32_t m0Chunk = CUBE_M0 * chunkSize;

        uint32_t zzTurns = mAligned / CUBE_M0;
        uint32_t l1Base = kFracOff;
        uint32_t ubBase = 0;
        for (uint32_t t = 0; t < zzTurns; t++) {
            uint32_t l1r = l1Base;
            uint32_t ubr = ubBase;
            for (uint32_t r = 0; r < CUBE_M0; r++) {
                // xL1[turn=t, row=r, kFrac, kByte] <- ubSrc[t*M0+r, kFrac*K0+kByte]
                //   (one MTE3 burst writes one row's chunkBlocks blocks into ZZ slots)
                DataCopy(xL1[l1r], ubSrc[ubr], zzP);
                l1r += UB_BLOCK_SIZE;
                ubr += chunkSize;
            }
            l1Base += l1TurnStride_;
            ubBase += m0Chunk;
        }
    }


    // MTE2 prefetch:  wL1 <- W[ch, colBatch*baseN .. +nSize,
    //                            kpBaseK .. +kSize]
    // Standard ZN  : [K1, N1, N0=16, K0=32]; srcStride = (N1 - ng) * 16.
    // Blocked  ZN  : compact concat of column blocks, full blocks are
    //                blockNG wide and the last block is actualBlockNG;
    //                srcStride = (actualBlockNG - ng) * 16.
template <int SCALE_TYPE>
__aicore__ inline void QBMInt8Compute<SCALE_TYPE>::PrefetchWeightToL1(uint32_t ch, uint32_t colBatch,
                                                                       uint32_t kpBaseK, uint32_t kSize,
                                                                       uint64_t x2BatchOff,
                                                                       uint32_t nSize,
                                                                       bool usePing)
    {
        LocalTensor<int8_t> wL1 = usePing
            ? wL1Ping_.Get<int8_t>() : wL1Pong_.Get<int8_t>();

        constexpr uint32_t FRAC_BLOCKS = K0_N0 / UB_BLOCK_SIZE;  // 16
        uint32_t ngCount = nSize / CUBE_N0;
        uint32_t blockLen = ngCount * FRAC_BLOCKS;
        uint32_t kpK1Off = kpBaseK / CUBE_K0_INT8;   // K1-row offset for this K-pass

        // Split K-range into full K-fractals + optional partial K-fractal.
        // K_aligned_down = floor(K / 32) * 32. The partial K-fractal (when
        // kTail_ > 0) holds K_tail K-bytes per (N1, N0) entry, packed in
        // compact GM layout [N1, N0, K_tail] after the K_aligned_down * N
        // bytes of full K-fractals.
        uint32_t kAlignedDown = (K_ / CUBE_K0_INT8) * CUBE_K0_INT8;
        uint32_t kEnd         = kpBaseK + kSize;
        uint32_t kEndFull     = Min(kEnd, kAlignedDown);
        uint32_t kFullSize    = (kEndFull > kpBaseK) ? (kEndFull - kpBaseK) : 0;
        bool needsPartial     = (kTail_ > 0) && (kEnd > kAlignedDown);
        uint32_t kfracsFull   = kFullSize / CUBE_K0_INT8;

        uint64_t wGmOff;
        uint32_t effN1;  // effective N1 for srcStride computation
        uint32_t ngStart = colBatch * baseNG_;

        if (weightBlockN1_ == 0) {
            // Standard ZN: [K1, N1, N0, K0]
            wGmOff = x2BatchOff +
                static_cast<uint64_t>(ch) * wGmChStride_ +
                static_cast<uint64_t>(kpK1Off) * N1_ * K0_N0 +
                static_cast<uint64_t>(ngStart) * K0_N0;
            effN1 = N1_;
        } else {
            // Compact blocked ZN. Full blocks (every block but the
            // last) are blockNG wide; the last block packs the remainder
            // at actualBlockNG. blockNG must divide every possible
            // baseNG so colBatch never spans blocks.
            uint32_t blockNG = weightBlockN1_;
            uint32_t blockIdx = ngStart / blockNG;
            uint32_t ngInBlock = ngStart - blockIdx * blockNG;
            uint32_t actualBlockNG = Min(blockNG, N1_ - blockIdx * blockNG);

            // Skip prior full blocks at width blockNG, then within-block
            // strides at width actualBlockNG (which equals blockNG for
            // full blocks; the last block is partial).
            uint32_t K1Total = K_ / CUBE_K0_INT8;
            uint64_t totalBlockSize = static_cast<uint64_t>(K1Total) * blockNG * K0_N0;
            uint64_t chOffInBlock = static_cast<uint64_t>(ch) * K1q_ * actualBlockNG * K0_N0;
            wGmOff = x2BatchOff +
                static_cast<uint64_t>(blockIdx) * totalBlockSize +
                chOffInBlock +
                static_cast<uint64_t>(kpK1Off) * actualBlockNG * K0_N0 +
                static_cast<uint64_t>(ngInBlock) * K0_N0;
            effN1 = actualBlockNG;
        }

        // Part 1: full K-fractals (standard ZN, multi-burst MTE2).
        if (kfracsFull > 0) {
            uint32_t srcStride = (effN1 - ngCount) * FRAC_BLOCKS;
            DataCopyParams wP = {
                static_cast<uint16_t>(kfracsFull),
                static_cast<uint16_t>(blockLen),
                static_cast<uint16_t>(srcStride), 0};
            // wL1[k1, n1, n0, k0] <- x2Gm[ch, kpK1Off+k1, ngStart+n1, n0, k0]
            //   k1 in [0, kfracsFull), n1 in [0, ngCount), packed in 32-B blocks
            DataCopy(wL1, x2Gm_[wGmOff], wP);
        }

        // Part 2: partial K-fractal (compact [N1, N0, K_tail] in GM).
        // Per-N0 outer loop with multi-burst inner DataCopy. Each call
        // reads 1 block (= 32 B) per N-group = K_tail real bytes + K_tail
        // overread (next N0's K_tail data, harmless since L0A K-tail is
        // V-pipe zeroed in Phase X so Mmad sum_{K=K_tail..K0-1} = 0).
        // Strides in 32-B blocks; gated by (CUBE_N0 * kTail_) % 32 == 0
        // (i.e. kTail must be EVEN for the per-N0 burst stride to land
        // on a block boundary; an odd kTail breaks the stride).
        if (needsPartial) {
            // Compact partial section starts after K_aligned_down * N bytes
            // of full K-fractals (Q == 1 path; Q > 1 forbids K-tail so this
            // branch never runs there).
            uint64_t partialGmBase = x2BatchOff +
                static_cast<uint64_t>(kAlignedDown) * N_ +
                static_cast<uint64_t>(ngStart) * CUBE_N0 * kTail_;

            // wL1 destination: K-fractal slot immediately after the full
            // K-fractals already written by Part 1 in this chunk.
            uint32_t wL1PartialOff = kfracsFull * blockLen * UB_BLOCK_SIZE;

            uint16_t partialBlockLen   = 1;  // 32 B per burst
            uint16_t partialSrcStride  =
                static_cast<uint16_t>((CUBE_N0 * kTail_) / UB_BLOCK_SIZE - 1);
            uint16_t partialDstStride  =
                static_cast<uint16_t>(CUBE_N0 - 1);  // wL1 N1 stride = N0*K0 -> N0-1 blocks gap

            for (uint32_t n0 = 0; n0 < CUBE_N0; n0++) {
                DataCopyParams wPp = {
                    static_cast<uint16_t>(ngCount),
                    partialBlockLen, partialSrcStride, partialDstStride};
                // wL1[k1=last, n1, n0, 0..K0) <- x2Gm[partial section][n1, n0, 0..kTail)
                //   one 32-B burst per n1, kTail real bytes + (K0-kTail) GM overread per burst
                DataCopy(
                    wL1[wL1PartialOff + n0 * CUBE_K0_INT8],
                    x2Gm_[partialGmBase + static_cast<uint64_t>(n0) * kTail_],
                    wPp);
            }
        }
    }

}  // namespace QBM

#endif  // QUANT_BATCH_MATMUL_V3_X_INT8_HELPERS_H
