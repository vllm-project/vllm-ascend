/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_x_int8.h
 * \brief Quantized batched matmul (int8 x int8 -> int32 -> fp16) on 310P.
 * \author Feodor Pisnitchenko
 *
 * Math:
 *   Y = (X1 @ X2) * scale   [* perTokenScale]
 *
 *   Block form (used when scale is grouped along K with Q groups,
 *   Kq = K / Q; collapses to the line above when Q = 1):
 *
 *     Y[b, m, n] = pt[b*M + m] *
 *                  \sum_{q=0..Q-1} fp16(
 *                    scale[q, n] *
 *                    \sum_{k = q*Kq .. (q+1)*Kq - 1} X1[b, m, k] * X2[b, k, n]
 *                  )
 *
 *   pt is omitted when perTokenScale is absent. scale is [Q, N]
 *   grouped or [1] per-tensor (broadcast).
 *
 * Shapes / dtypes:
 *   X1     [B, M, K]   int8    ND
 *   X2     [B, K, N]   int8    FRACTAL_NZ ([K1, N1, N0=16, K0=32])
 *   scale  [Q, N]      u64     high32 = VDEQ16 marker 0x4000,
 *                              low32  = fp32 bits of scale value
 *          or [1]      u64     per-tensor scalar
 *   pt     [B*M]       fp32    optional
 *   Y      [B, M, N]   fp16
 *   K0 = 32, N0 = 16, M0 = 16 (Mmad fragment shape).
 *
 * Why: GM->L1 weight bandwidth is the bottleneck. 8 AICs sharing L2
 * only saves bandwidth when they read the SAME GM address; we put
 * all cores on the same batchIdx and split only mTiles. When M is
 * too small for that, fall back to a flat vmIdx split (correctness
 * preserved, L2 reuse sacrificed).
 *
 * Tile / iteration:
 *   Per-core: slice of mTiles (split across MCoreNum) and slice of
 *   cbs (NCoreNum forced to 1 when fracM >= aicNum so cores share
 *   the cb range). baseM x baseN inner tile, K-pass loop over Kq
 *   with baseK columns. cb covers baseN slice along N.
 *
 *     for batchIdx in [0..B):                       // all cores share
 *       for mTileIdx in core's mTile range:         // split across cores
 *         Phase 0: pt[b*M + mStart..+mSize] GM->UB
 *         Phase X: X1[b, mStart..mEnd, 0..K] GM->UB->L1 (ZZ fractal)
 *         for cb in [cbStart..cbEnd):
 *           for q in [0..Q):
 *             load scale[q, cb*baseN..(cb+1)*baseN] GM->UB
 *             for kp in K-passes of length baseK:
 *               L0A <- xL1 slice                    // ZZ -> L0A
 *               L0B <- wL1 slice                    // ZN -> L0B
 *               L0C += L0A @ L0B                    // Mmad int32 acc
 *             UB[m, n] <- fp16(L0C * scale[q, n])   // VDEQ16
 *             foOut    += UB                        // channel acc (NZ)
 *           NZ->ND scatter (V Muls) -> copyOutBuf
 *           if pt: copyOutBuf[m, n] *= pertokenLT[m]
 *           Y[b, mStart..mEnd, cb*baseN..] <- copyOutBuf  // MTE3
 *
 * Per-pipe:
 *   MTE2  GM->UB (X1, scale, pertoken), GM->L1 (W).
 *   V     X1 zero-pad, VDEQ16, NZ->ND Muls, pertoken Mul.
 *   MTE3  UB->L1 (Phase X scatter), UB->GM (Y CopyOut).
 *   MTE1  L1->L0A (X), L1->L0B (W).
 *   M     Mmad K-pass loop, L0C int32 acc.
 *   S     Scalar loop bookkeeping (loop indices, GetValue on scale /
 *         pertoken, event SetFlag/WaitFlag dispatch).
 *
 * Event channels:
 *   M_MTE1 / MTE1_M (ID0)   L0A/L0B ownership
 *   MTE2_MTE1       (ID1)   W L1 ready for MTE1
 *   MTE2_V          (ID3)   pertoken loaded
 *   MTE2_V          (ID4)   scale ready for VDEQ16
 *   MTE2_V          (ID5)   Phase X K-chunk loaded
 *   MTE2_V          (ID6)   fp32 scale staging loaded
 *   V_MTE2          (ID2)   fp32 scale staging free
 *   V_MTE2          (ID3)   pertoken back-edge per mTile
 *   V_MTE2          (ID5)   dequantScratch / xPing alias gate
 *   V_MTE2          (ID7)   coalesced scale buffer free
 *   M_V             (ID3)   L0C ready for VDEQ16
 *   V_M             (ID0)   L0C free for next Mmad
 *   V_MTE3          (ID0)   V done, MTE3 safe to read UB
 *   MTE3_V          (ID6)   CopyOut done (foPing/foPong free)
 *   MTE3_V          (ID7)   Phase X scatter done
 *   MTE3_MTE1       (ID0)   xL1 ready (scatter -> MTE1 LoadX)
 *   MTE3_MTE2       (ID0)   Phase X UB ping/pong free for MTE2
 *   MTE1_MTE2       (ID0/1) wL1Ping/wL1Pong free (MTE1 -> MTE2)
 */

#ifndef QUANT_BATCH_MATMUL_V3_X_INT8_H
#define QUANT_BATCH_MATMUL_V3_X_INT8_H

#include "quant_batch_matmul_v3_x_utils.h"
#include "quant_batch_matmul_v3_x_config.h"
#include "kernel_operator.h"

namespace QBM {

constexpr uint32_t K0_N0 = CUBE_K0_INT8 * CUBE_N0;  // 512 = ZN fractal tile bytes

template <int SCALE_TYPE>
class QBMInt8Compute {
public:
    // ============================================================
    // Method declarations. Implementations live in per-method
    // headers #include'd at the end of this file.
    // ============================================================
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y, GM_ADDR workspace,
        const QBMParams* params, TPipe* tPipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void X1ChunkMte2(LocalTensor<int8_t> ubDst, uint64_t gmSrcOff,
                                        uint32_t mSize, uint32_t chunkBytes);

    __aicore__ inline void ScatterZZToL1(uint32_t mAligned,
                                          uint32_t kByteOff,
                                          uint32_t chunkSize,
                                          LocalTensor<int8_t> ubSrc);

    __aicore__ inline void PrefetchWeightToL1(uint32_t ch, uint32_t colBatch,
                                               uint32_t kpBaseK, uint32_t kSize,
                                               uint64_t x2BatchOff,
                                               uint32_t nSize,
                                               bool usePing);

    // ============================================================
    // Member var declarations.
    // ============================================================

    TPipe* pipe_ = nullptr;

    // Tiling params
    uint32_t coreIdx_, coreNum_;
    uint32_t MCoreNum_, NCoreNum_;
    uint32_t batch_, M_, K_, N_;
    uint32_t baseM_, baseN_;
    uint32_t hasPertoken_;
    uint32_t Q_, Kq_;
    uint32_t baseK_, kPasses_;
    uint32_t wChunkKPasses_;            // kPasses per wL1 ping/pong chunk
    uint32_t wL1ChunkBytes_;            // wChunkKPasses_ * baseK_ * baseN_
    uint32_t phaseXChunk_;              // Phase X K-chunk size (tiling-driven)
    uint32_t phaseXFullPairs_;          // number of complete ping+pong iterations
    uint32_t phaseXTail_;              // tail K-bytes after full pairs (0 = no tail)
    uint32_t tailPingSize_;            // tail ping chunk size (<= phaseXChunk_)
    uint32_t tailPongSize_;            // tail pong chunk size (0 if tail <= chunk)
    bool     pingPong_;                 // true when phaseXChunk_ < K_
    uint32_t scaleType_;
    uint32_t isPerTensor_;              // 1=scalar scale (broadcast once), 0=per-channel
    uint32_t scaleCoalesce_;            // 1 = load all Q scales once per cb into scaleU64PingLT_
    uint32_t l2HintMode_;               // CacheMode int (0=DISABLE,1=NORMAL,4=PERSISTENT)
    uint32_t mTileCntL2_;               // # M super-tiles (1 = no split)
    uint32_t nTileCntL2_;               // # N super-tiles (1 = no split)
    uint32_t mTileBlock_;               // baseM blocks per M super-tile
    uint32_t nTileBlock_;               // baseN blocks per N super-tile
    uint32_t dbL0c_;                    // 1 = L0C ping-pong (alternate halves per cb)
    uint32_t pertokenCoalesce_;         // 1 = single MTE2 covers all B*M; 0 = per-mTile
    uint32_t hasBias_;                  // 1 = bias[N] int32 broadcast into L0C before Mmad
    uint32_t ubCalcM_;                  // Inner-tile Phase D rows per mu (0 = disabled)
    uint32_t kTail_;                    // K % 32 (0 = K0-aligned; >0 = partial last K0-group)
    uint32_t kPadded_;                  // K rounded up to K0=32 (= K_ when kTail_==0)
    uint32_t weightBlockN1_;            // blocked ZN: N-groups per block (0=standard ZN)

    // Precomputed strides (set in Init, avoid repeated multiply in hot loops)
    uint32_t N1_, K1q_;                 // ZN tile counts: N/16, Kq/32
    uint32_t baseNG_;                   // baseN / CUBE_N0
    uint32_t baseKFracs_;               // baseK / CUBE_K0_INT8
    uint64_t MK_, MN_;                  // M*K, M*N -- X1 / Y per-batch byte strides
    uint64_t x2FracTotal_;              // K*N bytes -- X2 per-batch byte stride (compact GM int8)
    uint64_t wGmChStride_;              // (K/Q)*N bytes -- X2 per-channel byte stride within a batch
    uint32_t l0aTurnStride_;            // CUBE_M0 * K_padded -- L1->L0A per-turn stride
    uint32_t l0aKpStride_;              // baseKFracs * K0_INT8 * M0 -- K-pass L0A increment
    uint32_t l1TurnStride_;             // CUBE_M0 * K_padded -- ScatterZZ L1 stride

    // GM tensors
    GlobalTensor<int8_t> x1Gm_;           // X1 [B, M, K] int8
    GlobalTensor<int8_t> x2Gm_;           // X2 [B, K, N] int8 ZN fractal
    GlobalTensor<half> yGm_;              // Y  [B, M, N] fp16
    GlobalTensor<uint64_t> scaleGm_;      // scale [Q, N] uint64 (pre-encoded)
    GlobalTensor<float> scaleGmFp32_;     // scale [Q, N] float32 (raw)
    GlobalTensor<float> pertokenGm_;      // pertoken [B*M] float
    GlobalTensor<int32_t> biasGm_;        // bias [N] int32 (broadcast into L0C)

    // L0 buffers
    TBuf<TPosition::A2> l0aBuf_;          // L0A: M_a * Kq int8
    TBuf<TPosition::B2> l0bBuf_;          // L0B: Kq * Nb int8 ZN
    TBuf<QuePosition::CO1> l0cBuf_;       // L0C: M_a * Nb int32

    // L1 buffers
    TBuf<TPosition::A1> xL1Buf_;          // X1 ZZ fractal [M_a * K]
    TBuf<TPosition::B1> wL1Ping_;         // weight ping [Kq * Nb]
    TBuf<TPosition::B1> wL1Pong_;         // weight pong [Kq * Nb]

    // UB -- persistent tensors
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<half> outPing_;           // output ping [M_a * Nb] fp16
    LocalTensor<half> outPong_;           // output pong [M_a * Nb] fp16
    // Runtime alias to one of outPing_/outPong_; reassigned per cb (and per
    // ch>0 Add) to point at the current channel-accumulation target.
    LocalTensor<half> finalOutputLT_;
    LocalTensor<float> pertokenLT_;       // per-token scale [M_a] float
    LocalTensor<uint64_t> scaleU64PingLT_;   // VDEQ16 scale ping [Nb] uint64
    LocalTensor<uint64_t> scaleU64PongLT_;   // VDEQ16 scale pong [Nb] uint64
    LocalTensor<uint32_t> scatterIdxLT_;    // scatter offset table [Nb] uint32 (persistent)
    LocalTensor<int32_t> ubBias_;           // bias [N] int32 (persistent, loaded once at Process entry)

    // UB -- Phase X workspace (aliased with Phase D at offWs)
    LocalTensor<int8_t> xPingLT_;         // X1 ping [M_a * kHalf] int8
    LocalTensor<int8_t> xPongLT_;         // X1 pong [M_a * kHalf] int8

    // UB -- Phase D workspace (aliased with Phase X at offWs)
    LocalTensor<half> dequantScratchLT_;  // VDEQ16 scratch + pertoken broadcast [M_a * Nb] fp16
    LocalTensor<float> scaleFp32StagingLT_; // float32 scale staging [Nb] float
    LocalTensor<half> ptBroadcastLT_;     // Q>1 pertoken pre-broadcast [M_a * Nb] fp16 (allocated only when Q>1 + hasPertoken)
};

}  // namespace QBM

// ============================================================
// Method implementations (per-method headers).
// These are NOT standalone -- they assume the class declaration
// above is in scope. Do not include them directly.
// ============================================================
#include "quant_batch_matmul_v3_x_int8_init.h"
#include "quant_batch_matmul_v3_x_int8_helpers.h"
#include "quant_batch_matmul_v3_x_int8_process.h"

#endif  // QUANT_BATCH_MATMUL_V3_X_INT8_H
