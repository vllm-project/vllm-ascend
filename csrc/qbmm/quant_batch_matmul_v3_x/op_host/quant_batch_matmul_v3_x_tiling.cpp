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
 * \file quant_batch_matmul_v3_x_tiling.cpp
 * \brief Tile selection for QuantBatchMatmulV3X: candidate enumeration + cost-model pick + pertoken/scale coalesce.
 * \author Feodor Pisnitchenko
 *
 * Work = batch * mTiles * colBatches, block-distributed across AIC
 * cores. Picker prefers smaller baseN when a larger choice leaves
 * cores idle, then refines with the cycle cost model.
 */
#include "quant_batch_matmul_v3_x_tiling.h"

#include <algorithm>
#include <vector>
#include "register/op_impl_registry.h"
#include "../op_kernel/quant_batch_matmul_v3_x_tiling_key.h"
#include "../op_kernel/quant_batch_matmul_v3_x_config.h"
#include "tiling_cost_model.h"

#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif

using namespace ge;

namespace optiling {

static inline uint32_t AlignUpU(uint32_t val, uint32_t align) {
    return ((val + align - 1) / align) * align;
}

static inline uint32_t AlignDownU(uint32_t val, uint32_t align) {
    return (val / align) * align;
}

static void BuildKbCandidates(uint32_t K, std::vector<uint32_t> &out);

// ============================================================
// ComputeInt8Tiling -- choose (baseM, baseN) for INT8*INT8
//
// Constraints (310P, int8):
//   L0A: M * baseK               <= 64 KB  (K-split per pass)
//   L0B: baseK * Nb              <= 64 KB  (K-split per pass)
//   L0C: M * Nb * 4             <= 256 KB
//   L1:  M*K + 2*Kq*Nb          <= 1 MB
//   UB:  persistent + max(phaseX, phaseD) <= 256 KB
//
// Selection: fills all cores > fewer colBatches > larger baseM.
// ============================================================
struct Int8TilingResult {
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t phaseXChunk;
    uint32_t wChunkKPasses;   // kPasses fetched per MTE2 into wL1 ping/pong
};

// Coalesce paths and per-mTile scale-MTE2 savings:
//   Q == 1 -> [N] u64 once per mTile  (size = N*8 B,        save cbs-1).
//   Q  > 1 -> [Q*baseN] u64 once/cb   (size = Q*baseN*8 B,  save (Q-1)*cbs).
// Budget caps coalesce buffer so Phase X / Phase D workspace still fits
// in the 256 KB UB; 128 KB leaves >= 128 KB for workspace.
constexpr uint64_t QBMV3_SCALE_COALESCE_UB_BUDGET = 128 * 1024;

static uint32_t ComputeScaleCoalesce(uint32_t Q, uint32_t baseN, uint32_t N,
                                      uint32_t isPerTensor,
                                      uint32_t scaleType)
{
    if (isPerTensor != 0) return 0;                  // scalar path already 1*/mTile
    // fp32 path: Q>1 coalesce still needs a larger staging path;
    // restrict to Q == 1.
    if (scaleType != 2 && Q > 1) return 0;
    const uint64_t bytes = (Q <= 1)
        ? static_cast<uint64_t>(N) * 8
        : static_cast<uint64_t>(Q) * baseN * 8;
    if (bytes > QBMV3_SCALE_COALESCE_UB_BUDGET) return 0;
    return 1;
}

// Compute wChunkKPasses -- how many kPasses of weight fit in one
// ping/pong buffer given the L1 budget left over from xL1.
static uint32_t ComputeWChunkKPasses(uint32_t baseM, uint32_t baseN,
                                      uint32_t baseK, uint32_t K,
                                      uint32_t Kq, uint64_t l1Size)
{
    const uint32_t kPasses = (Kq + baseK - 1) / baseK;
    if (kPasses == 0) return 1;
    const uint64_t xL1 = static_cast<uint64_t>(baseM) * K;
    if (xL1 >= l1Size) return 1;
    const uint64_t perSide = (l1Size - xL1) / 2;   // ping + pong
    const uint64_t perKPass = static_cast<uint64_t>(baseK) * baseN;
    if (perKPass == 0) return 1;
    uint64_t fit = perSide / perKPass;
    if (fit < 1) fit = 1;
    if (fit > kPasses) fit = kPasses;
    return static_cast<uint32_t>(fit);
}

static Int8TilingResult ComputeInt8Tiling(
    uint32_t M, uint32_t N, uint32_t K, uint32_t Q,
    uint32_t batch, uint32_t aicNum,
    uint64_t ubSize, uint64_t l0aSize, uint64_t l0bSize,
    uint64_t l0cSize, uint64_t l1Size,
    uint32_t isPerTensor, uint32_t scaleType,
    uint32_t hasPertoken)
{
    constexpr uint32_t ROW_ALIGN = 16;   // Mmad M-tile
    constexpr uint32_t COL_ALIGN = 16;   // FRACTAL_NZ N0
    constexpr uint32_t K_ALIGN = 32;     // CUBE_K0_INT8

    uint32_t Kq = (Q > 0) ? K / Q : K;

    // Heuristic selector state.
    uint32_t bestM = 0, bestN = 0, bestK = 0, bestChunk = 0, bestCB = UINT32_MAX;
    bool bestFills = false;

    // Cost-model selector: collect every feasible candidate, pick argmin later.
    std::vector<Int8TilingResult> candidates;

    std::vector<uint32_t> kbCands;
    BuildKbCandidates(Kq, kbCands);
    if (kbCands.empty()) return {0, 0, 0, 0};

    // L0 ping/pong: each L0 buffer holds two K-passes, so per-pass budget
    // is half the physical L0 size. Halved here; kernel alternates offsets.
    const uint64_t l0aHalf = l0aSize / 2;
    const uint64_t l0bHalf = l0bSize / 2;

    constexpr uint32_t NUM_CANDIDATES = sizeof(qbmv3::NB_CANDIDATES) / sizeof(qbmv3::NB_CANDIDATES[0]);
    for (uint32_t ci = 0; ci < NUM_CANDIDATES; ci++) {
        uint32_t baseN = qbmv3::NB_CANDIDATES[ci];
        if (baseN > N) continue;
        if (baseN % COL_ALIGN != 0) continue;

        // L0B filter: smallest Kb candidate must fit the half-buffer at this Nb.
        if (static_cast<uint64_t>(kbCands.back()) * baseN > l0bHalf) continue;

        // L0C: baseM * baseN * 4 <= l0cSize
        uint32_t maxM_l0c = static_cast<uint32_t>(l0cSize / (static_cast<uint64_t>(baseN) * 4));
        maxM_l0c = AlignDownU(maxM_l0c, ROW_ALIGN);
        if (maxM_l0c < ROW_ALIGN) continue;

        // L1 pre-filter: even the smallest Kb must leave enough room for Mb.
        // If the smallest Kb can't fit Mb >= ROW_ALIGN, this baseN is hopeless.
        uint64_t wL1MinBoth = 2ULL * kbCands.back() * baseN;
        uint64_t l1MinForX = (l1Size > wL1MinBoth) ? (l1Size - wL1MinBoth) : 0;
        uint32_t maxM_l1Loose = static_cast<uint32_t>(l1MinForX / K);
        maxM_l1Loose = AlignDownU(maxM_l1Loose, ROW_ALIGN);
        if (maxM_l1Loose < ROW_ALIGN) continue;

        uint32_t maxM_cap = std::min(maxM_l0c, maxM_l1Loose);

        uint32_t scaleBytesPerN = 8;

        for (uint32_t tryM = std::min(maxM_cap, AlignUpU(M, ROW_ALIGN));
             tryM >= ROW_ALIGN;
             tryM -= ROW_ALIGN)
        {
            // Sweep Kb candidates descending; pick largest that clears L0A, L0B, L1.
            // Largest Kb minimises per-K-pass MTE2 setup overhead.
            //
            // The static kbCands list (powers-of-two of K plus a fixed
            // safety set) can skip the L0-ceiling baseK for a given
            // (tryM, baseN). Example: prefill-style (tryM=64, baseN=320)
            // has L0_max baseK = min(L0A/tryM, L0B/baseN aligned to 32)
            // = min(512, 96) = 96, but kbCands jumps from 128
            // (rejected by L0B) to 64, losing a third of the L0
            // ceiling. Prepending the per-(tryM, baseN) L0-fit baseK
            // to the candidate set recovers the gap when L1 also fits.
            uint32_t kbL0Fit = std::min<uint32_t>(
                static_cast<uint32_t>(l0aHalf / std::max<uint32_t>(tryM, 1U)),
                static_cast<uint32_t>(l0bHalf / std::max<uint32_t>(baseN, 1U)));
            kbL0Fit = std::min(kbL0Fit, qbmv3::MMAD_K_MAX);
            kbL0Fit = std::min(kbL0Fit, K);
            kbL0Fit = AlignDownU(kbL0Fit, qbmv3::K_FRACTAL_INT8_BYTES);
            uint32_t tryK = 0;
            // Try the L0-fit value first (if any), then fall back to the
            // static kbCands list.
            auto tryKb = [&](uint32_t kb) -> bool {
                if (kb < qbmv3::K_FRACTAL_INT8_BYTES) return false;
                if (static_cast<uint64_t>(tryM) * kb > l0aHalf) return false;
                if (static_cast<uint64_t>(kb) * baseN > l0bHalf) return false;
                uint64_t wL1 = 2ULL * kb * baseN;
                uint64_t xL1 = static_cast<uint64_t>(tryM) * K;
                if (xL1 + wL1 > l1Size) return false;
                tryK = kb;
                return true;
            };
            if (!tryKb(kbL0Fit)) {
                for (uint32_t kb : kbCands) {
                    if (tryKb(kb)) break;
                }
            }
            if (tryK == 0) continue;

            constexpr uint64_t UB_BANK_PAD = qbmv3::UB_BANK_PAD;
            uint64_t singleFo = static_cast<uint64_t>(tryM) * baseN * 2;
            // Pertoken UB region must match the kernel's ptSize. The kernel
            // allocates batch*M*4 bytes when pertokenCoalesce fires (the
            // persistent buffer holds the full B*M float32 vector and per-mTile
            // indexes into it), and tryM*4 bytes per-mTile otherwise. Mirror
            // that decision here using the same threshold (PT_COALESCE_BUDGET
            // = 4096) so the tile selector does not under-budget UB and pick
            // a chunk that the kernel will overflow at runtime.
            constexpr uint64_t PT_COALESCE_BUDGET_BYTES_LOCAL = 4096;
            const bool ptCoalesceLocal = (hasPertoken != 0) &&
                (static_cast<uint64_t>(batch) * M * 4U <= PT_COALESCE_BUDGET_BYTES_LOCAL);
            uint64_t ubPertoken = ptCoalesceLocal
                ? AlignUpU(static_cast<uint32_t>(batch * M * 4U), 32U)
                : AlignUpU(tryM * 4U, 32U);
            uint64_t ubScatterIdx = static_cast<uint64_t>(baseN) * 4;
            bool needScatterIdx = (!isPerTensor) && (scaleType != 2);
            // Scale coalesce decision must match kernel's InitBuffer exactly:
            //   Q == 1 coalesce -> one [N] uint64 ping, no pong
            //   Q  > 1 coalesce -> one [Q * baseN] uint64 ping, no pong
            //   pertensor      -> one [baseN] ping, no pong
            //   else           -> [baseN] ping + [baseN] pong
            const uint32_t coalesceLocal =
                ComputeScaleCoalesce(Q, baseN, N, isPerTensor, scaleType);
            uint64_t ubScalePing;
            if (coalesceLocal != 0) {
                ubScalePing = (Q <= 1) ? (static_cast<uint64_t>(N) * 8)
                                       : (static_cast<uint64_t>(Q) * baseN * 8);
            } else {
                ubScalePing = static_cast<uint64_t>(baseN) * scaleBytesPerN;
            }
            const bool allocScalePong = (isPerTensor == 0) && (coalesceLocal == 0);
            const uint64_t ubScalePong = allocScalePong
                ? (static_cast<uint64_t>(baseN) * scaleBytesPerN)
                : 0;
            uint64_t ubScaleTotal = ubScalePing + UB_BANK_PAD;
            if (allocScalePong) {
                ubScaleTotal += ubScalePong + UB_BANK_PAD;
            }
            uint64_t persistent = singleFo + UB_BANK_PAD + singleFo + UB_BANK_PAD +
                                  ubPertoken + UB_BANK_PAD + ubScaleTotal;
            if (needScatterIdx) {
                persistent += ubScatterIdx + UB_BANK_PAD;
            }

            // Phase X chunk search: try chunk=K single-buffer first (one MTE2
            // per mTile, no ping/pong overhead); fall back to halving ping/pong
            // chunks when K does not fit. Kernel switches modes by checking
            // phaseXChunk_ < K_.
            uint32_t chunk = 0;
            bool pxPingPong = true;
            uint64_t phaseXBytesFull =
                AlignUpU(static_cast<uint32_t>(tryM) * K, 32U);
            if (persistent + phaseXBytesFull <= ubSize) {
                chunk = K;
                pxPingPong = false;
            } else if (K % 64 == 0) {
                for (uint32_t c = AlignDownU(K / 2, K_ALIGN); c >= K_ALIGN;
                     c = AlignDownU(c / 2, K_ALIGN)) {
                    uint64_t phaseXBytes =
                        2ULL * AlignUpU(tryM * c, 32U) + UB_BANK_PAD;
                    if (persistent + phaseXBytes <= ubSize) {
                        chunk = c;
                        pxPingPong = true;
                        break;
                    }
                    if (c == K_ALIGN) break;
                }
            }
            if (chunk == 0) continue;

            uint64_t phaseX = pxPingPong
                ? (2ULL * AlignUpU(tryM * chunk, 32U) + UB_BANK_PAD)
                : AlignUpU(static_cast<uint32_t>(tryM) * K, 32U);
            // Phase D layout (must match init.h offset arithmetic):
            //   slot 0 @ offWs                : dequantScratchLT [singleFo]
            //   slot 1 @ offWs + singleFo+PAD : scaleFp32StagingLT
            //                                   [N*4 if fp32+coalesce, else baseN*4]
            //   slot 2 @ offWs + (slot 0 + slot 1 sizes + PADs) :
            //                                   ptBroadcastLT [singleFo]
            //                                   (only when Q>1 + hasPertoken)
            //
            // For non-pt path, the kernel uses ONLY slot 0 + slot 1,
            // packed tightly so high-tryM cells fit UB. The cost-gated
            // tail-aware override below stops the picker from preferring
            // a large-baseN cell (e.g. baseN=320) when the L2-hit
            // predictor flags it as bad.
            const bool needPtBroadcastLocal = (hasPertoken != 0) && (Q > 1);
            const uint64_t scaleStagingBytes =
                (coalesceLocal != 0 && scaleType != 2)
                    ? (static_cast<uint64_t>(N) * 4)
                    : (static_cast<uint64_t>(baseN) * 4);
            const uint64_t scaleStagingAligned =
                AlignUpU(static_cast<uint32_t>(scaleStagingBytes), 32U);
            uint64_t phaseD;
            if (needPtBroadcastLocal) {
                // Tight ptBroadcast layout: slot 2 sits at
                //   offWs + singleFo + PAD + scaleStagingAligned + PAD,
                // then ptBroadcast (singleFo) + PAD.
                phaseD = singleFo + UB_BANK_PAD +
                         scaleStagingAligned + UB_BANK_PAD +
                         singleFo + UB_BANK_PAD;
            } else {
                phaseD = singleFo + UB_BANK_PAD +
                         scaleStagingAligned + UB_BANK_PAD;
            }
            uint64_t workspace = std::max(phaseX, phaseD);

            if (persistent + workspace > ubSize) continue;

            uint32_t mCapped = std::min(tryM, AlignUpU(M, ROW_ALIGN));
            uint32_t mTiles = (M + mCapped - 1) / mCapped;
            uint32_t cb = (N + baseN - 1) / baseN;
            uint32_t work = batch * mTiles * cb;
            bool fills = (work >= aicNum);

            // Collect every feasible candidate for the cost-model selector.
            // Pre-compute wChunkKPasses + scaleCoalesce per candidate so the
            // cost model sees the actual MTE2 cadence.
            const uint32_t candWChunk = ComputeWChunkKPasses(
                tryM, baseN, tryK, K, Kq, l1Size);
            candidates.push_back({tryM, baseN, tryK, chunk, candWChunk});

            bool better = false;
            if (fills && !bestFills) {
                better = true;
            } else if (fills == bestFills) {
                better = (cb < bestCB || (cb == bestCB && tryM > bestM));
            }

            if (better) {
                bestM = tryM;
                bestN = baseN;
                bestK = tryK;
                bestChunk = chunk;
                bestCB = cb;
                bestFills = fills;
            }

        }
    }

    if (candidates.empty()) {
        return {0, 0, 0, 0, 0};
    }

    ProblemShape shape{};
    shape.B = batch;
    shape.M = M;
    shape.K = K;
    shape.N = N;
    shape.Q = Q;
    shape.hasPertoken = (hasPertoken != 0);
    shape.isPerTensor = (isPerTensor != 0);
    shape.scaleType   = scaleType;

    uint64_t bestCyc = UINT64_MAX;
    Int8TilingResult pick{0, 0, 0, 0, 0};
    for (const auto& c : candidates) {
        const uint32_t candCoalesce = ComputeScaleCoalesce(
            Q, c.baseN, N, isPerTensor, scaleType);
        TilingCandidate tile{c.baseM, c.baseN, c.baseK, c.phaseXChunk,
                             c.wChunkKPasses, candCoalesce};
        uint64_t cyc = EstimatePerCoreCriticalPath(tile, shape, aicNum);
        if (cyc < bestCyc) {
            bestCyc = cyc;
            pick = c;
        }
    }
    // Tail-aware Nb override (cost-gated): when a candidate has
    // strictly larger baseN AND strictly fewer cb AND a well-behaved
    // tail (exact divide or partial-cb >= 75 % of Nb), prefer it ONLY
    // if predicted_cyc <= bestCyc. Tie-breaker for cells the cost
    // model considers equivalent or better.
    if (pick.baseM > 0) {
        const uint32_t pickCbInit = (N + pick.baseN - 1) / pick.baseN;
        uint32_t bestCbCount = pickCbInit;
        for (const auto& c : candidates) {
            if (c.baseN <= pick.baseN) continue;
            const uint32_t candCb = (N + c.baseN - 1) / c.baseN;
            if (candCb >= bestCbCount) continue;
            const uint32_t tail = N % c.baseN;
            const bool tailOk = (tail == 0) ||
                (tail * 100U >= 75U * c.baseN);
            if (!tailOk) continue;
            const uint32_t candCoalesce = ComputeScaleCoalesce(
                Q, c.baseN, N, isPerTensor, scaleType);
            TilingCandidate ct{c.baseM, c.baseN, c.baseK,
                                c.phaseXChunk, c.wChunkKPasses,
                                candCoalesce};
            uint64_t candCyc = EstimatePerCoreCriticalPath(
                ct, shape, aicNum);
            if (candCyc > bestCyc) continue;
            bestCbCount = candCb;
            pick = c;
        }
        pick.baseM = std::min(pick.baseM, AlignUpU(M, ROW_ALIGN));
        pick.wChunkKPasses = ComputeWChunkKPasses(
            pick.baseM, pick.baseN, pick.baseK, K, Kq, l1Size);
    }
    // Heuristic fallback for small-M / memory-bound decode shapes: when
    // the heuristic picks a tile with strictly fewer cb, override the
    // cost picker. The cost model under-weights per-cb setup overhead
    // for low M.
    const bool memBoundDecode = (M <= 256U && (N >= 8192U || K >= 4096U));
    if (bestM > 0 && (M <= 128U || memBoundDecode)) {
        const uint32_t pickCb = (N + pick.baseN - 1) / pick.baseN;
        const uint32_t heurCb = (N + bestN - 1) / bestN;
        if (heurCb < pickCb || memBoundDecode) {
            pick.baseM = bestM;
            pick.baseN = bestN;
            pick.baseK = bestK;
            pick.phaseXChunk = bestChunk;
            pick.baseM = std::min(pick.baseM, AlignUpU(M, ROW_ALIGN));
            pick.wChunkKPasses = ComputeWChunkKPasses(
                pick.baseM, pick.baseN, pick.baseK, K, Kq, l1Size);
        }
    }
    return pick;
}

// ============================================================
// PickCoreGrid -- split aicNum cores across M and N axes
//
// Enumerates every factor pair (mc, nc) with mc*nc <= aicNum and
// picks the one that best balances per-core work. Score favours
// partitions that fully consume both axes; under-subscription is
// allowed when that gives a rounder split (e.g. 10 cores on a (31, 8)
// grid: (5, 2) uses all 10; (4, 2) uses 8 but divides M more evenly).
// Tie-break: larger total cores, then larger mc (more M locality).
// ============================================================
struct CoreGrid {
    uint32_t MCoreNum;
    uint32_t NCoreNum;
};

static CoreGrid PickCoreGrid(
    uint32_t aicNum, uint32_t fracM, uint32_t fracN,
    const TilingCandidate& tile,
    const ProblemShape& shape)
{
    CoreGrid best = {1, 1};
    uint64_t bestScore = UINT64_MAX;
    if (aicNum == 0) return best;

    // When fracM is large enough to give every core >=1 mTile, force
    // NCoreNum=1 so all cores walk the same cb range in lockstep --
    // concurrent same-address weight reads coalesce in L2. With
    // NCoreNum>1 the cb range is partitioned disjointly across cores
    // and reads do not coalesce.
    uint32_t ncRangeMax = aicNum;
    if (fracM >= aicNum) {
        ncRangeMax = 1;
    }

    for (uint32_t nc = 1; nc <= ncRangeMax && nc <= fracN; ++nc) {
        for (uint32_t mc = 1; mc * nc <= aicNum && mc <= fracM; ++mc) {
            // Per-core work quantum on each axis, rounded up.
            uint64_t perCoreM = (fracM + mc - 1) / mc;
            uint64_t perCoreN = (fracN + nc - 1) / nc;

            // L2-aware blended MTE2 cost. The first-touch + linear
            // spill cap handles K*N > L2 (spill regime) and K*N <=
            // L2 (fit regime) in one expression.
            uint64_t score = EstimatePerCoreWall(
                tile, shape,
                static_cast<uint32_t>(perCoreM),
                static_cast<uint32_t>(perCoreN),
                mc, nc);

            bool better = (score < bestScore)
                       || (score == bestScore && mc * nc > best.MCoreNum * best.NCoreNum)
                       || (score == bestScore && mc * nc == best.MCoreNum * best.NCoreNum && mc > best.MCoreNum);
            if (better) {
                bestScore = score;
                best = {mc, nc};
            }
        }
    }
    return best;
}

// ============================================================
// Kb candidate enumeration for the per-K-pass L1 policy.
//
// Divisors give integer kPasses for power-of-two K; safety-net absolutes
// cover shapes where divisors don't land on useful 32-aligned values.
// Output is 32-aligned, clamped to MMAD_K_MAX, deduped, descending.
// ============================================================
static void BuildKbCandidates(uint32_t K, std::vector<uint32_t> &out)
{
    out.clear();
    for (uint32_t d : qbmv3::KB_DIVISORS) {
        uint32_t kb = K / std::max(d, 1U);
        kb = AlignDownU(std::min(kb, qbmv3::MMAD_K_MAX), qbmv3::K_FRACTAL_INT8_BYTES);
        if (kb >= qbmv3::K_FRACTAL_INT8_BYTES) out.push_back(kb);
    }
    for (uint32_t kb : qbmv3::KB_SAFETY) {
        uint32_t v = std::min(kb, qbmv3::MMAD_K_MAX);
        v = AlignDownU(v, qbmv3::K_FRACTAL_INT8_BYTES);
        if (v >= qbmv3::K_FRACTAL_INT8_BYTES && v <= K) out.push_back(v);
    }
    std::sort(out.begin(), out.end(), std::greater<uint32_t>());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

ASCENDC_EXTERN_C ge::graphStatus TilingQBM(gert::TilingContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto x1Tensor = context->GetInputTensor(0);
    auto x2Tensor = context->GetInputTensor(1);
    if (x1Tensor == nullptr || x2Tensor == nullptr) return ge::GRAPH_FAILED;

    // Platform info is read directly from the TilingContext: the aclnn
    // runtime executor does not run the TilingParse callback, so a
    // GetCompiledInfo-based flow returns nullptr there and the tiling
    // fails with GRAPH_FAILED. Same approach as causal_conv1d_v310.
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) return ge::GRAPH_FAILED;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0U) return ge::GRAPH_FAILED;
    uint64_t ubSize = 0, l1Size = 0, l0ASize = 0, l0BSize = 0, l0CSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);

    // Get dimensions from x1 shape: [..., M, K]
    auto& x1Shape = x1Tensor->GetStorageShape();
    uint32_t dimNum = x1Shape.GetDimNum();
    uint32_t batch = 1, M = 0, K = 0;
    if (dimNum >= 3) {
        for (uint32_t i = 0; i < dimNum - 2; ++i) {
            batch *= x1Shape.GetDim(i);
        }
        M = x1Shape.GetDim(dimNum - 2);
        K = x1Shape.GetDim(dimNum - 1);
    } else {
        M = x1Shape.GetDim(0);
        K = x1Shape.GetDim(1);
    }

    // Get transpose_x2 attr (index 2 in attr list)
    bool transposeX2 = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto tx2 = attrs->GetAttrPointer<bool>(2);
        if (tx2 != nullptr) transposeX2 = *tx2;
    }

    // Get N from x2 (FRACTAL_NZ or ND, respecting transpose_x2)
    uint32_t N = static_cast<uint32_t>(
        DeriveN(x2Tensor->GetStorageShape(), transposeX2));

    // Detect optional pertoken_scale at IR index 5.
    // GetOptionalInputShape is safe for absent optional inputs; GetInputTensor
    // returns garbage for them during atc offline compilation.
    // Must check GetShapeSize > 0 to distinguish real input from zero-element placeholder.
    uint32_t hasPertoken = 0;
    auto* pertokenShape = context->GetOptionalInputShape(5);
    if (pertokenShape != nullptr &&
        pertokenShape->GetStorageShape().GetShapeSize() > 0) {
        hasPertoken = 1;
    }

    // Detect optional bias at IR index 4. Bias is int32 [N] added to the
    // int32 Mmad accumulator before VDEQ16 dequant:
    //   out = (x1 @ x2 + bias) * scale + offset
    // Implementation uses BroadCastVecToMM (UB -> L0C) before the first
    // Mmad of each cb, then Mmad with init_c=false accumulates onto bias.
    uint32_t hasBias = 0;
    auto* biasShape = context->GetOptionalInputShape(4);
    if (biasShape != nullptr &&
        biasShape->GetStorageShape().GetShapeSize() > 0) {
        hasBias = 1;
    }

    // scaleType: 0=FP32 (encode in kernel), 2=UINT64 (pre-encoded)
    uint32_t scaleType = 0;
    auto scaleDesc = context->GetInputDesc(2);
    if (scaleDesc != nullptr) {
        auto scaleDt = scaleDesc->GetDataType();
        if (scaleDt == ge::DT_UINT64 || scaleDt == ge::DT_INT64) scaleType = 2;
    }

    uint32_t quantGroupNum = 1;
    uint32_t isPerTensor = 0;
    auto scaleTensor = context->GetInputTensor(2);
    if (scaleTensor != nullptr) {
        auto& scaleShape = scaleTensor->GetStorageShape();
        if (scaleShape.GetDimNum() >= 2) {
            // Grouped-quant scale must be [Q, N]. Reject anything else early --
            // without the dim[1]==N check a [N, 1] shape silently becomes Q=N,
            // which reads past the allocated scale and dequants per-token.
            if (static_cast<uint32_t>(scaleShape.GetDim(1)) != N) {
                return ge::GRAPH_FAILED;
            }
            quantGroupNum = scaleShape.GetDim(0);
        }
        // Scalar scale: shape [1] or total elements == 1
        int64_t scaleElems = scaleShape.GetShapeSize();
        if (scaleElems == 1) {
            isPerTensor = 1;
        }
    }

    // Reject configs where the per-channel K-slice (Kq = K/Q) is not
    // a multiple of CUBE_K0_INT8 (=32). The kernel uses integer division
    // K1q = Kq / 32, so non-aligned Kq drops K-elements at the end and
    // reads weights at the wrong K range per channel. K must be
    // divisible by Q*32 for grouped quantization.
    if (quantGroupNum > 1 && (K % (quantGroupNum * 32) != 0)) {
        return ge::GRAPH_FAILED;
    }

    // Bias support: only Q == 1 path is implemented. Q > 1 would require
    // per-channel bias accumulation in fp16 after channel VDEQ16s, which
    // differs from the L0C-bias-init pattern used for Q==1.
    if (hasBias != 0 && quantGroupNum > 1) {
        return ge::GRAPH_FAILED;
    }

    // K-tail support: K not divisible by K0=32 needs the kernel to zero
    // the partial last K0-group of activation in UB (V-pipe Duplicate(0))
    // before scatter to L1. Weight ZN is already zero-padded in the
    // K-tail by the caller's ND->ZN conversion. (Grouped quant already
    // requires K % (Q*32) == 0 above, which implies kTail == 0 for Q>1.)
    uint32_t kTail = K % 32U;

    // The partial K-fractal MTE2 in WeightMte2 reads the compact
    // [N1, N0, K_tail] GM section with a per-N0 stride of CUBE_N0 * kTail
    // bytes, which must land on a 32-byte block boundary. CUBE_N0 is 16, so
    // that holds only for even kTail; an odd kTail silently mis-strides the
    // weight reads. Reject it here rather than corrupt results.
    if (kTail % 2U != 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t K_padded = (kTail != 0) ? (K + (32U - kTail)) : K;

    // Reserve UB for the persistent ubBias_ buffer (N int32 + bank pad)
    // so the tile picker's feasibility check sees the reduced workspace
    // budget. Mirrors the layout in the kernel's Init.
    uint64_t ubSizeForPicker = ubSize;
    if (hasBias != 0) {
        uint64_t biasReserve =
            ((static_cast<uint64_t>(N) * sizeof(int32_t) + 31U) / 32U) * 32U +
            qbmv3::UB_BANK_PAD;
        if (biasReserve >= ubSizeForPicker) {
            return ge::GRAPH_FAILED;
        }
        ubSizeForPicker -= biasReserve;
    }

    // Picker sees K_padded so UB/L1/L0 sizing reserves space for the
    // zero-padded K-tail. Kernel internally uses K_actual (= K) for
    // MTE2 source bytes and Mmad K parameter; K_padded for buffer
    // strides and scatter destination sizes.
    Int8TilingResult t = ComputeInt8Tiling(
        M, N, K_padded, quantGroupNum, batch, aicNum,
        ubSizeForPicker, l0ASize,
        l0BSize, l0CSize,
        l1Size, isPerTensor, scaleType,
        hasPertoken);
    if (t.baseM == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t baseM = t.baseM;
    uint32_t baseN = t.baseN;
    uint32_t baseK = t.baseK;

    uint32_t totalMTiles = (M + baseM - 1) / baseM;
    uint32_t totalColBatch = (N + baseN - 1) / baseN;
    uint32_t totalWork = batch * totalMTiles * totalColBatch;

    const uint64_t knBytes = static_cast<uint64_t>(K) * static_cast<uint64_t>(N);

    // Grid partitioning: kernel owns (myVmRange * myCbRange) per core, where
    // virtualM = batch * totalMTiles. Folding batch into the M axis lets
    // PickCoreGrid distribute batches across cores when totalMTiles * cb alone
    // would leave cores idle (e.g. batch>1, mTiles=cb=1).
    uint32_t virtualM = batch * totalMTiles;
    const uint32_t gridCoalesce =
        ComputeScaleCoalesce(quantGroupNum, baseN, N, isPerTensor, scaleType);
    TilingCandidate gridTile{baseM, baseN, baseK, t.phaseXChunk,
                             t.wChunkKPasses, gridCoalesce};
    ProblemShape gridShape{};
    gridShape.B = batch;
    gridShape.M = M;
    gridShape.K = K;
    gridShape.N = N;
    gridShape.Q = quantGroupNum;
    gridShape.hasPertoken = (hasPertoken != 0);
    gridShape.isPerTensor = (isPerTensor != 0);
    gridShape.scaleType = scaleType;
    CoreGrid grid = PickCoreGrid(aicNum, virtualM, totalColBatch,
                                  gridTile, gridShape);
    uint32_t gridCores = grid.MCoreNum * grid.NCoreNum;
    uint32_t usedCoreNum = std::min(gridCores, std::max(1U, totalWork));

    QBMTilingData tilingData;
    tilingData.set_batch(batch);
    tilingData.set_M(M);
    tilingData.set_K(K);
    tilingData.set_N(N);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_hasPertoken(hasPertoken);
    tilingData.set_quantGroupNum(quantGroupNum);
    tilingData.set_maxOneTurnToken(baseM);
    tilingData.set_maxWeightColOneTurn(baseN);
    tilingData.set_baseK(baseK);
    tilingData.set_scaleType(scaleType);
    tilingData.set_isPerTensor(isPerTensor);
    tilingData.set_phaseXChunk(t.phaseXChunk);
    uint32_t pxChunk = t.phaseXChunk;
    uint32_t pxFullPairs = 0;
    uint32_t pxTail = K;
    if (pxChunk < K) {
        // Ping/pong loop followed by a tail when 2*pxChunk doesn't divide K.
        pxFullPairs = K / (2 * pxChunk);
        pxTail = K - pxFullPairs * 2 * pxChunk;
    }
    tilingData.set_phaseXFullPairs(pxFullPairs);
    tilingData.set_phaseXTail(pxTail);

    tilingData.set_MCoreNum(grid.MCoreNum);
    tilingData.set_NCoreNum(grid.NCoreNum);
    tilingData.set_wChunkKPasses(t.wChunkKPasses);
    const uint32_t scaleCoalesce =
        ComputeScaleCoalesce(quantGroupNum, baseN, N,
                             isPerTensor, scaleType);
    tilingData.set_scaleCoalesce(scaleCoalesce);

    // L2 cache hint mode for every GM tensor's SetL2CacheHint.
    // CacheMode enum: DISABLE=0, NORMAL=1, PERSISTENT=4.
    //   DISABLE: each core has at most one work item -- no temporal reuse,
    //            so polluting L2 with the weight wastes capacity.
    //   PERSISTENT: K*N fits in L2_HEADROOM AND there is reuse (cooperative-
    //               batch fires, or B==1 with multi-mTile dispatch). Locks
    //               the weight in L2 across iterations.
    //   NORMAL: default; weight too big to live in L2 fully, but normal LRU
    //           replacement is still beneficial.
    uint32_t l2HintMode = 1U;  // NORMAL
    const bool noReuse = (totalWork <= aicNum);
    const bool cooperativeBatch = (grid.NCoreNum == 1U && batch > 1U);
    const bool sharedWeightFits =
        (knBytes <= qbmv3::L2_HEADROOM_BYTES) &&
        (cooperativeBatch || batch == 1U);
    if (noReuse) {
        l2HintMode = 0U;  // DISABLE
    } else if (sharedWeightFits) {
        l2HintMode = 4U;  // PERSISTENT
    }
    tilingData.set_l2HintMode(l2HintMode);

    // L2 super-tile geometry is disabled: a single super-tile covers the
    // full work space, and the kernel's outer (mTSuperIdx, nTSuperIdx)
    // loops collapse to one iteration.
    tilingData.set_mTileCntL2(1U);
    tilingData.set_nTileCntL2(1U);
    tilingData.set_mTileBlock(totalMTiles);
    tilingData.set_nTileBlock(totalColBatch);

    // L0C ping-pong: when 2*baseM*baseN*4 fits L0C, double-buffer L0C so
    // the Mmad of cb_{i+1} on half b can overlap the VDEQ16 of cb_i on
    // half a. Falls back to single-buffer when the budget does not fit.
    const uint64_t l0cHalfBytes =
        static_cast<uint64_t>(baseM) * baseN * 4U;
    const uint32_t dbL0c =
        (2U * l0cHalfBytes <= l0CSize) ? 1U : 0U;
    tilingData.set_dbL0c(dbL0c);

    // Pertoken coalesce: when (B*M*4) fits a small UB budget, fire
    // ONE MTE2 at Process entry covering ALL B*M floats; per-mTile use
    // indexes into the persistent buffer. Saves ~422 cyc setup *
    // (waves-1) for small-row shapes. Falls back to per-mTile when
    // the buffer would not fit. 4 KB stays well under workspace.
    constexpr uint32_t PT_COALESCE_BUDGET_BYTES = 4096;
    const uint32_t pertokenCoalesce =
        (hasPertoken &&
         static_cast<uint64_t>(batch) * M * 4U <= PT_COALESCE_BUDGET_BYTES)
        ? 1U : 0U;
    tilingData.set_pertokenCoalesce(pertokenCoalesce);
    tilingData.set_hasBias(hasBias);
    tilingData.set_kTail(kTail);
    // Inner-tile Phase D is disabled: ubCalcM=0 routes the kernel to its
    // single-shot Phase D drain over the full mAligned range.
    tilingData.set_ubCalcM(0U);

    context->SetBlockDim(usedCoreNum);

    // 310P uses BASIC + NOT_PERTOKEN: unified vector dequant for all paths.
    context->SetTilingKey(GET_TPL_TILING_KEY(
        QUANT_BATCH_MATMUL_V3_X_B_TRANS,
        QUANT_BATCH_MATMUL_V3_X_KERNEL_TEMPLATE_TYPE_BASIC,
        QUANT_BATCH_MATMUL_V3_X_NOT_PERTOKEN,
        QUANT_BATCH_MATMUL_V3_X_OPTION_ATTR_NONE));

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) return ge::GRAPH_FAILED;
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForQBM(gert::TilingParseContext* context) {
    // The aclnn runtime executor does not invoke the TilingParse callback;
    // TilingQBM reads platform info directly from the TilingContext instead.
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantBatchMatmulV3X)
.Tiling(TilingQBM)
.TilingParse<QBMCompileInfo>(TilingPrepareForQBM);

}  // namespace optiling
