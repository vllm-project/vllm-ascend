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
 * \file tiling_cost_model.cpp
 * \brief Per-pipe cycle cost model for tile selection on Ascend 310P.
 * \author Feodor Pisnitchenko
 */
#include "tiling_cost_model.h"

#include <algorithm>
#include "../op_kernel/quant_batch_matmul_v3_x_config.h"
#include "l2_hit_predictor.h"

namespace optiling {

namespace {

constexpr uint32_t CUBE_M0 = 16;
constexpr uint32_t CUBE_N0 = 16;
constexpr uint32_t CUBE_K0 = 32;   // int8 K-fractal width

inline uint64_t DmaCycles(uint64_t bytes, uint32_t bpc, uint32_t setup) {
    return static_cast<uint64_t>(setup) + (bytes + bpc - 1) / bpc;
}

inline uint32_t AlignUpU(uint32_t val, uint32_t align) {
    return ((val + align - 1) / align) * align;
}

}  // namespace

PipeBreakdown ComputePipeCyclesPerMTile(const TilingCandidate& tile,
                                         const ProblemShape& shape,
                                         uint32_t cbsPerMTile)
{
    PipeBreakdown p{0, 0, 0, 0, 0, 0};

    const uint32_t Q = std::max(shape.Q, 1U);
    const uint32_t Kq = shape.K / Q;
    const uint32_t k_passes = (Kq + tile.baseK - 1) / tile.baseK;
    const uint32_t m_aligned = AlignUpU(tile.baseM, CUBE_M0);
    const uint32_t total_cb_full = (shape.N + tile.baseN - 1) / tile.baseN;
    const uint32_t total_cb = (cbsPerMTile == 0) ? total_cb_full : cbsPerMTile;
    const uint32_t inner_iters = total_cb * Q;
    const uint32_t ng_count = tile.baseN / CUBE_N0;
    const uint32_t phase_x_chunk = std::max(tile.phaseXChunk, 1U);

    // ----- Phase X (once per mTile) -----
    const uint32_t total_x_phases = shape.K / phase_x_chunk;
    const uint64_t x_mte2_bytes = static_cast<uint64_t>(tile.baseM) * phase_x_chunk;
    p.mte2 += static_cast<uint64_t>(total_x_phases) *
              DmaCycles(x_mte2_bytes, qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);

    const uint64_t zero_pad_cyc = (m_aligned > tile.baseM)
        ? std::max<uint64_t>(1,
              (static_cast<uint64_t>(m_aligned - tile.baseM) * phase_x_chunk) / 16)
        : 1;
    p.vector += static_cast<uint64_t>(total_x_phases) * zero_pad_cyc;

    const uint32_t x_mte3_turns = m_aligned / CUBE_M0;
    const uint32_t x_mte3_bytes_per_dma = (phase_x_chunk / CUBE_K0) * CUBE_K0;
    const uint64_t x_mte3_total_dmas =
        static_cast<uint64_t>(x_mte3_turns) * CUBE_M0 * total_x_phases;
    // MTE3_L1_BPC is 128 B/cyc (UB->L1) -- not currently in qbmv3::; inline.
    constexpr uint32_t MTE3_L1_BPC = 128;
    p.mte3 += x_mte3_total_dmas *
              (qbmv3::MTE3_SETUP_CYC + x_mte3_bytes_per_dma / MTE3_L1_BPC);

    if (shape.hasPertoken) {
        p.mte2 += DmaCycles(static_cast<uint64_t>(tile.baseM) * 4,
                            qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
    }

    // ----- Inner (per channel * per cb) ops -----
    // Weight MTE2 cadence = cb * Q * ceil(k_passes / wChunkKPasses).
    // Total bytes unchanged (Kq * baseN per ch); only setup count changes.
    // wChunkKPasses == 0 reverts to one MTE2 per kPass (single DMA of
    // Kq * baseN per ch).
    const uint32_t w_chunk_kp = (tile.wChunkKPasses == 0) ? k_passes
                                                          : tile.wChunkKPasses;
    const uint32_t w_chunks_per_iter =
        (k_passes + w_chunk_kp - 1) / std::max(w_chunk_kp, 1U);
    const uint64_t w_bytes_per_dma =
        (w_chunks_per_iter == 0) ? static_cast<uint64_t>(Kq) * tile.baseN
                                 : (static_cast<uint64_t>(Kq) * tile.baseN /
                                    w_chunks_per_iter);
    p.mte2 += static_cast<uint64_t>(inner_iters) * w_chunks_per_iter *
              DmaCycles(w_bytes_per_dma, qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);

    // Scale MTE2 cadence follows kernel ComputeScaleCoalesce:
    //   pertensor           -> 0 MTE2 (V-pipe fills once per mTile)
    //   Q == 1 coalesce     -> 1 MTE2 per mTile, N * 8 bytes
    //   Q  > 1 coalesce     -> cb MTE2s per mTile, Q * baseN * 8 bytes each
    //   else (non-coalesce) -> cb * Q MTE2s, baseN * 8 bytes each
    if (!shape.isPerTensor) {
        if (tile.scaleCoalesce != 0 && Q <= 1) {
            const uint64_t scale_bytes_mtile = static_cast<uint64_t>(shape.N) * 8;
            p.mte2 += DmaCycles(scale_bytes_mtile,
                                qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        } else if (tile.scaleCoalesce != 0 && Q > 1) {
            const uint64_t scale_bytes_cb =
                static_cast<uint64_t>(Q) * tile.baseN * 8;
            p.mte2 += static_cast<uint64_t>(total_cb) *
                      DmaCycles(scale_bytes_cb,
                                qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        } else {
            const uint64_t scale_bytes = static_cast<uint64_t>(tile.baseN) * 8;
            p.mte2 += static_cast<uint64_t>(inner_iters) *
                      DmaCycles(scale_bytes,
                                qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        }
    }

    const uint64_t l0a_bytes = static_cast<uint64_t>(m_aligned) * tile.baseK;
    const uint64_t l0b_bytes =
        static_cast<uint64_t>(tile.baseK / CUBE_K0) * ng_count * CUBE_K0 * CUBE_N0;
    p.mte1 += static_cast<uint64_t>(inner_iters) * k_passes *
              DmaCycles(l0a_bytes, qbmv3::MTE1_L0A_BW_BPC, qbmv3::MTE1_SETUP_CYC);
    p.mte1 += static_cast<uint64_t>(inner_iters) * k_passes *
              DmaCycles(l0b_bytes, qbmv3::MTE1_L0B_BW_BPC, qbmv3::MTE1_SETUP_CYC);

    const uint64_t macs_per_mmad =
        static_cast<uint64_t>(m_aligned) * tile.baseN * tile.baseK;
    const uint64_t mmad_cyc =
        (macs_per_mmad + qbmv3::MMAD_INT8_MACS_PCYC - 1) / qbmv3::MMAD_INT8_MACS_PCYC;
    p.cube += static_cast<uint64_t>(inner_iters) * k_passes * mmad_cyc;

    const uint64_t vdeq_elems = static_cast<uint64_t>(m_aligned) * tile.baseN;
    const uint64_t vdeq_cyc = std::max<uint64_t>(16, vdeq_elems / 16 + 8);
    p.vector += static_cast<uint64_t>(inner_iters) * vdeq_cyc;
    if (Q > 1) {
        const uint64_t channel_accum_cyc = std::max<uint64_t>(1, vdeq_elems / 64);
        p.vector += static_cast<uint64_t>(total_cb) * (Q - 1) * channel_accum_cyc;
    }

    // ----- Scatter + CopyOut (once per cb) -----
    const uint64_t scatter_cyc =
        static_cast<uint64_t>(tile.baseM) * (4 + ng_count);
    p.vector += static_cast<uint64_t>(total_cb) * scatter_cyc;
    const uint64_t co_bytes = static_cast<uint64_t>(tile.baseM) * tile.baseN * 2;
    p.mte3 += static_cast<uint64_t>(total_cb) *
              DmaCycles(co_bytes, qbmv3::MTE3_BW_BPC, qbmv3::MTE3_SETUP_CYC);

    // ----- Sync overhead -----
    const uint32_t num_syncs_phase_x = 4 * total_x_phases + 4;
    const uint32_t num_syncs_channel = 7 * inner_iters;
    const uint32_t num_syncs_colbatch = 3 * total_cb;
    const uint32_t total_syncs =
        num_syncs_phase_x + num_syncs_channel + num_syncs_colbatch;
    p.scalar += static_cast<uint64_t>(total_syncs) * qbmv3::SYNC_PER_PAIR_CYC;

    return p;
}

uint64_t EstimateMte2BlendedPerCore(const TilingCandidate& tile,
                                     const ProblemShape& shape,
                                     uint32_t mc,
                                     uint32_t nc,
                                     uint32_t mTilesPerCore,
                                     uint32_t cbsPerCore)
{
    if (mc == 0 || nc == 0 || mTilesPerCore == 0 || cbsPerCore == 0) {
        return 0;
    }

    // Per-wave byte budget under the harmonic-mean MTE2 bandwidth model
    // for a 14 MB shared L2 with an empirical 70 % weight share.
    // Activation byte count uses min(M, baseM) so a small-M shape with
    // M < baseM does not over-count: only M real rows are MTE2-loaded,
    // not the M_aligned rows the tile reserves.
    const uint32_t mEffective = (shape.M < tile.baseM) ? shape.M : tile.baseM;
    const uint64_t Ab = static_cast<uint64_t>(mEffective) * shape.K;
    const uint64_t Bb = static_cast<uint64_t>(shape.K) * tile.baseN;
    const uint64_t mcnc = static_cast<uint64_t>(mc) * nc;

    // Cross-wave L2 reuse: the kernel iterates vmIdx-outer, cb-inner.
    // Same cb at different vmIdxs touches the SAME weight chunk; if
    // the weight fits in L2, every subsequent visit is an L2 hit.
    // Symmetric on activation: same vmIdx across different cbs reuses
    // the activation slice. Approximate the effect by reducing the
    // per-wave A/B byte terms by the in-core temporal reuse factor.
    uint64_t Ab_eff = Ab;
    uint64_t Bb_eff = Bb;
    const uint64_t kn_bytes =
        static_cast<uint64_t>(shape.K) * shape.N;
    const uint64_t mk_bytes =
        static_cast<uint64_t>(tile.baseM) * shape.K;
    if (mTilesPerCore > 1U && kn_bytes <= qbmv3::L2_HEADROOM_BYTES) {
        Bb_eff = Bb / mTilesPerCore;
    }
    if (cbsPerCore > 1U && mk_bytes <= qbmv3::L2_HEADROOM_BYTES) {
        Ab_eff = Ab / cbsPerCore;
    }

    const uint64_t gm_first_touch = static_cast<uint64_t>(mc) * Ab_eff +
                                     static_cast<uint64_t>(nc) * Bb_eff;
    // M-siblings (nc cores per M-row-group) re-read the same A slice;
    // N-siblings (mc cores per N-column-group) re-read the same B slice.
    const uint64_t l2_reuse = (mcnc - mc) * Ab_eff + (mcnc - nc) * Bb_eff;

    // Linear spill cap: fraction of first-touch bytes that cannot fit
    // L2_EFF_BYTES falls back to GM, dragging the matching share of
    // would-be L2 hits with it. Scaled by 1024 to stay in integer math.
    constexpr uint64_t SCALE = 1024;
    uint64_t spill = 0;
    if (gm_first_touch > qbmv3::L2_EFF_BYTES) {
        spill = ((gm_first_touch - qbmv3::L2_EFF_BYTES) * SCALE) /
                gm_first_touch;
    }

    const uint64_t gm_bytes = gm_first_touch +
                              (spill * l2_reuse) / SCALE;
    const uint64_t l2_bytes = ((SCALE - spill) * l2_reuse) / SCALE;

    const uint32_t gm_bpc = std::max(qbmv3::MTE2_BW_BPC, 1U);
    const uint32_t l2_bpc = std::max(qbmv3::L2_BW_BPC, 1U);
    const uint64_t t_gm = gm_bytes / gm_bpc;
    const uint64_t t_l2 = l2_bytes / l2_bpc;
    // 2 DMAs per core per wave (1 A, 1 B). Aggregate across cores.
    const uint64_t num_dmas = mcnc * 2;
    const uint64_t t_setup = num_dmas * qbmv3::MTE2_SETUP_CYC;

    const uint64_t t_wave_agg = t_gm + t_l2 + t_setup;
    const uint64_t per_core_wave = t_wave_agg / mcnc;
    const uint64_t waves_per_core = static_cast<uint64_t>(mTilesPerCore) *
                                     cbsPerCore;
    uint64_t mte2_total = per_core_wave * waves_per_core;

    // L2-hit-rate predictor as a miss-rate amplifier. When the predicted
    // L2 hit % is below the gate, multiply mte2_total by
    // (1 + (gate - hit) * slope), capped. Bad-L2-hit cells look
    // MTE2-slow even when nominally cube-bound, breaking ties without
    // affecting good-L2-hit cells.
    const uint64_t actPerCore =
        static_cast<uint64_t>(mTilesPerCore) * Ab;
    const uint64_t wPerCore =
        static_cast<uint64_t>(cbsPerCore) * Bb;
    const uint64_t perCoreWS = actPerCore + wPerCore;
    const double hitPct = PredictL2HitPct(
        tile.baseM, tile.baseN, tile.baseK,
        shape.B, shape.M, shape.K, shape.N,
        mTilesPerCore, cbsPerCore,
        actPerCore, wPerCore, perCoreWS);
    // miss_factor curve: 1.0 at hit >= 80 %, rising linearly with
    // slope 0.04 toward higher values at low hit. Cap = 2.7.
    if (hitPct < 80.0) {
        double miss_factor = 1.0 + (80.0 - hitPct) * 0.04;
        if (miss_factor > 2.7) miss_factor = 2.7;
        mte2_total = static_cast<uint64_t>(
            static_cast<double>(mte2_total) * miss_factor);
    }
    return mte2_total;
}

uint64_t EstimatePerCoreWall(const TilingCandidate& tile,
                              const ProblemShape& shape,
                              uint32_t mTilesPerCore,
                              uint32_t cbsPerCore,
                              uint32_t mc,
                              uint32_t nc)
{
    if (mTilesPerCore == 0 || cbsPerCore == 0) return 0;
    if (mc == 0) mc = 1;
    if (nc == 0) nc = 1;

    // Non-MTE2 pipes: per-mTile totals * mTilesPerCore.
    const PipeBreakdown p = ComputePipeCyclesPerMTile(tile, shape, cbsPerCore);
    const uint64_t non_mte2_per_mtile = std::max({p.mte1, p.mte3,
                                                   p.cube, p.vector, p.scalar});
    const uint64_t non_mte2_total =
        non_mte2_per_mtile * mTilesPerCore;

    // MTE2 under the L2-aware blended model.
    const uint64_t blended_mte2 = EstimateMte2BlendedPerCore(
        tile, shape, mc, nc, mTilesPerCore, cbsPerCore);

    // Scale + pertoken MTE2 are small and not covered by the Ab/Bb blend.
    // Charge them as straight per-core counts (no L2 sharing bonus).
    const uint32_t Q = std::max(shape.Q, 1U);
    uint64_t aux_mte2 = 0;
    if (!shape.isPerTensor) {
        if (tile.scaleCoalesce != 0 && Q <= 1) {
            aux_mte2 += static_cast<uint64_t>(mTilesPerCore) *
                DmaCycles(static_cast<uint64_t>(shape.N) * 8,
                          qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        } else if (tile.scaleCoalesce != 0 && Q > 1) {
            aux_mte2 += static_cast<uint64_t>(mTilesPerCore) * cbsPerCore *
                DmaCycles(static_cast<uint64_t>(Q) * tile.baseN * 8,
                          qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        } else {
            aux_mte2 += static_cast<uint64_t>(mTilesPerCore) * cbsPerCore * Q *
                DmaCycles(static_cast<uint64_t>(tile.baseN) * 8,
                          qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
        }
    }
    if (shape.hasPertoken) {
        aux_mte2 += static_cast<uint64_t>(mTilesPerCore) *
            DmaCycles(static_cast<uint64_t>(tile.baseM) * 4,
                      qbmv3::MTE2_BW_BPC, qbmv3::MTE2_SETUP_CYC);
    }

    const uint64_t mte2_total = blended_mte2 + aux_mte2;
    return std::max(mte2_total, non_mte2_total);
}

uint64_t EstimatePerCoreCriticalPath(const TilingCandidate& tile,
                                      const ProblemShape& shape,
                                      uint32_t aicNum)
{
    const uint32_t mTiles = (shape.M + tile.baseM - 1) / tile.baseM;
    const uint32_t virtualM = shape.B * mTiles;
    const uint32_t cores = std::max(aicNum, 1U);
    const uint32_t total_cb = (shape.N + tile.baseN - 1) / tile.baseN;
    // Assume M-sharded (mc = min(cores, virtualM), nc = 1) -- the grid
    // that maximises weight reuse via L2 for big-K*N shapes. Callers
    // pick the final grid with PickCoreGrid, which re-evaluates every
    // factor pair via EstimatePerCoreWall.
    const uint32_t mc = std::max(1U, std::min(cores, virtualM));
    const uint32_t nc = 1U;
    const uint32_t mTiles_per_core = (virtualM + mc - 1) / mc;
    return EstimatePerCoreWall(tile, shape, mTiles_per_core, total_cb, mc, nc);
}

}  // namespace optiling
