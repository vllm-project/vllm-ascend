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
 * \file tiling_cost_model.h
 * \brief Analytical per-pipe cycle cost model for tile selection on Ascend 310P -- declarations.
 * \author Feodor Pisnitchenko
 */
#ifndef QBMV3_TILING_COST_MODEL_H
#define QBMV3_TILING_COST_MODEL_H

#include <cstdint>

namespace optiling {

struct ProblemShape {
    uint32_t B;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t Q;
    bool     hasPertoken;
    bool     isPerTensor;
    uint32_t scaleType;   // 0=fp32, 2=uint64 (matches tiling.cpp)
};

struct TilingCandidate {
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t phaseXChunk;
    // Setting either coalesce field to 0 reverts the cost model to the
    // one-MTE2-per-kPass / one-MTE2-per-ch baseline cadence.
    uint32_t wChunkKPasses;   // kPasses fused per weight MTE2 (>= 1)
    uint32_t scaleCoalesce;   // 0 = no coalesce, 1 = coalesce active
};

struct PipeBreakdown {
    uint64_t mte2;
    uint64_t mte1;
    uint64_t mte3;
    uint64_t cube;
    uint64_t vector;
    uint64_t scalar;
};

// Per-pipe cycle totals for processing one mTile's worth of work.
// Covers Phase X (once), the full (colBatches * Q channels) inner loop,
// the cb-count scatter + CopyOut, and the sync-overhead estimate.
// cbsPerMTile = 0 is shorthand for ceil(N / baseN) (full N); pass a
// smaller value to model only the subset of cbs one core actually owns.
PipeBreakdown ComputePipeCyclesPerMTile(const TilingCandidate& tile,
                                         const ProblemShape& shape,
                                         uint32_t cbsPerMTile = 0);

// Per-core wall cycles for a given (mc * nc) grid partition. Uses an
// L2-aware blended MTE2 cost that accounts for cross-core reuse:
// M-siblings (nc cores with same mi) share the same activation slice;
// N-siblings (mc cores with same ni) share the same weight slice.
// Spill cap fires when per-wave first-touch exceeds L2_EFF_BYTES.
uint64_t EstimatePerCoreWall(const TilingCandidate& tile,
                              const ProblemShape& shape,
                              uint32_t mTilesPerCore,
                              uint32_t cbsPerCore,
                              uint32_t mc,
                              uint32_t nc);

// Upper-bound per-core wall-time estimate used at tile-ranking time
// before the (MCoreNum, NCoreNum) grid is chosen. Assumes M-sharded
// distribution (mc = min(aicNum, virtualM), nc = 1) -- the grid that
// maximises weight reuse via L2 in the blended model.
uint64_t EstimatePerCoreCriticalPath(const TilingCandidate& tile,
                                      const ProblemShape& shape,
                                      uint32_t aicNum);

// Per-core MTE2 wall cycles under the blended (Ab + Bb) bandwidth
// model for a given grid (mc, nc).
uint64_t EstimateMte2BlendedPerCore(const TilingCandidate& tile,
                                     const ProblemShape& shape,
                                     uint32_t mc,
                                     uint32_t nc,
                                     uint32_t mTilesPerCore,
                                     uint32_t cbsPerCore);

}  // namespace optiling

#endif  // QBMV3_TILING_COST_MODEL_H
