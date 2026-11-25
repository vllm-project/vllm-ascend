// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_PARALLEL_PLAN_H
#define ATTN_RES_FWD_PARALLEL_PLAN_H

#include <algorithm>
#include <cstdint>

#include "../attn_res_fwd_validation.h"
#include "../op_kernel/attn_res_fwd_ub_layout.h"

namespace AttnResFwd {

constexpr uint32_t COOPERATIVE_CORE_GAIN = 2;
constexpr uint32_t COOPERATIVE_MIN_HIDDEN = 512;
// Above quarter-device token occupancy, use uniform multi-worker token groups.
// Lower occupancy may use global row scheduling to shorten the scoring waves.
constexpr uint32_t COOPERATIVE_LOW_TOKEN_OCCUPANCY = 4;
constexpr uint32_t COOPERATIVE_MAX_OUTPUT_SPLITS = 8;
constexpr uint32_t COOPERATIVE_MAX_SCORE_SPLITS = 32;
constexpr uint32_t COOPERATIVE_MAX_ROWS = 128;
constexpr uint32_t COOPERATIVE_HIDDEN_ALIGN = TILING_HIDDEN_ALIGNMENT;
constexpr uint32_t COOPERATIVE_SLOT_BYTES = 32;
constexpr uint32_t COOPERATIVE_FLOAT_BLOCK = COOPERATIVE_SLOT_BYTES / sizeof(float);
constexpr uint32_t COOPERATIVE_UB_ALLOWANCE = 32 * 1024;
// Whole H: two inputs, one reserved output queue and one reusable BF16 row.
// Split H: one input, one output queue and one reusable BF16 row.
constexpr uint32_t COOPERATIVE_WHOLE_B16_ROWS = 4;
constexpr uint32_t COOPERATIVE_SPLIT_B16_ROWS = 3;
constexpr uint32_t COOPERATIVE_FP32_ROWS = 3;  // score weight, current row, accumulator.
constexpr uint32_t COOPERATIVE_SCALAR_SLOTS = 2;

struct ParallelPlan {
  uint32_t usedCores{0};
  uint32_t outputSplits{0};
  uint32_t scoreSplits{0};
  uint32_t tileElems{0};
  uint64_t workspaceBytes{0};
  uint32_t residentGroup{0};  // Reserved for existing tiling/probe layout; always zero.
  uint32_t rowParallel{0};
  uint32_t tokenGroupSize{0};
  uint32_t cachedRows{0};  // Additional BF16 rows retained by the output owner.
};

// Cooperative kernels are reserved for work shared by multiple score workers.
// A one-worker token group belongs to the legacy token-parallel kernels.
inline bool ShouldUseCooperativeKernel(const ParallelPlan& plan) {
  return plan.usedCores != 0 && plan.tokenGroupSize != 1;
}

inline uint64_t RowParallelUbBytes(uint32_t rows, uint64_t tile, bool regBase = false) {
  const uint64_t metaAlign =
      (static_cast<uint64_t>(rows) + COOPERATIVE_FLOAT_BLOCK - 1) / COOPERATIVE_FLOAT_BLOCK * COOPERATIVE_FLOAT_BLOCK;
  const uint64_t metaElems =
      regBase ? (metaAlign + COOPERATIVE_HIDDEN_ALIGN - 1) / COOPERATIVE_HIDDEN_ALIGN * COOPERATIVE_HIDDEN_ALIGN
              : std::max<uint64_t>(metaAlign, COOPERATIVE_HIDDEN_ALIGN);
  const uint64_t slots = std::max<uint64_t>(metaAlign, rows);
  const uint32_t rowPadding =
      !regBase && tile * sizeof(float) % COOPERATIVE_UB_ADDRESS_PERIOD_BYTES == 0 ? COOPERATIVE_ROW_PADDING_BYTES : 0;
  return tile * (COOPERATIVE_WHOLE_B16_ROWS * sizeof(uint16_t) + COOPERATIVE_FP32_ROWS * sizeof(float)) +
         (2 * metaElems + metaAlign * COOPERATIVE_FLOAT_BLOCK + slots * COOPERATIVE_FLOAT_BLOCK +
          COOPERATIVE_SCALAR_SLOTS * COOPERATIVE_FLOAT_BLOCK) *
             sizeof(float) +
         COOPERATIVE_SLOT_BYTES + COOPERATIVE_UB_ALLOWANCE + rowPadding;
}

// A5 emits the reusable BF16 row directly, so token groups need no output queue.
inline uint64_t TokenGroupUbBytes(uint32_t rows, uint64_t tile, bool regBase) {
  if (!regBase) {
    return RowParallelUbBytes(rows, tile, regBase);
  }
  return RowParallelUbBytes(rows, tile, regBase) - tile * sizeof(uint16_t);
}

// Only choose cooperation when token parallelism uses at most half the AIVs.
// Prefer whole-H output ownership when the full-row buffer budget fits UB.
inline ParallelPlan MakeParallelPlan(uint32_t tokens, uint32_t rows, uint32_t hidden, uint32_t cores, uint64_t ubBytes,
                                     bool outputNorm, bool supportsCooperative, bool regBase = false) {
  ParallelPlan plan;
  if (!supportsCooperative || tokens == 0 || rows < 2 || rows > COOPERATIVE_MAX_ROWS ||
      hidden < COOPERATIVE_MIN_HIDDEN || tokens > cores / COOPERATIVE_CORE_GAIN) {
    return plan;
  }
  const uint64_t rowTasks = static_cast<uint64_t>(tokens) * rows;
  const uint64_t wholeTile = (static_cast<uint64_t>(hidden) + COOPERATIVE_HIDDEN_ALIGN - 1) / COOPERATIVE_HIDDEN_ALIGN *
                             COOPERATIVE_HIDDEN_ALIGN;
  if (RowParallelUbBytes(rows, wholeTile, regBase) <= ubBytes) {
    const uint32_t maxGroup = std::min(rows, cores / tokens);
    const uint32_t groupWaves = (rows + maxGroup - 1) / maxGroup;
    // Use the smallest uniform group with the same critical score-row count.
    const uint32_t group = (rows + groupWaves - 1) / groupWaves;
    const uint64_t globalCores = std::min<uint64_t>(cores, rowTasks);
    const uint64_t globalWaves = (rowTasks + cores - 1) / cores;
    const bool occupied = tokens > cores / COOPERATIVE_LOW_TOKEN_OCCUPANCY;
    // Low-occupancy groups must save worker loads without adding a score wave.
    if (occupied || (groupWaves <= globalWaves && tokens * group < globalCores)) {
      plan.tokenGroupSize = group;
      const uint32_t ownedRows = (rows + group - 1) / group;
      const uint64_t spare = ubBytes - TokenGroupUbBytes(rows, wholeTile, regBase);
      plan.cachedRows =
          static_cast<uint32_t>(std::min<uint64_t>(ownedRows - 1, spare / (wholeTile * sizeof(uint16_t))));
    }
    plan.usedCores = plan.tokenGroupSize != 0 ? tokens * plan.tokenGroupSize : static_cast<uint32_t>(globalCores);
    plan.scoreSplits = 1;
    plan.outputSplits = 1;
    plan.tileElems = static_cast<uint32_t>(wholeTile);
    plan.workspaceBytes = rowTasks * COOPERATIVE_SLOT_BYTES;
    plan.rowParallel = 1;
    return plan;
  }
  // UB capacity, rather than a fixed H threshold, selects the fallback.
  // Slice tasks may outnumber cores and are scheduled with a strided loop.
  const uint32_t maxScoreSplits = std::min(COOPERATIVE_MAX_SCORE_SPLITS, hidden / COOPERATIVE_MIN_HIDDEN);
  const uint32_t firstScoreSplits =
      static_cast<uint32_t>(std::min<uint64_t>(maxScoreSplits, std::max<uint64_t>(2, cores / rowTasks)));
  const uint32_t outputSplits =
      std::min({COOPERATIVE_MAX_OUTPUT_SPLITS, cores / tokens, hidden / COOPERATIVE_MIN_HIDDEN});
  const uint64_t units = (static_cast<uint64_t>(hidden) + COOPERATIVE_HIDDEN_ALIGN - 1) / COOPERATIVE_HIDDEN_ALIGN;
  for (uint32_t scoreSplits = firstScoreSplits; scoreSplits <= maxScoreSplits; ++scoreSplits) {
    const uint64_t tileElems =
        std::max((units + scoreSplits - 1) / scoreSplits, (units + outputSplits - 1) / outputSplits) *
        COOPERATIVE_HIDDEN_ALIGN;
    const uint64_t metaAlign =
        (static_cast<uint64_t>(rows) + COOPERATIVE_FLOAT_BLOCK - 1) / COOPERATIVE_FLOAT_BLOCK * COOPERATIVE_FLOAT_BLOCK;
    // Three BF16 rows (input queue, output queue, reusable row) and three FP32
    // rows (score weight, current row, accumulator); metadata includes Brcb's
    // full eight-block destination, even when only its first block is used.
    const uint64_t metaElems =
        regBase ? (metaAlign + COOPERATIVE_HIDDEN_ALIGN - 1) / COOPERATIVE_HIDDEN_ALIGN * COOPERATIVE_HIDDEN_ALIGN
                : std::max<uint64_t>(metaAlign, COOPERATIVE_HIDDEN_ALIGN);
    const uint64_t slotCount =
        std::max({metaAlign, static_cast<uint64_t>(rows) * scoreSplits, static_cast<uint64_t>(outputSplits)});
    const uint64_t requiredUb =
        tileElems * (COOPERATIVE_SPLIT_B16_ROWS * sizeof(uint16_t) + COOPERATIVE_FP32_ROWS * sizeof(float)) +
        (2 * metaElems + metaAlign * COOPERATIVE_FLOAT_BLOCK + slotCount * COOPERATIVE_FLOAT_BLOCK +
         COOPERATIVE_SCALAR_SLOTS * COOPERATIVE_FLOAT_BLOCK) *
            sizeof(float) +
        COOPERATIVE_SLOT_BYTES + COOPERATIVE_UB_ALLOWANCE;
    if (requiredUb > ubBytes) {
      continue;
    }
    plan.outputSplits = outputSplits;
    plan.scoreSplits = scoreSplits;
    plan.tileElems = static_cast<uint32_t>(tileElems);
    const uint64_t scoreTasks = rowTasks * scoreSplits;
    const uint32_t outputTasks = tokens * plan.outputSplits;
    plan.usedCores = std::max(static_cast<uint32_t>(std::min<uint64_t>(cores, scoreTasks)), outputTasks);
    // RMS reads and score writes can overlap across different cores. Use
    // separate regions, so a score cannot overwrite another core's RMS input.
    plan.workspaceBytes =
        (scoreTasks * (scoreSplits > 1 ? 2 : 1) + (outputNorm ? outputTasks : 0)) * COOPERATIVE_SLOT_BYTES;
    return plan;
  }
  return plan;
}

}  // namespace AttnResFwd
#endif
