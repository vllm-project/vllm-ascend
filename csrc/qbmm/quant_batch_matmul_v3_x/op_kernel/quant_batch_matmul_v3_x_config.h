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
 * \file quant_batch_matmul_v3_x_config.h
 * \brief Compile-time hardware constants (UB / L1 / L0 sizes, pipe bandwidths) shared by host tiling and kernel.
 * \author Feodor Pisnitchenko
 *
 * No __aicore__ content -- safe to include from both host tiling and
 * kernel code.
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_CONFIG_H
#define QUANT_BATCH_MATMUL_V3_X_CONFIG_H

#include <cstdint>

namespace qbmv3 {

// Padding between adjacent persistent UB sub-buffers. A non-zero value
// avoids bank conflicts on 310P's 32-bank UB.
constexpr uint32_t UB_BANK_PAD = 4096;

// Candidate Nb tile sizes enumerated by the tiling selector (ordered largest
// -> smallest so the first feasible candidate wins). All entries are multiples
// of 16 to align with Mmad's 16*16 micro-tile.
constexpr uint32_t NB_CANDIDATES[] = {320, 256, 192, 128, 96, 64, 32, 16};

// Kb candidates. Divisors yield `Kb = K / d`, which produces an integer
// kPasses for K in powers of two (common case). The safety-net absolutes
// cover shapes where no divisor lands on a useful 32-aligned value.
constexpr uint32_t KB_DIVISORS[] = {1, 2, 4, 8, 16, 32, 64};
constexpr uint32_t KB_SAFETY[]   = {64, 128, 256, 512, 1024, 2048};

static_assert(NB_CANDIDATES[0] % 16 == 0, "Nb candidates must be multiples of 16");

// 310P memory hierarchy, from Ascend310P3.ini. Platform-reported sizes at
// runtime override these; the constants exist as fallbacks and for
// compile-time asserts.
constexpr uint32_t L0A_TOTAL_BYTES = 64 * 1024;
constexpr uint32_t L0B_TOTAL_BYTES = 64 * 1024;
constexpr uint32_t L0C_TOTAL_BYTES = 256 * 1024;
constexpr uint32_t L1_TOTAL_BYTES  = 1024 * 1024;
constexpr uint32_t UB_TOTAL_BYTES  = 256 * 1024;
constexpr uint32_t L2_TOTAL_BYTES  = 16 * 1024 * 1024;

// Ping/pong halves used when L0 double-buffering is active.
constexpr uint32_t L0A_HALF_BYTES = L0A_TOTAL_BYTES / 2;
constexpr uint32_t L0B_HALF_BYTES = L0B_TOTAL_BYTES / 2;

// L2 effective budget for weight residency (14 MB leaves room for X1,
// scale, output, VDEQ state, metadata). 70% of that is the share that
// the MTE2 blended cost model treats as available for weight in its
// first-touch spill cap.
constexpr uint32_t L2_HEADROOM_BYTES   = 14 * 1024 * 1024;
constexpr uint32_t L2_WEIGHT_SHARE_PCT = 70;
constexpr uint64_t L2_EFF_BYTES =
    static_cast<uint64_t>(L2_HEADROOM_BYTES) * L2_WEIGHT_SHARE_PCT / 100;

// L2 super-tile budget: the A + B + C stripes of one super-tile
// (mb*baseM*K + nb*K*baseN + mb*nb*baseM*baseN*2 bytes) must fit
// within this fraction of L2_TOTAL. 52 % is empirical -- tighter
// caps under-utilise L2, looser caps cause cross-super-tile spill.
constexpr uint32_t L2_SPLIT_RATIO_PCT  = 52;

// 310P pipe throughput + setup latencies (Ascend310P3.ini). Bandwidth is
// per-core bytes per cycle; setup is per-DMA cycles.
constexpr uint32_t MTE2_BW_BPC       = 17;     // GM -> UB / L1
constexpr uint32_t MTE3_BW_BPC       = 17;     // UB -> GM
constexpr uint32_t MTE1_L0A_BW_BPC   = 1024;   // L1 -> L0A
constexpr uint32_t MTE1_L0B_BW_BPC   = 256;    // L1 -> L0B
constexpr uint32_t L0C_UB_BW_BPC     = 512;    // L0C -> UB (VDEQ16)
constexpr uint32_t L2_BW_BPC         = 114;    // GM -> L1/UB when L2 hit
constexpr uint32_t MTE2_SETUP_CYC    = 422;    // per-DMA setup
constexpr uint32_t MTE1_SETUP_CYC    = 31;
constexpr uint32_t MTE3_SETUP_CYC    = 15;
constexpr uint32_t MMAD_INT8_MACS_PCYC = 4096; // cube throughput
constexpr uint32_t V_FP16_OPS_PCYC     = 128;  // vector throughput (fp16)
constexpr uint32_t VDEQ16_STARTUP_CYC  = 8;    // VDEQ16 pipeline startup
constexpr uint32_t SYNC_PER_PAIR_CYC   = 20;   // SetFlag + WaitFlag pair
constexpr uint32_t CLOCK_MHZ           = 1080;

// Mmad K-axis fractal: 32-byte stride for int8 (16 rows * 32 cols).
constexpr uint32_t K_FRACTAL_INT8_BYTES = 32;

// Hardware cap on a single Mmad's inner-K parameter (register range
// [0, 4095] aligned down to K_FRACTAL_INT8 = 32 = 4064). K > MMAD_K_MAX
// requires K-splitting across multiple Mmad issues.
constexpr uint32_t MMAD_K_MAX = 4064;

// VDEQ16 scale encoding marker: upper 32 bits of the uint64 scale are this
// constant; lower 32 bits are the float32 bit pattern.
constexpr uint32_t VDEQ16_MARKER = 0x00004000;

// Blocked-ZN outer-N block size: 4080 fractals * 16 = 65280 elements.
// Chosen so MTE2 srcStride fits in 16 bits when N > 65280 -- without
// blocking the stride overflows.
constexpr uint32_t BLOCKED_ZN_NG      = 4080;
constexpr uint32_t BLOCKED_ZN_N_ELEMS = BLOCKED_ZN_NG * 16;

}  // namespace qbmv3

#endif  // QUANT_BATCH_MATMUL_V3_X_CONFIG_H
