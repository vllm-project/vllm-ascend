/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating_tiling_utils.h
 * \brief rowsPerIter and Bulk DMA helper functions.
 */

#ifndef FUSED_GDN_GATING_TILING_UTILS_H
#define FUSED_GDN_GATING_TILING_UTILS_H

#include <cstdint>

namespace FusedGdnGating {

// NPU hardware constants.
constexpr uint32_t VECTOR_BYTES_PER_ITER = 256;
constexpr uint32_t DATACOPY_MIN_BYTES = 32;
constexpr uint32_t BF16_PER_BLOCK = DATACOPY_MIN_BYTES / 2;  // 16
constexpr uint32_t MASK_ALIGN_ELEMS = 64;

/// Align count to vector unit width (256 bytes) for given dtype size.
inline uint32_t AlignCountToVectorBytes(uint32_t count, uint32_t dtypeSize)
{
    uint32_t elemsPerIter = VECTOR_BYTES_PER_ITER / dtypeSize;
    return ((count + elemsPerIter - 1) / elemsPerIter) * elemsPerIter;
}

/// Check if Bulk DMA is viable: (R * nh) % 64 == 0, nh % 16 == 0.
inline bool CanUseBulkDma(uint32_t numHeads, uint32_t rowsPerIter)
{
    // Condition 1: (rows_per_iter * num_heads) % 64 == 0
    //   fp32 vector unit processes 64 elements per repeat; the bulk operation
    //   must align with this granularity to avoid tail handling.
    constexpr uint32_t fp32VecElems = VECTOR_BYTES_PER_ITER / 4;
    if ((rowsPerIter * numHeads) % fp32VecElems != 0) {
        return false;
    }

    // Condition 2: num_heads % 16 == 0
    //   DMA minimum transfer size is 32 bytes; for bf16/fp16 (2 bytes per element),
    //   this equals 16 elements. If num_heads is not a multiple of 16, the last
    //   few elements of each row require separate handling, negating the bulk benefit.
    constexpr uint32_t bf16BlockElems = DATACOPY_MIN_BYTES / 2;
    if (numHeads % bf16BlockElems != 0) {
        return false;
    }

    return true;
}

/*!
 * \brief Compute optimal rows_per_iter from UB budget.
 *
 * UB breakdown matches kernel Init(): 3 single-row fp32 constants
 * + 2 multi-row fp32 constants (R * ubDim * 4 each)
 * + 3 half + 6 fp32 per-row buffers (scaled by R).
 * ubDim = ceil(numHeads / 16) * 16 (matching kernel DMA_ALIGN_ELEMS).
 * Result clamped to power-of-2, max 128.
 */
inline uint32_t ComputeRowsPerIter(uint32_t numHeads, uint64_t ubBudget,
                                   uint32_t ubDim = 0)
{
    if (ubDim == 0) {
        // Match the kernel's fp32 compute/mask alignment.
        ubDim = ((numHeads + MASK_ALIGN_ELEMS - 1) / MASK_ALIGN_ELEMS) * MASK_ALIGN_ELEMS;
    }
    uint32_t maskUbDim = ubDim;

    // 2 parameter input queues + 2 fp32 constant buffers, each 1 row.
    // Use fp32 for the parameter queues as a conservative upper bound.
    uint32_t sharedBytes = 4 * ubDim * static_cast<uint32_t>(sizeof(float));

    // Multi-row constant buffers (precomputed once, scaled by R):
    //   dtBiasMultiBuf_ + negExpMultiBuf_: 2 fp32 buffers.
    uint32_t constPerRowBytes = 2 * ubDim * static_cast<uint32_t>(sizeof(float));

    // Per-row (per-chunk): 3 bf16/fp16 buffers + 5 fp32 buffers + 1 uint8 mask buffer.
    uint32_t perRowBytes = 3 * ubDim * static_cast<uint32_t>(sizeof(int16_t))   // a, b, betaOut
                         + 5 * ubDim * static_cast<uint32_t>(sizeof(float))     // g, x, betaX, tmp, betaFp32
                         + 1 * maskUbDim * static_cast<uint32_t>(sizeof(uint8_t)); // threshold mask

    if (perRowBytes == 0) {
        return 1;
    }

    uint32_t maxRows = 1;
    if (ubBudget > sharedBytes) {
        maxRows = static_cast<uint32_t>((ubBudget - sharedBytes) / (perRowBytes + constPerRowBytes));
    }

    // Round down to nearest power of 2 (128, 64, 32, ..., 1).
    if (maxRows >= 128) { return 128; }
    if (maxRows >= 64)  { return 64;  }
    if (maxRows >= 32)  { return 32;  }
    if (maxRows >= 16)  { return 16;  }
    if (maxRows >= 8)   { return 8;   }
    if (maxRows >= 4)   { return 4;   }
    if (maxRows >= 2)   { return 2;   }
    return 1;
}

} // namespace FusedGdnGating

#endif // FUSED_GDN_GATING_TILING_UTILS_H
