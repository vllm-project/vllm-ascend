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
 * \file quant_matmul_tiling_common.h
 * \brief Shared constants and runtime state for MX tiling generation.
 */
#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "tiling/platform/platform_ascendc.h"

// Static hardware information queried from the Ascend platform runtime.
struct QuantMatmulPlatformInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0cSize{0UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

// Original problem shape provided by the caller.
struct QuantMatmulArgs {
    uint64_t m = 0UL;
    uint64_t k = 0UL;
    uint64_t n = 0UL;
    bool transA = false;
    bool transB = true;
};

// Intermediate tiling state built step by step by the tiling engine.
//
// The engine first determines a basic block shape, then derives tail-handling
// policy, L1/L0 buffering depth, and finally the information consumed by the
// kernel launch path.
struct QuantMatmulRunInfo {
    // Base tile shape used by the scheduler and compute pipeline.
    uint64_t baseM{0UL};
    uint64_t baseN{0UL};
    uint64_t baseK{0UL};

    // stepKa/stepKb count how many baseK tiles one outer streaming step spans.
    uint64_t stepKa{0UL};
    uint64_t stepKb{0UL};
    uint64_t depthA1{0UL};
    uint64_t depthB1{0UL};

    // Reserved fields for future L1 shape bookkeeping.
    uint64_t mL1{0UL};
    uint64_t nL1{0UL};
    uint64_t kL1{0UL};

    // Reserved fields for future tail-count and buffer bookkeeping.
    uint64_t mTailCnt{0UL};
    uint64_t nTailCnt{0UL};
    uint64_t l1BufferNum{0UL};

    // Output buffering mode in L0C.
    uint64_t dbL0c{0UL};

    // Basic scheduler statistics after the base tile is selected.
    uint64_t mBlockCnt{0UL};
    uint64_t nBlockCnt{0UL};
    uint64_t totalBlockCnt{0UL};

    // Tail split factors used when the last scheduling round is further divided.
    uint64_t mTailTile{1UL};
    uint64_t nTailTile{1UL};
    uint64_t mTailSize{0UL};
    uint64_t nTailSize{0UL};
    uint64_t tailBlockCnt{0UL};

    // Load-balance metadata for merged tail blocks on M and N.
    uint64_t mBaseTailSplitCnt{1UL};
    uint64_t mTailMain{0UL};
    uint64_t nBaseTailSplitCnt{1UL};
    uint64_t nTailMain{0UL};

    // Scale reuse factors in L1 for A and B.
    uint64_t scaleFactorA{0UL};
    uint64_t scaleFactorB{0UL};

    // Whether the current shape allows the "A full load" fast path.
    bool isAFullLoad{false};
};

