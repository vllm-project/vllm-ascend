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
 * \file quant_matmul_tiling_data.h
 * \brief Serialized tiling data passed from the host launcher to the kernel.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Serialized tiling result passed from host code to the kernel entry.
//
// The field order is part of the host-device contract, so layout stability is
// more important here than convenience of reordering members.
#pragma pack(push, 8)
struct alignas(8) QuantMatmulTilingData {
    // Original problem shape.
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};

    // Base tile shape selected by the tiling engine.
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};

    // Amount of K covered by one scale fragment staged in L1.
    uint32_t scaleKL1{0};

    // Tail split factors used in the final scheduling round.
    uint32_t mTailTile{1};
    uint32_t nTailTile{1};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};

    // Launch-time AIC count and buffering parameters.
    // Number of AICs launched for this kernel instance.
    uint32_t usedCoreNum{0};
    // Number of baseK tiles consumed before the rolling L1 buffers advance.
    uint8_t stepK{0};
    // Number of rolling A/B buffer slots reserved in L1.
    uint8_t nBufferNum{0};

    // Output buffering mode selected for the kernel implementation.
    uint8_t dbL0c{0};
};
#pragma pack(pop)

