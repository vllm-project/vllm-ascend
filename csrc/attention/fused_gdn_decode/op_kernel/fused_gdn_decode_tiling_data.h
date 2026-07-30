/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUSED_GDN_DECODE_TILING_DATA_H
#define FUSED_GDN_DECODE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace FusedGdnDecode {
#pragma pack(push, 8)
struct alignas(8) FusedGdnDecodeTilingData {
    uint32_t b;
    uint32_t h;
    uint32_t hv;
    uint32_t k;
    uint32_t v;
    uint32_t bv;
    uint32_t vTiles;
    uint32_t stateBufferNum;
    uint32_t totalTasks;
    uint32_t mixedStride;
    uint32_t stateSlotStride;
    uint32_t stateHeadStride;
    uint32_t outBatchStride;
    float scale;
    float softplusThreshold;
    uint32_t ubRestBytes;
};
#pragma pack(pop)
} // namespace FusedGdnDecode

#endif // FUSED_GDN_DECODE_TILING_DATA_H
