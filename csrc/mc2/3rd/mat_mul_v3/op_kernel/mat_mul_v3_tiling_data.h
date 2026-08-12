/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_v3_tiling_data.h
 * \brief
 */

#ifndef MAT_MUL_V3_TILING_DATA_H
#define MAT_MUL_V3_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) Mc2L2cacheUseInfo {
    uint32_t l2CacheFlag;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2L2cacheTilePara {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2MatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isNzA;
    uint32_t isNzB;
    uint32_t isHf32;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct alignas(8) Mc2MatmulV3TilingData {
    AscendC::tiling::TCubeTiling matmulTiling;
    Mc2L2cacheTilePara tileL2cacheTiling;
    Mc2MatMulRunInfo matmulRunInfo;
    Mc2L2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};
#pragma pack(pop)

#endif
