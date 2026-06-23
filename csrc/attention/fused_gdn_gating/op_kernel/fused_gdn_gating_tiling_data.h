/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating_tiling_data.h
 * \brief Tiling data shared between host-side tiling and device-side kernel.
 */

#ifndef FUSED_GDN_GATING_TILING_DATA_H
#define FUSED_GDN_GATING_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace FusedGdnGating {

#pragma pack(push, 8)

struct alignas(8) FusedGdnGatingTilingData {

    uint32_t numHeads;
    float beta;

    // --------------------------------------------------------
    // 2. 910/910B 架构专用参数
    // --------------------------------------------------------
    uint32_t numBatches;
    uint32_t rowsPerIter;
    uint32_t useBulkDma;

    // --------------------------------------------------------
    // 3. 310P 架构专用参数
    // --------------------------------------------------------
    uint32_t usedCoreNum;
    uint32_t alignedLength;
    uint32_t tailLength;
    uint32_t tileRows;
    float inv_beta;

    float threshold;
};

#pragma pack(pop)

} // namespace FusedGdnGating

#endif // FUSED_GDN_GATING_TILING_DATA_H
