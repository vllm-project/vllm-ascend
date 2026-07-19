/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope_tiling_data.h */
#ifndef DGEMMA_FUSED_NORM_ROPE_TILING_DATA_H
#define DGEMMA_FUSED_NORM_ROPE_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"
namespace DgemmaFusedNormRope {
#pragma pack(push, 8)
struct alignas(8) DgemmaFusedNormRopeTilingData {
    uint32_t numTokens;
    uint32_t numQHeads;
    uint32_t numKvHeads;
    uint32_t headDim;      // = 256
    uint32_t rotaryDim;    // = headDim/2 = 128
    float    epsilon;
    float    invHeadDim;   // 1.0f/headDim, precomputed host-side
};
#pragma pack(pop)
}
#endif
