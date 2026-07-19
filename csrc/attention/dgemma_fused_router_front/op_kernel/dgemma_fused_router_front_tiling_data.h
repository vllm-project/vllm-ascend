/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_tiling_data.h
 *  \brief Tiling for MIX op: RMSNorm+root_size+scale (vector) -> GateLinear GEMM (cube). */
#ifndef DGEMMA_FUSED_ROUTER_FRONT_TILING_DATA_H
#define DGEMMA_FUSED_ROUTER_FRONT_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"
namespace DgemmaFusedRouterFront {
#pragma pack(push, 8)
struct alignas(8) DgemmaFusedRouterFrontTilingData {
    // MIX: vector normalizes x[m,k] -> scratch[m,k]; cube does scratch @ W.T -> out[m,n]
    uint32_t m;            // num tokens
    uint32_t k;            // hidden_size = 2816
    uint32_t n;            // num_experts = 128 (router logits width)
    uint32_t topK;         // Gemma4 top_k = 8
    uint32_t m0;           // L1 tile M (per-split rows)
    uint32_t swizzlCount;  // block swizzle count
    uint32_t coreNum;      // aic core count
    uint32_t syncReadyFlag;// AIV -> AIC norm scratch ready flag
    uint32_t syncDoneFlag; // AIC -> AIV logits scratch ready flag
    float    epsilon;
    float    invHidden;    // 1.0f/hidden_size (for RMSNorm mean)
    float    rootSize;     // 1.0f/sqrt(hidden_size)
};
#pragma pack(pop)
}
#endif
