/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_tiling_data.h
 *  \brief Tiling for MIX op: qkv_proj GEMM (cube) + q/k/v RMSNorm + neox RoPE (vector). */
#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TILING_DATA_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"
namespace DgemmaFusedQkvProjNormRope {
#pragma pack(push, 8)
struct alignas(8) DgemmaFusedQkvProjNormRopeTilingData {
    // ---- GEMM tiling (cube): y[m,n] = hidden[m,k] @ Wqkv.T (Wqkv row-major [n,k] -> ColumnMajor [k,n]) ----
    uint32_t m;            // num tokens
    uint32_t k;            // hidden_size = 2816
    uint32_t n;            // qkv out = 8192
    uint32_t m0;           // L1 tile M (per-split rows)
    uint32_t swizzlCount;  // block swizzle count
    uint32_t coreNum;      // aic core count; high bits carry optional output mode flags
    // ---- epilogue (vector): split + RMSNorm + neox RoPE ----
    uint32_t numQHeads;    // 16 -> qSize = 16*headDim
    uint32_t numKvHeads;   // 8  -> kvSize = 8*headDim
    uint32_t headDim;      // 256
    uint32_t rotaryDim;    // headDim/2 = 128
    uint32_t syncDoneFlag; // AIC -> AIV qkv workspace ready flag
    uint32_t syncReadyFlag;// AIV -> AIC startup alignment flag
    float    epsilon;
    float    invHeadDim;   // 1.0f/headDim precomputed host-side
};
#pragma pack(pop)
}
#endif
