/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: FusedDeepMoe tilingData definition file
 * Author: WANG Qiankun
 * Create: 2025-07-19
 * Note:
 * History: 2025-07-19 create FusedDeepMoe tilingData definition file
 */

#ifndef FUSED_DEEP_MOE_TILING_H
#define FUSED_DEEP_MOE_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct FusedDeepMoeInfo {
    uint32_t epRankSize;           // epRankSize
    uint32_t epRankId;             // epRankId
    uint32_t moeExpertNum;         // moe expert number
    uint32_t moeExpertNumPerRank;  // moe expert number per rank
    uint32_t sharedExpertNum;      // shared expert number
    uint32_t sharedExpertRankNum;  // shared expert rank number
    uint32_t quantMode;            // quant mode
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aicNum;               // aivNum
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t gmm1HLen;
};

struct FusedDeepMoeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    FusedDeepMoeInfo disGmmDeqSwigluQuantGmmDeqComInfo;
};

constexpr uint32_t GM_ALIGN_BYTE = 512;
constexpr uint32_t CUSTOM_PRELOAD_STAGES = 1;
constexpr uint32_t CUSTOM_L1_STAGES = 2;
constexpr uint32_t CUSTOM_L0A_STAGES = 2;
constexpr uint32_t CUSTOM_L0B_STAGES = 2;
constexpr uint32_t CUSTOM_L0C_STAGES = 1;
constexpr bool CUSTOM_ENABLE_UNIT_FLAG = true;
constexpr bool CUSTOM_ENABLE_SHUFFLE_K = true;

constexpr uint32_t GMM1_L1M = 256;
constexpr uint32_t GMM1_L1N = 128;
constexpr uint32_t GMM1_L1K = 512;
constexpr uint32_t GMM1_L0K = 128;
constexpr uint32_t GMM1_EPIM = 64;
constexpr uint32_t GMM1_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM1_SWIZZLE_DIRECTION = 0;

constexpr uint32_t GMM2_L1A_STAGES = 4;
constexpr uint32_t GMM2_L1B_STAGES = 2;
constexpr uint32_t GMM2_L0A_STAGES = 4;
constexpr uint32_t GMM2_L0B_STAGES = 2;
constexpr uint32_t GMM2_L1M = 128;
constexpr uint32_t GMM2_L1N = 256;
constexpr uint32_t GMM2_L1K = 512;
constexpr uint32_t GMM2_L0K = 128;
constexpr uint32_t GMM2_EPIM = 32;
constexpr uint32_t GMM2_SWIZZLE_OFFSET = 3;
constexpr uint32_t GMM2_SWIZZLE_DIRECTION = 0;

constexpr uint32_t WORKSPACE_STAGES = 4;

constexpr uint32_t EXEC_FLAG_DEEP_FUSE = (1U << 0);

#endif  // FUSED_DEEP_MOE_TILING_H
