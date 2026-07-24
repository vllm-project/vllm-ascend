/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * BSD 3-Clause License.
 *
 * SolveTri 公共定义
 */
 #ifndef SOLVE_TRI_COMMON_H
 #define SOLVE_TRI_COMMON_H

 #include "kernel_operator.h"

 // AIC/AIV 同步标志常量
 constexpr uint64_t SYNC_AIV_AIC_FLAG_SOLVE = 3;
 constexpr uint64_t SYNC_AIC_AIV_FLAG_SOLVE = 5;

 // GM 共享 workspace slot（ND 行优先）
 constexpr int32_t GM_WS_I    = 0;
 constexpr int32_t GM_WS_INEG = 1;
 constexpr int32_t GM_WS_ZERO = 2;
 constexpr int32_t GM_NUM_SHARED_SLOTS = 3;

 // AIV 核生成辅助矩阵的参数
 constexpr int32_t ROWS_PER_AIV_CORE = 8;
 constexpr int32_t DIAG_BLOCK_ELEMS  = ROWS_PER_AIV_CORE * 16;  // 8x16

 // 8x16 ND 块中对角 mask（偶数条带）
 // 对角在 col 0..7: elem = i*16 + i
 constexpr uint64_t DIAG_MASK_8X16_EVEN[2] = {
     0x0008000400020001ULL,
     0x0080004000200010ULL
 };

 #endif  // SOLVE_TRI_COMMON_H
