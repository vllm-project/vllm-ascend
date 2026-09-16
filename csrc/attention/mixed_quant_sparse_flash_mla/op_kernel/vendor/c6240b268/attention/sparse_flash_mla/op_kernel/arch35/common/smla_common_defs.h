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
 * \file smla_common_defs.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子共用的
 *        常量、核内/跨核 flag 宏与对齐函数定义。
 */

#ifndef SMLA_COMMON_DEFS_H
#define SMLA_COMMON_DEFS_H

#include <stdint.h>

// ===== 全局常量（三算子共享） =====
constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536; // 65536表示64*1024
constexpr uint32_t BUFFER_SIZE_16K = 16384;      // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768;      // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_96K = 98304;      // 98304表示96 * 1024
constexpr uint32_t BUFFER_SIZE_256K = 262144;    // 262144表示256 * 1024
constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;

// ===== smla / mqsmla 共享（qsmla 无 cube buffer） =====
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33U; // 同token最大规约块数

// C 侧 buffer 元素个数 (tensor 偏移/地址递增用)
constexpr uint32_t L1Q_ELEM_PER_BUF = 16384;        // Q_T, 32KB
constexpr uint32_t L1_RIGHT_ELEM_PER_BLOCK = 65536; // Q_T, 128KB
constexpr uint32_t L0A_ELEM_PER_BUF = 8192;         // Q_T, 16KB
constexpr uint32_t L0B_ELEM_PER_BUF = 16384;        // Q_T, 32KB
constexpr uint32_t L0C_ELEM_PER_BUF = 32768;        // T  , 128KB

// ===== C 侧核内 flag id (各 HardEvent 命名空间独立) =====
#define INNERCORE_L0AB(s) (s)       // 0,1   M_MTE1 / MTE1_M
#define INNERCORE_L0C(s) (s)        // 0,1   FIX_M / M_FIX
#define INNERCORE_L1Q(s) (s)        // 0,1,2 MTE1_MTE2 / MTE2_MTE1
#define INNERCORE_L1KV(s) (3 + (s)) // 3,4,5 MTE1_MTE2 / MTE2_MTE1

// ===== 跨核 flag id (mode 4) =====
#define CROSSCORE_L1P(s) (s) // 0,1
#define CROSSCORE_BMM2 (2)
#define CROSSCORE_BMM1(s) (3 + (s))  // 3,4
#define CROSSCORE_V0RES(s) (5 + (s)) // 5,6,7 (GM backward)

// ===== 对齐函数 =====
namespace AttentionCommon {
__aicore__ constexpr uint64_t Align2Func(uint64_t data)
{
    return (data + 1UL) >> 1UL << 1UL; // 向上2对齐, +1移位2
}

__aicore__ constexpr uint64_t Align8Func(uint64_t data)
{
    return (data + 7UL) >> 3UL << 3UL; // 向上8对齐, +7移位3
}

__aicore__ constexpr uint64_t Align16Func(uint64_t data)
{
    return (data + 15UL) >> 4UL << 4UL; // 向上16对齐, +15移位4
}

__aicore__ constexpr uint64_t Align64Func(uint64_t data)
{
    return (data + 63UL) >> 6UL << 6UL; // 向上64对齐, +63移位6
}
} // namespace AttentionCommon

#endif // SMLA_COMMON_DEFS_H
