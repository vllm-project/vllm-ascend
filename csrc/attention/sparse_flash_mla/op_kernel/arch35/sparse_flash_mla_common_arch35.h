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
 * \file sparse_flash_mla_common_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_COMMON_ARCH35_H
#define SPARSE_FLASH_MLA_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../sparse_flash_mla_common.h"
#if __has_include("common/static_buffer.h")
#include "common/static_buffer.h"
#endif

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;

// ===== C 侧 buffer 元素个数 (tensor 偏移/地址递增用) =====
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

// ===== V 侧主流程 =====
#define INNERCORE_STAGE1(s) (4 + (s)) // 4,5  V_MTE3 / MTE3_V (stage1->L1)
#define INNERCORE_STAGE2 (6)          // 6    V_MTE3 / MTE3_V (vec2结果+attentionOut拷出+staging, 串行复用)
#define INNERCORE_STAGE0OUT_MTE3_MTE2(s) (s) // 0,1  Vec0 stage0OutBuf (原 mte3ToMte2)
#define INNERCORE_STAGE0OUT_MTE2_MTE3(s) (s) // 0,1  Vec0 stage0OutBuf (原 mte2ToMte3)
#define INNERCORE_SINKS_SYNC (7)             // 7    MTE2_V / V_MTE2

// ===== GetKVPhyAddr 独立相位 (保留原值; 与主流程相位分离, 可复用) =====
#define INNERCORE_PHYADDR_BLKTABLE_FREE (3)   // V_MTE2
#define INNERCORE_PHYADDR_BLKTABLE_READY (8)  // MTE2_V
#define INNERCORE_PHYADDR_SPARSEIDX_FREE (4)  // V_MTE2
#define INNERCORE_PHYADDR_SPARSEIDX_READY (6) // MTE2_V
#define INNERCORE_PHYADDR_KVADDR_READY (5)    // V_MTE3
#define INNERCORE_PHYADDR_KVADDR_FREE (7)     // MTE3_V

// ===== batch-consistency / LSE / FD / init =====
#define INNERCORE_REDUCE_MAXSUM_V_MTE2 (2)         // V_MTE2
#define INNERCORE_INTRAPARTIALO_V_MTE2 (3)         // V_MTE2
#define INNERCORE_FD_V_MTE2(s) (4 + (s))           // 4,5 V_MTE2
#define INNERCORE_REDUCE_MTE2_V (2)                // MTE2_V
#define INNERCORE_FD_MTE2_V (3)                    // MTE2_V
#define INNERCORE_LSE_V_MTE3 (1)                   // V_MTE3
#define INNERCORE_STAGE_FD_MTE3_V (7)              // MTE3_V (Stage* staging)
#define INNERCORE_LSE_MTE3_V (0)                   // MTE3_V
#define INNERCORE_FD_MTE3_V (1)                    // MTE3_V
#define INNERCORE_INITOUT_MTE3_V (0)               // MTE3_V (init, 与 LSE 相位分离)
#define INNERCORE_INTRALSE_MTE3_MTE2(s) (2 + (s))  // 2,3
#define INNERCORE_INTRAATTN_MTE3_MTE2(s) (4 + (s)) // 4,5
#define INNERCORE_FD_MTE3_MTE2 (6)

// ===== 跨核 flag id (mode 4) =====
#define CROSSCORE_L1P(s) (s) // 0,1
#define CROSSCORE_BMM2 (2)
#define CROSSCORE_BMM1(s) (3 + (s))  // 3,4
#define CROSSCORE_V0RES(s) (5 + (s)) // 5,6,7 (仅 CSA kernel)

constexpr uint32_t L0AB_SHARED_SIZE_64K = 65536;  // 65536表示64*1024
constexpr uint32_t L0C_SHARED_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t BUFFER_SIZE_8K = 8192;     // 8192表示8 * 1024
constexpr uint32_t BUFFER_SIZE_16K = 16384;   // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768;   // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_64K = 65536;   // 65536表示64 * 1024
constexpr uint32_t BUFFER_SIZE_96K = 98304;   // 98304表示96 * 1024
constexpr uint32_t BUFFER_SIZE_128K = 131072; // 131072表示128 * 1024
constexpr uint32_t BUFFER_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33U;

namespace SMLAKernel {
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
} // namespace SMLAKernel

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool IS_FD, SMLA_LAYOUT LAYOUT_T, \
              SMLA_LAYOUT KV_LAYOUT_T, SMLATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, bool IS_BATCH_CONSISTENCY, \
              bool IS_VEC_S2PHYADDR>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, IS_FD, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, IS_BATCH_CONSISTENCY, \
        IS_VEC_S2PHYADDR

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(IS_FD, bool, false) \
    X(LAYOUT_T, SMLA_LAYOUT, SMLA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, SMLA_LAYOUT, SMLA_LAYOUT::PA_BBND) \
    X(TEMPLATE_MODE, SMLATemplateMode, SMLATemplateMode::CSA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(IS_BATCH_CONSISTENCY, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) \
    end

#endif
