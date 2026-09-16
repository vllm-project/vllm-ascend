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
 * \file mixed_quant_sparse_flash_mla_common_arch35.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../mixed_quant_sparse_flash_mla_common.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/static_buffer.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/smla_common_defs.h"

// ===== V 侧主流程 buffer slot id (MTE2_V/V_MTE2/MTE3_V/V_MTE3 各自独立) =====
#define INNERCORE_STAGE0_IN(s) (s)        // 0,1: stage0In, MTE2_V / V_MTE2
#define INNERCORE_STAGE0_OUT(s) (2 + (s)) // 2,3: stage0Out, MTE3_V / V_MTE3
#define INNERCORE_STAGE1(s) (4 + (s))     // 4,5: stage1, MTE3_V / V_MTE3
#define INNERCORE_STAGE2 (6)              // 6:   stage2, MTE3_V / V_MTE3
#define INNERCORE_SINKS_SYNC (7)          // 7:   sinks,  MTE2_V / V_MTE2

// ===== GetKVPhyAddr 独立阶段, 保留原 flag 值 =====
#define INNERCORE_PHYADDR_BLKTABLE_FREE (3)   // V_MTE2, blkTable 空闲
#define INNERCORE_PHYADDR_BLKTABLE_READY (8)  // MTE2_V, blkTable 就绪
#define INNERCORE_PHYADDR_SPARSEIDX_FREE (4)  // V_MTE2, sparseIdx 空闲
#define INNERCORE_PHYADDR_SPARSEIDX_READY (6) // MTE2_V, sparseIdx 就绪
#define INNERCORE_PHYADDR_KVADDR_READY (5)    // V_MTE3, kvPhyAddr 就绪
#define INNERCORE_PHYADDR_KVADDR_FREE (7)     // MTE3_V, kvPhyAddr 空闲

// ===== V 侧其余核内 flag id (FD / batch-consistency / LSE / init) =====
// 各 HardEvent 命名空间独立; 主流程 stage 槽位 id 已占 0~6/7, 其余事件在其命名空间内取不冲突值。
// V_MTE2: STAGE0_IN 0,1
#define INNERCORE_REDUCE_MAXSUM_V_MTE2 (2) // batch-consistency reduce max/sum
#define INNERCORE_INTRAPARTIALO_V_MTE2 (3) // batch-consistency partial O
#define INNERCORE_FD_V_MTE2(s) (4 + (s))   // 4,5: flash decode

// MTE2_V: STAGE0_IN 0,1; SINKS 7
#define INNERCORE_REDUCE_MTE2_V (2) // batch-consistency reduce
#define INNERCORE_FD_MTE2_V (3)     // flash decode

// V_MTE3: STAGE0_OUT 2,3; STAGE1 4,5; STAGE2 6
#define INNERCORE_LSE_V_MTE3 (1) // LSE out (条件阶段)

// MTE3_V: STAGE0_OUT 2,3; STAGE1 4,5; STAGE2 6
#define INNERCORE_STAGE_FD_MTE3_V (7) // FD/BC staging (StageVec1Lse)
#define INNERCORE_LSE_MTE3_V (0)      // LSE out (条件阶段)
#define INNERCORE_FD_MTE3_V (1)       // FD mte3ToV (SyncAll 之后阶段)
#define INNERCORE_INITOUT_MTE3_V (0)  // init 阶段 (CleanOutput)

// MTE3_MTE2 (batch consistency + fd)
#define INNERCORE_INTRALSE_MTE3_MTE2(s) (s)        // 0,1
#define INNERCORE_INTRAATTN_MTE3_MTE2(s) (2 + (s)) // 2,3
#define INNERCORE_FD_MTE3_MTE2 (4)

namespace BaseApi {
using AttentionCommon::Align2Func;
using AttentionCommon::Align8Func;
using AttentionCommon::Align16Func;
using AttentionCommon::Align64Func;
} // namespace BaseApi

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, QSMLA_LAYOUT LAYOUT_T, \
              QSMLA_LAYOUT KV_LAYOUT_T, QSMLATemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, \
              SCALE_CONTIGUOUS_MODE QUANT_MODE, bool IS_BATCH_CONSISTENCY, bool IS_VEC_S2PHYADDR, bool HIGH_PERF>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, QUANT_MODE, \
        IS_BATCH_CONSISTENCY, IS_VEC_S2PHYADDR, HIGH_PERF

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::BSND) \
    X(KV_LAYOUT_T, QSMLA_LAYOUT, QSMLA_LAYOUT::PA_BBND) \
    X(TEMPLATE_MODE, QSMLATemplateMode, QSMLATemplateMode::CSA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(QUANT_MODE, SCALE_CONTIGUOUS_MODE, SCALE_CONTIGUOUS_MODE::CONTIGUOUS) \
    X(IS_BATCH_CONSISTENCY, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false) \
    X(HIGH_PERF, bool, false)

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
