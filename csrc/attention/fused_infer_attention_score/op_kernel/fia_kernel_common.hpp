/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file kernel_common.hpp
* \brief
*/

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include "attn_infra/fused_base_defs.hpp"
#include "attn_infra/arch/fused_arch.hpp"
#include "attn_infra/layout/fused_layout.hpp"

#include "attn_infra/gemm/block/fused_block_mmad.hpp"
#include "attn_infra/gemm/fused_gemm_dispatch_policy.hpp"
#include "attn_infra/gemm/fused_gemm_type.hpp"

#include "attn_infra/arch/fused_cross_core_sync.hpp"
#include "attn_infra/arch/fused_resource.hpp"
#include "attn_infra/epilogue/block/fused_block_epilogue.hpp"
#include "attn_infra/epilogue/fused_epilogue_dispatch_policy.hpp"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace KernelCommon {
    constexpr uint32_t QK_READY_ID = 1;
    constexpr uint32_t SOFTMAX_READY_ID = 2;
    constexpr uint32_t PV_READY_ID = 3;
    constexpr uint32_t PRE_LAUNCH = 2;
    constexpr uint32_t N_SPLIT_HELPER = 2;
    constexpr uint32_t MAX_KV_STACK_LEN = 512;
    constexpr uint32_t Q_TILE_CEIL = 128;
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    constexpr uint32_t L1_MAX_SIZE = 524288;
    constexpr uint32_t L1_MAX_N_NUM = 128;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t COMP_TRIU_MASK_DIM_LEN = 2048;
    constexpr uint32_t NUM_32 = 32;
    constexpr uint32_t NUM_128 = 128;
    constexpr uint32_t NUM_256 = 256;
    constexpr uint32_t FLOAT_SIZE = 4;
    constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;

    template <typename T>
    __aicore__ inline
    T AlignUp(T a, T b)
    {
        return (b == 0) ? 0 : (a + b - 1) / b * b;
    }

    template <typename T>
    __aicore__ inline
    T Max(T a, T b)
    {
        return (a > b) ? a : b;
    }

    namespace FaiKernel {
        constexpr uint32_t BLOCK_SIZE = 16;

        enum class cvPipeLineType : uint32_t {
            FAI_COMMON_NORMAL = 0,
            FAI_COMMON_CHUNK_MASK = 1,
        };

        enum class MaskType : uint32_t {
            NO_MASK = 0,
            MASK_CAUSAL = 1,
            MASK_SPEC = 2,
            MASK_SWA = 4,
            FULL_MASK = 5
        };

        enum class inputLayout : uint32_t {
            BSND = 0,
            TND = 1
        };
    };

    struct FAIKernelParams {
        // Data members
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR pseShift;
        GM_ADDR mask;
        GM_ADDR blockTables;
        GM_ADDR actualQseqlen;
        GM_ADDR actualKvseqlen;
        GM_ADDR o;
        GM_ADDR lse;
        GM_ADDR workSpace;
        GM_ADDR tiling;
        GM_ADDR sink;

        // Methods
        __aicore__ inline FAIKernelParams() {}

        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR pseShift_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR o_, GM_ADDR lse_, GM_ADDR workSpace_, GM_ADDR tiling_, GM_ADDR sink_)
            : q(q_), k(k_), v(v_), pseShift(pseShift_), mask(mask_), blockTables(blockTables_), actualQseqlen(actualQseqlen_),
                actualKvseqlen(actualKvseqlen_), o(o_), lse(lse_), workSpace(workSpace_), tiling(tiling_), sink(sink_) {}
    };

    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    __aicore__ inline uint32_t GetKvNBlockTile(uint32_t rowNumPerQSGTile, uint32_t kvHead)
    {
        uint32_t rowNumCeilPerQSGKvNTile = Q_TILE_CEIL;
        uint32_t kvNBlockTile = rowNumCeilPerQSGKvNTile / rowNumPerQSGTile;
        kvNBlockTile = kvNBlockTile < kvHead ? kvNBlockTile : kvHead;
        kvNBlockTile = kvNBlockTile < 1 ? 1 : kvNBlockTile;
        return kvNBlockTile;
    }

    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }

    __aicore__ inline uint32_t GetQSBlockTileDecode(uint32_t qSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL < qSeqlen ? Q_TILE_CEIL : qSeqlen;
        return qSBlockTile;
    }
    __aicore__ inline uint32_t GetKSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t kSBlockTile = MAX_KV_STACK_LEN;
        return kSBlockTile;
    }
}
#endif