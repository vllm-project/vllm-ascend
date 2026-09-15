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
* \file fused_infer_attention_score_common.h
* \brief
*/

#ifndef FUSED_INFER_ATTENTION_SCORE_COMMON_H
#define FUSED_INFER_ATTENTION_SCORE_COMMON_H

#include "attn_infra/common.hpp"
#include "attn_infra/arch.hpp"
#include "attn_infra/layout.hpp"

#include "attn_infra/gemm_qk.hpp"
#include "attn_infra/gemm_pv.hpp"
#include "attn_infra/gemm_common.hpp"

#include "attn_infra/epilogue_online_softmax.hpp"
#include "attn_infra/epilogue_rescale_o.hpp"
#include "attn_infra/epilogue_init_outputs.hpp"
#include "attn_infra/epilogue_common.hpp"
#include "attn_infra/combine_scale.hpp"
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

    namespace FaiKernel {
        constexpr uint32_t BLOCK_SIZE = 16;

        enum class MaskType : uint32_t {
            NO_MASK = 0,
            MASK_CAUSAL = 1
        };

        enum class inputLayout : uint32_t {
            TND = 1
        };
    };

    struct FAIKernelParams {
        // Data members
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR mask;
        GM_ADDR blockTables;
        GM_ADDR o;
        GM_ADDR lse;
        GM_ADDR sparseStats;
        GM_ADDR workSpace;
        GM_ADDR tiling;

        // Methods
        __aicore__ inline FAIKernelParams() {}

        __aicore__ inline FAIKernelParams(GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_,
                GM_ADDR o_, GM_ADDR lse_, GM_ADDR sparseStats_, GM_ADDR workSpace_, GM_ADDR tiling_)
            : q(q_), k(k_), v(v_), mask(mask_), blockTables(blockTables_),
                o(o_), lse(lse_), sparseStats(sparseStats_), workSpace(workSpace_), tiling(tiling_) {}
    };

    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }

    __aicore__ inline uint32_t GetQSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }

    __aicore__ inline uint32_t GetKSBlockTile(uint32_t kvSeqlen)
    {
        uint32_t kSBlockTile = MAX_KV_STACK_LEN;
        return kSBlockTile;
    }
}
#endif
