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
 * \file mega_moe_workspace_info.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#include "kernel_math_util.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
#define GM_ADDR uint8_t*
#define HOST_DEVICE
#include "host_math_util.h"
#endif

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t TOKEN_SCALE_SIZE = 9U * 1024U;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_GMM1_OFFSET = 12UL;
constexpr uint64_t IDX_GMM2_OFFSET = 13UL;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_EPILOGUE_FLAG = 8;
constexpr uint16_t FLAG_ID_MAX_PER_V = 16;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;
// wave-grain dispatch flag: must match MegaMoeImpl::L1_TILE_M
constexpr int64_t DISPATCH_WAVE_TILE_M = 256LL;
// GMM2 tile size for token group calculation
constexpr int64_t L1_TILE_M_128 = 128LL;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t TWO_FLAG = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E4M3_QUANT = 4U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t HALF_TO_FP32 = 2U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
// Combine quantization modes
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
// Combine buffer constants
constexpr uint32_t TRIPLE_SIZE = 8U;  // 每个 token 的三元组大小（8 个 int32）
// GroupedMatmul modes
constexpr uint8_t GROUPED_MATMUL_MODE_GENERAL = 0U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W4 = 1U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
} // namespace

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR swigluQuantDataPtr;
    GM_ADDR swigluQuantScalePtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR tripleInfoPtr;
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR flagDispatchToGmm1Ptr;
    GM_ADDR flagSendCntToGmm1Ptr;
    GM_ADDR cumsumInfoPtr;
    GM_ADDR gmm1MmadResPtr;
    GM_ADDR gmm2MmadResPtr;
    GM_ADDR gmm2CombineSyncCounterPtr;

    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        workspaceSize = 0;
        dispatchRevDataPtr = base;

        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScalePtr = base + workspaceSize;

        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h /
            MXFP_SCALE_GROUP_NUM, ALIGN_512);

        swigluQuantDataPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize *
            tilingData->hiddenDim / SWIGLU_N_HALF, ALIGN_512);

        swigluQuantScalePtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim /
            SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        expertRevTokenNumsPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->expertPerRank * ALIGN_32 * tilingData->aicNum, ALIGN_512);

        tripleInfoPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        flagSwiGluToGmm2Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE;

        flagDispatchToGmm1Ptr = base + workspaceSize;
        // wave-grain dispatch flag: per expert allocate one slot per wave (L1_TILE_M=256 rows),
        // aligned up to INT_CACHELINE so each expert's slot block stays cache-line clean.
        int64_t maxWavesPerExpert = Ops::Base::CeilDiv(
            static_cast<int64_t>(tilingData->maxOutputSize), DISPATCH_WAVE_TILE_M);
        int64_t dispatchFlagSlotsPerExpert = Ops::Base::CeilAlign(maxWavesPerExpert,
            static_cast<int64_t>(INT_CACHELINE));
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * dispatchFlagSlotsPerExpert;

        // SendCntCal(aiv1) -> GMM1 consumer(aic) 的 count-ready 软同步标志位。
        // 每个 (expert, aicCore) 独占一个 cache line(INT_CACHELINE 个 int32)，
        // 专家数 > 16 时不再复用单一 CrossCore 硬同步 flag，避免计数器反转。
        flagSendCntToGmm1Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * tilingData->expertPerRank * tilingData->aicNum;

        // Conditional allocation for A8W4 / combine-quant paths.
        // 以下条件分配与 mega_moe.h 编译期守卫 (ENABLE_A8W4 / CombineQuantMode) 一致，
        // 由 TilingKey 保证同步。
        // A8W4-only: cumsum GM backup and GMM1 intermediate result.
        cumsumInfoPtr = nullptr;
        gmm1MmadResPtr = nullptr;
        gmm2MmadResPtr = nullptr;
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
            // cumsumInfo: per-core backup of cumsum state (expertPerRank × epWorldSize int32 per core).
            cumsumInfoPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(SIZE_INT_32 * tilingData->expertPerRank * tilingData->epWorldSize),
                ALIGN_32) * tilingData->aicNum;
            // gmm1MmadRes: GMM1 matmul output (maxOutputSize × hiddenDim bf16).
            gmm1MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // gmm2MmadRes: GMM2 matmul output (maxOutputSize × h bf16), needed by A8W4 and combine-quant.
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            gmm2MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->h;
        }

        // Combine-quant-only workspace buffers
        gmm2CombineSyncCounterPtr = nullptr;
        if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            // Token group completion counters
            gmm2CombineSyncCounterPtr = base + workspaceSize;
            // 紧密排布：每个 expert 分配 blockAivNum 个 slot
            int64_t stridePerExpert = Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->blockAivNum) * INT_CACHELINE * SIZE_INT_32, ALIGN_128);
            workspaceSize += stridePerExpert * tilingData->expertPerRank;
        }
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
