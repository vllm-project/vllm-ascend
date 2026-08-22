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
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t *
#define HOST_DEVICE
#endif
#include "mega_moe_impl_base.h"

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
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
constexpr uint16_t FLAG_ID_MAX_PER_V = 16;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t MAX_AICORE_NUM = 36;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t TWO_FLAG = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t WEIGHT_INDEX = 3U;
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
constexpr int64_t DEQUANT_SCALE_EXPAND = 2U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
// Combine quantization modes
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
// Combine buffer constants
constexpr uint32_t META_INFO_SIZE = 8U; // 每个 token 的 metaInfo 大小（8 个 int32）
// GroupedMatmul modes
constexpr uint8_t GROUPED_MATMUL_MODE_GENERAL = 0U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W4 = 1U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
// a4w4 混合场景：GMM1 走 generic a4w4，GMM2 走 A8W4。GMM2 需要 gmm2MmadResPtr workspace。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4 = 3U;
// a4w4 NZ 场景：weight1/weight2 均为 fp4 NZ 格式。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4_NZ = 4U;
constexpr int64_t TOPO_TYPE_MTE = 0U;  // mte
constexpr int64_t TOPO_TYPE_URMA = 1U; // urma

// 返回从 flagSwiGluToGmm2Ptr 开始的连续 flag workspace 总字节数。基础布局按 expert 依次包含
// SwiGLU-ready、Dispatch-ready 和 send-count flag；MTE A8W8 wave 追加 GMM2-ready flag，
// URMA 或非 wave Combine 量化路径再追加 group-sync counter。Host tiling 用它申请/配置清零批次，
// kernel 用同一公式划分并清零该连续区域。
HOST_DEVICE int64_t CalcMegaMoeFlagWorkspaceSize(const MegaMoeTilingData *tilingData)
{
    bool isA8W8 = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_GENERAL ||
                  tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
    bool useMteA8W8Wave = tilingData->topoType == TOPO_TYPE_MTE && isA8W8;
    bool useGroupSyncCounters =
        tilingData->topoType == TOPO_TYPE_URMA ||
        (tilingData->combineQuantMode != COMBINE_NO_QUANT && !useMteA8W8Wave);
    int64_t maxWavesPerExpert =
        Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize),
                           static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256));
    int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
    int64_t swigluFlagSlotsPerExpert =
        useMteA8W8Wave ? waveFlagSlotsPerExpert : static_cast<int64_t>(INT_CACHELINE);
    int64_t routedExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

    int64_t flagElementCount =
        routedExpertCount *
        (swigluFlagSlotsPerExpert + waveFlagSlotsPerExpert +
         static_cast<int64_t>(INT_CACHELINE) * tilingData->aicNum);
    if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        flagElementCount += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE;
    }
    if (useMteA8W8Wave) {
        flagElementCount += routedExpertCount * tilingData->aicNum * INT_CACHELINE;
    }
    if (useGroupSyncCounters) {
        flagElementCount += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) *
                            routedExpertCount * INT_CACHELINE;
    }
    if (tilingData->sharedExpertNum > 0) {
        int64_t tokenGroupCount = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs),
                                                     static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256));
        flagElementCount += tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) *
                            static_cast<int64_t>(INT_CACHELINE);
    }
    return flagElementCount * SIZE_INT_32;
}

} // namespace

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR dispatchRevWeightsPtr{nullptr};
    GM_ADDR swigluQuantDataPtr;
    GM_ADDR swigluQuantScalePtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR metaInfoPtr;
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR flagDispatchToGmm1Ptr;
    GM_ADDR flagSendCntCalToUpdParamsPtr;
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertSwigluDataPtr{nullptr};
    GM_ADDR sharedExpertSwigluScalePtr{nullptr};
    GM_ADDR gmm1TileStatusPtr{nullptr}; // GMM1 tile 就绪状态位区（仅 prefetch 软同步分配）
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    GM_ADDR maskSlotPtr{nullptr};       // urma发送mask临时GM
    GM_ADDR dispatchL1CommPtr{nullptr}; // dispatch L1 communication workspace
    GM_ADDR dispatchCursorPtr{nullptr}; // dispatch cnt for each expert
    GM_ADDR dispatchDonePtr{nullptr};   // dispatch done
    GM_ADDR dispatchL2CommPtr{nullptr}; // dispatch l2 communication workspace

    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t serverNum = 1)
    {
        workspaceSize = 0;
        dispatchRevDataPtr = base;

        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScalePtr = base + workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * dispatchScaleElementsPerToken, ALIGN_512);

        if (tilingData->topkWeightsPrefetch == 1 && tilingData->topoType == TOPO_TYPE_URMA) {
            dispatchRevWeightsPtr = base + workspaceSize;
            uint32_t weightAlignBytes = Ops::Base::CeilAlign(
                static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->maxOutputSize) * static_cast<int64_t>(weightAlignBytes), ALIGN_512);
        }

        swigluQuantDataPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / MegaMoeImpl::SWIGLU_N_HALF, ALIGN_512);

        swigluQuantScalePtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim /
                                                  MegaMoeImpl::SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM,
                                              ALIGN_512);

        expertRevTokenNumsPtr = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(tilingData->moeExpertPerRank * ALIGN_32 * tilingData->aicNum, ALIGN_512);

        metaInfoPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        // 以下三组 flag 仅由路由 MoE 专家使用；共享专家路径不使用这些 flag。
        bool isGenericGmm = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_GENERAL ||
                            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
        bool useMteA8W8Wave = tilingData->topoType == TOPO_TYPE_MTE && isGenericGmm;
        bool useGroupSyncCounters =
            tilingData->topoType == TOPO_TYPE_URMA ||
            (tilingData->combineQuantMode != COMBINE_NO_QUANT && !useMteA8W8Wave);

        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize),
                               static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        // Only the wave pipeline can consume SwiGLU at 256-row granularity.
        // Legacy and layered paths retain one cache-line slot per expert.
        int64_t swigluFlagSlotsPerExpert =
            useMteA8W8Wave ? waveFlagSlotsPerExpert : static_cast<int64_t>(INT_CACHELINE);
        int64_t routedExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

        // Keep every scalar notification in one contiguous workspace region.
        // Each logical slot owns one 64B cache line, so ResetFlagList can clear
        // the complete region in one partitioned pass.
        flagSwiGluToGmm2Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * routedExpertCount * swigluFlagSlotsPerExpert;
        flagDispatchToGmm1Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * routedExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsPtr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * routedExpertCount * tilingData->aicNum;

        // Keep the per-AIC ready sequence contiguous with the preceding flags so ResetFlagList can clear all of them
        // with the same MTE reset pass. Each sequence occupies one 64B cacheline.
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
            flagGmmToEpiloguePtr = base + workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }

        if (useMteA8W8Wave) {
            gmm2ReadyPtr = base + workspaceSize;
            // slotIdx is expertIdx in PER_EXPERT mode and batchIdx in
            // PER_BATCH mode. Each AIC arrival owns one 64B cache line.
            workspaceSize += SIZE_INT_32 * routedExpertCount * tilingData->aicNum * INT_CACHELINE;
        }

        if (useGroupSyncCounters) {
            gmm2CombineSyncCounterPtr = base + workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) *
                             routedExpertCount * INT_CACHELINE * SIZE_INT_32;
        }

        // Shared expert GMM2 tile counter: tile 级 flag counter, 每 shared expert 一组 slot。
        // 纳入连续 flag 区以便 ResetFlagList 一次性清零。
        if (tilingData->sharedExpertNum > 0) {
            sharedExpertGmm2TileCounterPtr = base + workspaceSize;
            int64_t tokenGroupCount = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs),
                                                         static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256));
            workspaceSize += SIZE_INT_32 * tokenGroupCount *
                             static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
        }

        // Conditional allocation for A8W4 / combine-quant paths.
        // 以下条件分配与 mega_moe.h 编译期守卫 (ENABLE_A8W4 / ENABLE_A4W4 / CombineQuantMode) 一致，
        // 由 TilingKey 保证同步。
        // A8W4-only: cumsum GM backup and GMM1 intermediate result.
        cumsumInfoPtr = nullptr;
        gmm1MmadResPtr = nullptr;
        gmm2MmadResPtr = nullptr;
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 || tilingData->topkWeightsPrefetch == 1) {
            // cumsumInfo: per-core backup of cumsum state (moeExpertPerRank × epWorldSize int32 per core).
            if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
                cumsumInfoPtr = base + workspaceSize;
                workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_32 *
                                                                           tilingData->moeExpertPerRank *
                                                                           tilingData->epWorldSize),
                                                      ALIGN_32) *
                                 tilingData->aicNum;
            }
            // gmm1MmadRes: GMM1 matmul output (maxOutputSize × hiddenDim bf16).
            gmm1MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // gmm2MmadRes: GMM2 matmul output (maxOutputSize × h bf16), needed by A8W4 GMM1 path,
        // A4W4 hybrid path (GMM2 uses A8W4), A4W4_NZ, and combine-quant.
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ ||
            tilingData->combineQuantMode != COMBINE_NO_QUANT || tilingData->topoType == TOPO_TYPE_URMA ||
            useMteA8W8Wave) {
            if (useMteA8W8Wave) {
                workspaceSize = Ops::Base::CeilAlign(workspaceSize, static_cast<int64_t>(ALIGN_512));
            }
            gmm2MmadResPtr = base + workspaceSize;
            // The MTE wave path gives every routed row a unique [H] BF16
            // region for the complete kernel lifetime.  It is not a batch or
            // ring slot, so AIC never needs a consumer ACK before advancing.
            int64_t gmm2OutputBytes = static_cast<int64_t>(SIZE_BF_16) *
                                      static_cast<int64_t>(tilingData->maxOutputSize) *
                                      static_cast<int64_t>(tilingData->h);
            workspaceSize += useMteA8W8Wave ?
                                 Ops::Base::CeilAlign(gmm2OutputBytes, static_cast<int64_t>(ALIGN_512)) :
                                 gmm2OutputBytes;
        }

        // GMM1 tile 状态位区（仅 prefetch 路径分配，用于 AIC→AIV 软同步）。
        // 每个 tile 状态和末尾 allDone 状态都独占一条 64B cache line。
        gmm1TileStatusPtr = nullptr;
        if (tilingData->topkWeightsPrefetch == 1) {
            gmm1TileStatusPtr = base + workspaceSize;
            int64_t statusSlots =
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->maxTilesPerExpert + 1;
            int64_t statusBytes = SIZE_INT_32 * statusSlots * INT_CACHELINE;
            workspaceSize += Ops::Base::CeilAlign(statusBytes, ALIGN_512);
        }
        if (tilingData->topoType == TOPO_TYPE_URMA) {
            maskSlotPtr = base + workspaceSize;
            int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
            int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
                                   (int64_t)sizeof(int32_t);
            int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
            int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32; // mask + 32B count

            workspaceSize += Ops::Base::CeilAlign(
                (int64_t)tilingData->moeExpertPerRank * tilingData->epWorldSize * maskSlotSize,
                (int64_t)ALIGN_512);

            dispatchL1CommPtr = base + workspaceSize;
            int64_t serverWorkspaceBytes =
                static_cast<int64_t>(ALIGN_32) + static_cast<int64_t>(tilingData->bs) * ALIGN_32;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchCursorPtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(static_cast<int64_t>(serverNum * SIZE_INT_32), static_cast<int64_t>(ALIGN_512));

            dispatchDonePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->aicNum * SIZE_INT_32),
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchL2CommPtr = base + workspaceSize;
            int64_t flagSnapshotBytes = static_cast<int64_t>(tilingData->bs) * static_cast<int64_t>(sizeof(uint64_t));
            int64_t dispatchL2ScratchBytes = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(ALIGN_512));
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * dispatchL2ScratchBytes;
        }

        // Shared expert workspace buffers
        if (tilingData->sharedExpertNum > 0) {
            // sharedExpertResult: GMM2 output for shared experts [sharedExpertNum × bs × h]
            sharedExpertResultPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->h, ALIGN_512);
            // sharedExpertGmm1Out: GMM1 output for shared experts [sharedExpertNum × bs × hiddenDim]
            sharedExpertGmm1OutPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim, ALIGN_512);
            // sharedExpertInputData: GMM1 input data [bs × h]
            sharedExpertInputDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->h, ALIGN_512);
            // sharedExpertInputScale: GMM1 input scale [bs × CeilDiv(h,32)*2]
            sharedExpertInputScalePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs *
                                                      Ops::Base::CeilDiv(static_cast<uint32_t>(tilingData->h),
                                                                         static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM)) *
                                                      MXFP_MULTI_BASE_SIZE,
                                                  ALIGN_512);
            // sharedExpertSwigluData: SwiGLU quant output [sharedExpertNum × bs × hiddenDim/2] fp8
            sharedExpertSwigluDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum *
                                                      tilingData->hiddenDim / MegaMoeImpl::SWIGLU_N_HALF,
                                                  ALIGN_512);
            // sharedExpertSwigluScale: SwiGLU scale [sharedExpertNum × bs × hiddenDim/2/32]
            sharedExpertSwigluScalePtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim /
                                         MegaMoeImpl::SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM,
                                     ALIGN_512);
        }
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
