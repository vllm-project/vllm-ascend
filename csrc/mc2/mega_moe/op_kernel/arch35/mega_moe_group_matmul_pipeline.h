/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GROUP_MATMUL_PIPELINE_H
#define MEGA_MOE_GROUP_MATMUL_PIPELINE_H

#include "mega_moe_group_matmul_compute.h"
#include "mega_moe_group_matmul_prologue.h"
#include "mega_moe_group_matmul_epilogue.h"

namespace MegaMoeImpl {
namespace Detail {

__aicore__ inline void WaitGmm1TileStatus(const Params &params, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                                 (static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx) *
                                     INT_CACHELINE;
    int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
    while (AscendC::ReadGmByPassDCache(statusAddr) != roundTag) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < 100) {
        }
    }
}

__aicore__ inline void NotifyGmm1TileStatus(const Params &params, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm1TileStatusPtr) +
                                 (static_cast<int64_t>(expertIdx) * params.tilingData->maxTilesPerExpert + loopIdx) *
                                     INT_CACHELINE;
    AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
}

// Controls the current generic GMM1/SwiGLU tile pipeline. Cross-stage waits remain here;
// ComputeSwigluTileGeneric publishes each completed tile immediately after the dependency-free
// SwiGLU operator returns so the original GMM1/GMM2 overlap is preserved.
template <typename Policy, typename BlockMmad, typename ElementC, typename MakeLayoutC, bool TopkWeightsPrefetch,
          bool IsShared, bool IsGmm1Interleaved, typename WorkSet, typename ExtraArgs>
__aicore__ inline void ControlGmm1SwigluPipelineGeneric(WorkSet &workSet, uint32_t startLoopIdx,
                                                        uint32_t tileNum, ExtraArgs &args)
{
    const auto &config = workSet.config;
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.n, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params, false, enableL0CPingPong);

        uint32_t ubBufSize =
            config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
        auto l0cOutUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
        auto l0cOutUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);
        uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t kLoc = Get<K_VALUE>(blockCoord);
            auto gmBlockA = workSet.gmA.Slice(
                Te::MakeCoord(mLoc, kLoc),
                Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
            auto gmBlockScaleA = workSet.gmScaleA.Slice(
                Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if constexpr (!TopkWeightsPrefetch) {
                if constexpr (IsGmm1Interleaved) {
                    if (args.vecSetSyncCom >= 2) {
                        WaitForVector(args.pingpongIdx);
                    }
                } else if (args.vecSetSyncCom != 0) {
                    WaitForVector();
                }
            }

            ComputeGmm1TileGeneric<BlockMmad, ElementC, TopkWeightsPrefetch, IsGmm1Interleaved>(
                blockMmad, config, workSet.gmB, workSet.gmScaleB, workSet.gmBias,
                workSet.gmmAddrInfo.gmm1OutGlobal, workSet.layouts.c, actualShape, mLoc, nLoc, kLoc, gmBlockA,
                gmBlockScaleA, l0cOutUbFirst, l0cOutUbSecond, args.pingpongIdx);

            if constexpr (TopkWeightsPrefetch) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                NotifyGmm1TileStatus(workSet.params, args.expertIdx, loopIdx);
            } else {
                if constexpr (IsGmm1Interleaved) {
                    NotifyVector(args.pingpongIdx);
                    args.vecSetSyncCom++;
                    args.pingpongIdx = 1U - args.pingpongIdx;
                } else {
                    NotifyVector();
                    args.vecSetSyncCom = 1;
                }
            }
        }
    } else {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            if constexpr (TopkWeightsPrefetch) {
                WaitGmm1TileStatus(workSet.params, args.expertIdx, loopIdx);
            } else if constexpr (IsGmm1Interleaved) {
                WaitForCube(args.pingpongIdx);
            } else {
                WaitForCube();
            }

            ComputeSwigluTileGeneric<MakeLayoutC, ElementC, TopkWeightsPrefetch, IsGmm1Interleaved>(
                config, workSet.gmmAddrInfo.gmm1OutGlobal, workSet.layouts.c, args.swigluQuantOp,
                args.expertBeforeCnt, actualShape, mLoc, nLoc, args.pingpongIdx);

            if constexpr (!TopkWeightsPrefetch) {
                if constexpr (IsGmm1Interleaved) {
                    NotifyCube(args.pingpongIdx);
                    args.pingpongIdx = 1U - args.pingpongIdx;
                } else {
                    NotifyCube();
                }
            }
        }
    }
}

// Prototype: GroupMatmulExecGeneric. Dispatches one generic GMM stage to its AIC or AIV implementation.
template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, typename MakeLayoutC,
          bool TopkWeightsPrefetch, bool IsLayered = false, bool IsShared = false,
          bool IsGmm1Interleaved = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(WorkSet &workSet, uint32_t startLoopIdx, uint32_t tileNum,
                                              ExtraArgs &args)
{
    if constexpr (Policy::IS_GMM1) {
        ControlGmm1SwigluPipelineGeneric<Policy, BlockMmad, ElementC, MakeLayoutC, TopkWeightsPrefetch, IsShared,
                                         IsGmm1Interleaved>(workSet, startLoopIdx, tileNum, args);
    } else if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGmm2Generic<CombineQuantMode, Policy, BlockMmad, ElementC, IsShared, TopkWeightsPrefetch, IsLayered>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    }
}

// Controls the current A8W4 GMM1/SwiGLU tile pipeline. The AIV0 weight prologue and
// AIC MMAD keep their Blaze-internal handshake; stage-boundary synchronization and
// downstream completion publication are owned here.
template <typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC, typename MakeLayoutC,
          bool IsShared, typename WorkSet, typename Config, bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void ControlGmm1SwigluPipelineA8W4(WorkSet &workSet, const Params &params,
                                                     const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                                     uint32_t startLoopIdx, uint32_t tileNum,
                                                     int32_t &gmTileSequence, ExtraArgs &args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            uint32_t waveIdx = mLoc / config.tileM;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }

            ComputeGmm1TileA8W4(blockMmad, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB,
                                workSet.l0cOutGm, config, actualShape, mLoc, nLoc);

            if constexpr (TopkWeightsPrefetch) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                NotifyGmm1TileStatus(params, args.expertIdx, loopIdx);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmTileSequence);
            }
        }
    } else if (GetSubBlockIdx() == 0) {
        BlockPrologue blockPrologue;
        AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
    } else {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
            auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
            auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
            uint32_t mLoc = Get<M_VALUE>(blockCoord);
            uint32_t nLoc = Get<N_VALUE>(blockCoord);
            int32_t expectedReadySequence = gmTileSequence;
            if constexpr (TopkWeightsPrefetch) {
                WaitGmm1TileStatus(params, args.expertIdx, loopIdx);
            } else {
                expectedReadySequence = gmTileSequence + 1;
                while (AscendC::ReadGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag) < expectedReadySequence) {
                    int64_t startCycle = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - startCycle < 100) {
                    }
                }
            }

            ComputeSwigluTileA8W4<ElementC, MakeLayoutC, decltype(workSet.l0cOutGm),
                                  decltype(args.swigluQuantOp), Config, TopkWeightsPrefetch>(
                args.swigluQuantOp, workSet.l0cOutGm, config, actualShape, mLoc, nLoc, args.expertBeforeCnt);
            if constexpr (!TopkWeightsPrefetch) {
                gmTileSequence = expectedReadySequence;
            }
        }
    }
}

// Prototype: GroupMatmulExecA8W4. Dispatches one A8W4 GMM stage across the block's AIC and two AIVs.
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
          typename MakeLayoutC, bool IsShared, bool IsLayered = false, typename WorkSet, typename Config,
          bool TopkWeightsPrefetch, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet &workSet, const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                           const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                           int32_t &gmTileSequence, ExtraArgs &args)
{
    if constexpr (Policy::IS_GMM1) {
        ControlGmm1SwigluPipelineA8W4<Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC, IsShared,
                                      WorkSet, Config, TopkWeightsPrefetch>(
            workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, gmTileSequence, args);
    } else if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        AicComputeGmm2A8W4<CombineQuantMode, Policy, IsShared, IsLayered, BlockMmad, decltype(workSet.scheduler),
                           decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                           decltype(workSet.l0cOutGm), Config, TopkWeightsPrefetch>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.l0cOutGm,
            gmmAddrInfo, gmTileSequence, config, startLoopIdx, tileNum);
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else {
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered) {
                AivGmm2PostA8W4<ElementC, MakeLayoutC>(workSet.scheduler, workSet.l0cOutGm, params, gmmAddrInfo,
                                                       gmTileSequence, args.groupCnt, config, startLoopIdx, tileNum);
            }
        }
    }
}

} // namespace Detail
} // namespace MegaMoeImpl

#endif // MEGA_MOE_GROUP_MATMUL_PIPELINE_H
