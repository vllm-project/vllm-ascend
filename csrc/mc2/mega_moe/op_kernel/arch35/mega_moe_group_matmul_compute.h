/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GROUP_MATMUL_COMPUTE_H
#define MEGA_MOE_GROUP_MATMUL_COMPUTE_H

#include <type_traits>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "mega_moe_base.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>; // 3: SwizzleOffset

// Waits until the input tile required by the current GMM stage is ready.
template <typename Policy, bool IsShared, typename Config>
__aicore__ inline void WaitForUpstreamReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / config.tileM;
        uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + flagOffset;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t waveStart = waveIdx * L1_TILE_M_256;
        uint32_t waveM = waveStart + L1_TILE_M_256 > config.m ? config.m - waveStart : L1_TILE_M_256;
        constexpr uint32_t sourceNFactor = Config::SOURCE_GMM1_INTERLEAVED ? SWIGLU_N_HALF : 1U;
        uint32_t targetLoops = Ops::Base::CeilDiv(waveM, config.swigluTileM) *
                               Ops::Base::CeilDiv(config.k * sourceNFactor, L1_TILE_N);
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.swigluToGmm2Flag + flagOffset;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler({config.m, config.k, config.n},
                                         BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.swigluTileM),
                                                                              static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        while (targetLoops != AscendC::ReadGmByPassDCache(gmmAddrInfo.swigluToGmm2Flag)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

// Notifies all AIV consumers assigned to the completed GMM2 token group.
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout &slotLayout,
                                                              __gm__ int32_t *expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);
    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

// Notifies the AIV consumer assigned to the completed shared-expert token group.
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset,
                                                        __gm__ int32_t *sharedExpertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

namespace Detail {

// Prototype: the GMM1 branch in AicComputeGeneric. Computes one generic GMM1 tile without
// waiting for Dispatch or notifying SwiGLU; the template pipeline controller owns those dependencies.
template <typename BlockMmad, typename ElementC, bool TopkWeightsPrefetch, bool IsGmm1Interleaved, typename Config,
          typename TensorB,
          typename TensorScaleB, typename TensorBias, typename LayoutC, typename ActualShape, typename TensorA,
          typename TensorScaleA, typename TensorC>
__aicore__ inline void ComputeGmm1TileGeneric(
    BlockMmad &blockMmad, const Config &config, TensorB &gmB, TensorScaleB &gmScaleB, TensorBias &gmBias,
    GM_ADDR gmm1OutputAddr, const LayoutC &layoutC, const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
    uint32_t kLoc, TensorA &gmBlockA, TensorScaleA &gmBlockScaleA, TensorC &l0cOutUbFirst, TensorC &l0cOutUbSecond,
    uint16_t pingpongIdx)
{
    typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                               Get<K_VALUE>(actualShape), 0};

    if constexpr (TopkWeightsPrefetch) {
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                      reinterpret_cast<__gm__ ElementC *>(gmm1OutputAddr)),
                                  layoutC);
        if constexpr (IsGmm1Interleaved) {
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = gmC.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
        } else {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                uint32_t nOffset = nLoc + weightBlock * config.outputN;
                auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nOffset),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nOffset),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                  Get<N_VALUE>(actualShape)));
                auto tensorBlockGm = gmC.Slice(
                    Te::MakeCoord(mLoc, nOffset),
                    Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockGm, singleShape);
            }
        }
    } else {
        if constexpr (IsGmm1Interleaved) {
            auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                      Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                              Get<N_VALUE>(actualShape)));
            auto tensorUb = pingpongIdx == 0U ? l0cOutUbFirst : l0cOutUbSecond;
            auto tensorBlockUb = tensorUb.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias, tensorBlockUb, singleShape);
        } else {
            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto gmBlockB = gmB.Slice(Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM),
                                  Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBias,
                          weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }
        }
    }
}

// Prototype: the GMM1 branch in AicComputeA8W4. Computes one A8W4 GMM1 tile without
// Dispatch/GMM1 or GMM1/SwiGLU synchronization. The AIV0-AIC Blaze handshake remains internal to GMM.
template <typename BlockMmad, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, typename ActualShape>
__aicore__ inline void ComputeGmm1TileA8W4(BlockMmad &blockMmad, TensorA &gmA, TensorScaleA &gmScaleA,
                                           TensorScaleB &gmScaleB, TensorC &l0cOutGm, const Config &config,
                                           const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc)
{
    auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
    auto gmBlockScaleA =
        gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

    for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
        uint32_t nOffset = nLoc + weightBlock * config.outputN;
        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
    }
}

// Executes the generic GMM2 tile loop. GMM1 uses the dependency-free tile leaf above.
template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC, bool IsShared,
          bool TopkWeightsPrefetch, bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGmm2Generic(BlockMmad &blockMmad, WorkSet &workSet, uint32_t startLoopIdx,
                                             uint32_t tileNum, ExtraArgs &args)
{
    static_assert(!Policy::IS_GMM1, "GMM1 must use ControlGmm1SwigluPipelineGeneric");
    const auto &config = workSet.config;
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        if constexpr (!std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            logicalCoreCount *= 2U;
        }
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));

        if constexpr (std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB = workSet.gmScaleB.Slice(
            Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
            Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                      reinterpret_cast<__gm__ ElementC *>(workSet.gmmAddrInfo.gmm2OutGlobal)),
                                  workSet.layouts.c);
        auto gmBlockC = gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                  Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout,
                                                   workSet.gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, workSet.gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// Executes the A8W4 GMM2 tile loop. GMM1 uses the dependency-free tile leaf above.
template <uint8_t CombineQuantMode, typename Policy, bool IsShared, bool IsLayered = false, typename BlockMmad,
          typename Scheduler, typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC,
          typename Config, bool TopkWeightsPrefetch>
__aicore__ inline void AicComputeGmm2A8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA,
                                          TensorScaleA &gmScaleA, TensorScaleB &gmScaleB, TensorC &l0cOutGm,
                                          const GMMAddrInfo &gmmAddrInfo, int32_t &gmTileSequence,
                                          const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    static_assert(!Policy::IS_GMM1, "GMM1 must use ControlGmm1SwigluPipelineA8W4");
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if (loopIdx == startLoopIdx) {
            WaitForUpstreamReady<Policy, IsShared>(gmmAddrInfo, config, mLoc);
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
        constexpr bool hasAiv1GmEpilogue = CombineQuantMode == COMBINE_NO_QUANT && !IsShared && !IsLayered;
        if constexpr (hasAiv1GmEpilogue) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++gmTileSequence);
        }
    }
}

} // namespace Detail
} // namespace MegaMoeImpl

#endif // MEGA_MOE_GROUP_MATMUL_COMPUTE_H
