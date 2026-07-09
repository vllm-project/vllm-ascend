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
 * \file mega_moe_impl.h
 * \brief
 */

#ifndef MEGA_MOE_IMPL_H
#define MEGA_MOE_IMPL_H
#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"

#include "tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_swizzle.h"
#include "blaze/gemm/block/block_mmad_mx_fp8fp4.h"
#include "blaze/prologue/block_prologue_mx_fp8fp4.h"

#include "mega_moe_impl_base.h"
#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
using BlockScheduler = typename Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;  // 3: SwizzleOffset
// =================================================================================================
// ComputeCoreGrouping：计算当前 core 所属的 group 及其在 group 内的位置
// =================================================================================================
// 将 totalCores 个 core 均匀分配到 numGroups 个 group 中，余数分配给前 remainder 个 group。
__aicore__ inline void ComputeCoreGrouping(uint32_t coreId, uint32_t numGroups, uint32_t totalCores,
    uint32_t& myGroup, uint32_t& myIdxInGrp, uint32_t& myGrpSize)
{
    uint32_t baseSize = totalCores / numGroups;      // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;     // 余数，前 remainder 个 group 多分配 1 个 core
    uint32_t boundary = remainder * (baseSize + 1);  // 前 remainder 个 group 占用的 core 总数
    
    // 判断当前 core 是否在前 remainder 个 group 中（这些 group 有 baseSize+1 个 core）
    if (coreId < boundary) {
        myGroup = coreId / (baseSize + 1);           // 所属 group 索引
        myIdxInGrp = coreId % (baseSize + 1);        // 在 group 内的索引
        myGrpSize = baseSize + 1;                    // 当前 group 的 core 数
    } else {
        // 当前 core 在后面的 group 中（这些 group 只有 baseSize 个 core）
        uint32_t adjusted = coreId - boundary;       // 减去前 remainder 个 group 占用的 core 数
        myGroup = remainder + adjusted / baseSize;   // 所属 group 索引 = remainder + 偏移
        myIdxInGrp = adjusted % baseSize;            // 在 group 内的索引
        myGrpSize = baseSize;                        // 当前 group 的 core 数
    }
}

// =================================================================================================
// ComputeGroupRange：计算指定 group 包含的 core 范围
// =================================================================================================
// ComputeCoreGrouping 的逆操作：给定 groupIdx，返回该 group 的起始 core 和 core 数量。
// 用于 GMM2 量化路径中，AIC 计算完一个 tile 后，通知负责该 token group 的所有 AIV core。
__aicore__ inline void ComputeGroupRange(uint32_t groupIdx, uint32_t numGroups, uint32_t totalCores,
    uint32_t& grpCoreStart, uint32_t& grpCoreSize)
{
    uint32_t baseSize = totalCores / numGroups;      // 每个 group 的基础 core 数
    uint32_t remainder = totalCores % numGroups;     // 余数，前 remainder 个 group 多分配 1 个 core
    
    if (groupIdx < remainder) {
        // 当前 group 在前 remainder 个 group 中，有 baseSize+1 个 core
        grpCoreSize = baseSize + 1;
        grpCoreStart = groupIdx * (baseSize + 1);  // 起始 core = groupIdx * (baseSize+1)
    } else {
        // 当前 group 在后面的 group 中，只有 baseSize 个 core
        grpCoreSize = baseSize;
        // 起始 core = 前 remainder 个 group 占用的 core 数 + 偏移
        grpCoreStart = remainder * (baseSize + 1) + (groupIdx - remainder) * baseSize;
    }
}

// =================================================================================================
// NotifyCombineTileComplete：AIC 完成一个 tile 后，通过 AtomicAdd 通知负责该 token group 的 AIV
// =================================================================================================
// counterPtr 内存布局: 每个 expert 一段, 每段 blockAivNum 个 slot, 每个 slot = INT_CACHELINE int32
//   expert:  [slot_0][slot_1][slot_2]...[slot_n]
// IsA8W4=false (A8W8/A4W4): 所有 blockAivNum 个核参与, 逻辑索引 = 物理 ID
// IsA8W4=true  (A8W4):    仅 sub=1 的核参与, 逻辑索引 i 映射为物理 ID = i*2+1
template <bool IsA8W4 = false>
__aicore__ inline void NotifyCombineTileComplete(
    uint32_t mLoc, uint32_t m, uint32_t tileM,
    uint32_t blockAivNum, uint32_t groupIdx,
    __gm__ int32_t* counterPtr)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
    uint32_t participatingCores = IsA8W4 ? (blockAivNum / 2) : blockAivNum;
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m, tileM);
    uint32_t tokenGroupIdx = mLoc / tileM;
    uint32_t grpCoreStart = 0, grpCoreSize = 0;
    ComputeGroupRange(tokenGroupIdx, tokenGroupsThisExpert, participatingCores, grpCoreStart, grpCoreSize);
    int64_t baseOffset = static_cast<int64_t>(groupIdx) * blockAivNum * INT_CACHELINE;
    for (uint32_t i = grpCoreStart; i < grpCoreStart + grpCoreSize; i++) {
        uint32_t physicalId = IsA8W4 ? (i * 2 + 1) : i;
        AscendC::AtomicAdd(counterPtr + baseOffset + physicalId * INT_CACHELINE, int32_t(1));
    }
}

// =================================================================================================
// WaitForUpstreamReady：等待上游 GMM 计算完成，GMM1/GMM2 分流（A8W8/A4W4 和 A8W4 共用）
// =================================================================================================
template <typename Policy, typename Config>
__aicore__ inline void WaitForUpstreamReady(
    const GMMAddrInfo& gmmAddrInfo, const Config& config, uint32_t mLoc)
{
    if constexpr (Policy::IS_GMM1) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t targetValue = (mLoc + L1_TILE_M_256 > config.m) ? (config.m - mLoc) : L1_TILE_M_256;
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + waveIdx;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    } else {
        BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256),
                static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        __gm__ int32_t* flagValueAddr = gmmAddrInfo.swigluToGmm2Flag;
        while (targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }
}

// ==================================================================================
// 统一配置结构体 — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4 两条路径的配置
// ==================================================================================
namespace Detail {
struct Gmm1Policy {
    static constexpr bool IS_GMM1 = true;
};

struct Gmm2Policy {
    static constexpr bool IS_GMM1 = false;
};

// BlockMmadSelector — 通过偏特化处理 A8W8/A4W4 和 A8W4 的 BlockMmad 签名差异
template <bool IsA8W4, typename C>
struct BlockMmadSelector;

template <typename C>
struct BlockMmadSelector<false, C> {
    using type = Blaze::Gemm::Block::BlockMmad<
        typename C::DispatchPolicy,
        typename C::ElementAType, typename C::LayoutA,
        typename C::ElementBType, typename C::LayoutB,
        typename C::ElementCType, typename C::LayoutC,
        typename C::BiasType, typename C::LayoutBias>;
};

template <typename C>
struct BlockMmadSelector<true, C> {
    using type = Blaze::Gemm::Block::BlockMmad<
        typename C::DispatchPolicy,
        AscendC::Std::tuple<typename C::ElementAType, typename C::ElementMxScaleAType>,
        AscendC::Std::tuple<typename C::MakeLayoutA, typename C::MakeLayoutScaleA>,
        AscendC::Std::tuple<typename C::ElementBType, typename C::ElementMxScaleBType>,
        AscendC::Std::tuple<typename C::MakeLayoutB, typename C::MakeLayoutScaleB>,
        typename C::ElementCType, typename C::MakeLayoutC, void, void>;
};

// ==================================================================================
// 统一 Config — 通过 IsA8W4 模板参数区分 A8W8/A4W4 和 A8W4
// 含公共与差异类型别名，BlockMmad 通过 trait 选择
// ==================================================================================
template <bool IsA8W4, typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
    typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false>
struct Config {
    using ElementAType = ElementA;
    using ElementBType = ElementB;
    using ElementCType = ElementC;
    using ElementMxScaleAType = ElementMxScaleA;
    using ElementMxScaleBType = ElementMxScaleB;

    static constexpr uint32_t C0_SIZE_A = AuxGetC0Size<ElementA>();
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    static constexpr uint32_t C0_SIZE_SCALE = 2U;

    static constexpr uint32_t C0_SIZE_B = IsA8W4 ? 32U : AuxGetC0Size<ElementB>();

    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutScaleA = Te::ScaleANDLayoutPtn;
    using LayoutScaleB = Te::ScaleBDNLayoutPtn;

    using BiasType = float;
    using LayoutBias = Te::NDExtLayoutPtn;
    using DispatchPolicy = Std::conditional_t<IsA8W4,
        Blaze::Gemm::MatmulMxFp8Fp4DynamicKL1TailResplit,
        Blaze::Gemm::MatmulWithScaleMx<>>;
    using LayoutB = Std::conditional_t<IsA8W4,
        Te::ZNLayoutPtn,
        Std::conditional_t<IsWeightNZ, Te::ZNLayoutPtn, Te::DNExtLayoutPtn>>;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE_A>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleA = Te::FrameLayoutFormat<LayoutScaleA, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutScaleB = Te::FrameLayoutFormat<LayoutScaleB, Std::Int<C0_SIZE_SCALE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutAType = decltype(MakeLayoutA{}(uint32_t {}, uint32_t {}));
    using LayoutBType = decltype(MakeLayoutB{}(uint32_t {}, uint32_t {}));
    using LayoutScaleAType = decltype(MakeLayoutScaleA{}(uint32_t {}, uint32_t {}));
    using LayoutScaleBType = decltype(MakeLayoutScaleB{}(uint32_t {}, uint32_t {}));
    using LayoutCType = decltype(MakeLayoutC{}(uint32_t {}, uint32_t {}));
    using LayoutBiasType = decltype(Te::MakeFrameLayout<LayoutBias>(uint32_t {}, uint32_t {}));

    using BlockMmad = typename BlockMmadSelector<IsA8W4, Config>::type;

    // BlockPrologue（仅 A8W4 使用；A8W8/A4W4 路径用 void 占位）
    using BlockPrologue = Std::conditional_t<IsA8W4,
        Blaze::Gemm::Prologue::BlockPrologue<DispatchPolicy, ElementA, ElementB>,
        void>;

    struct ProblemConfig {
        static __aicore__ inline typename BlockMmad::L1Params DefaultL1Params()
        {
            if constexpr (IsA8W4) {
                return typename BlockMmad::L1Params{.kL1 = L1_TILE_K, .scaleKL1 = 4096};
            } else {
                return typename BlockMmad::L1Params{
                    .kL1 = L1_TILE_K, .scaleKL1 = L1_TILE_K * SCALE_K_L1_RATE, .l1BufNum = 2};
            }
        }

        uint32_t m = 0;
        uint32_t n = 0;
        uint32_t k = 0;
        uint32_t outputN = 0;
        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t scaleK = 0;
        uint32_t tileM = 0;        // A8W8/A4W4 路径用
        typename BlockMmad::L1Params l1Params = DefaultL1Params();
    };

    struct LayoutBundle {
        LayoutAType a;
        LayoutBType b;
        LayoutScaleAType scaleA;
        LayoutScaleBType scaleB;
        LayoutCType c;
        LayoutBiasType bias;  // A8W8/A4W4 路径用，A8W4 不使用
    };

    __aicore__ static inline ProblemConfig BuildProblemConfig(const ProblemShape& problemShape)
    {
        ProblemConfig config;
        config.m = Get<M_VALUE>(problemShape);
        if constexpr (Policy::IS_GMM1) {
            config.n = Get<N_VALUE>(problemShape);
            config.k = Get<K_VALUE>(problemShape);
        } else {
            config.n = Get<K_VALUE>(problemShape);
            config.k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
        }
        config.outputN = Policy::IS_GMM1 ? config.n / SWIGLU_N_HALF : config.n;
        config.blockNum = GetBlockNum();
        config.blockIdx = GetBlockIdx() / GetTaskRation();
        config.scaleK = CeilDiv(config.k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        if constexpr (!IsA8W4) {
            if constexpr (Policy::IS_GMM1) {
                config.tileM = L1_TILE_M_256;
            } else {
                config.tileM = (CombineQuantMode == COMBINE_NO_QUANT) ? L1_TILE_M_128 : L1_TILE_M_256;
            }
        }
        return config;
    }

    __aicore__ static inline LayoutBundle BuildLayouts(const ProblemConfig& config)
    {
        LayoutBundle layouts;
        layouts.a = MakeLayoutA{}(config.m, config.k);
        layouts.b = MakeLayoutB{}(config.k, config.n);
        layouts.scaleA = MakeLayoutScaleA{}(config.m, config.scaleK);
        layouts.scaleB = MakeLayoutScaleB{}(config.scaleK, config.n);
        if constexpr (IsA8W4) {
            layouts.c = MakeLayoutC{}(config.m, config.n);
        } else {
            layouts.bias = Te::MakeFrameLayout<LayoutBias>(1U, config.n);
            if constexpr (Policy::IS_GMM1) {
                layouts.c = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    layouts.c = MakeLayoutC{}(L1_TILE_M_128, L1_TILE_N);
                } else {
                    layouts.c = MakeLayoutC{}(config.m, config.n);
                }
            }
        }
        return layouts;
    }
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename ElementC,
    typename WorkSet, typename ExtraArgs>
__aicore__ inline void AicComputeGeneric(
    BlockMmad& blockMmad, WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    constexpr uint32_t ubBufSize = Policy::IS_GMM1
        ? MAX_SINGLE_MN_ALIGN32_NUM_256
        : ((CombineQuantMode == COMBINE_NO_QUANT) ? MAX_SINGLE_MN_ALIGN32_NUM_128 : 0);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    auto l0cOutUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), workSet.layouts.c);
    auto l0cOutUbSecond = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), workSet.layouts.c);

    const auto& config = workSet.config;
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        if constexpr (Policy::IS_GMM1) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForUpstreamReady<Policy>(workSet.gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
            if (args.vecSetSyncCom) {
                WaitForVector();
            }
        } else {
            if (loopIdx == startLoopIdx) {
                WaitForUpstreamReady<Policy>(workSet.gmmAddrInfo, config, mLoc);
            }
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                if (args.vecSetSyncCom2 >= 2) {
                    WaitForVector(args.pingpongIdx);
                }
            }
        }

        auto gmBlockA = workSet.gmA.Slice(
            Te::MakeCoord(mLoc, kLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        auto gmBlockScaleA = workSet.gmScaleA.Slice(
            Te::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
            Te::MakeShape(Get<M_VALUE>(actualShape), CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM)));
        typename BlockMmad::BlockShape singleShape{
            Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), Get<K_VALUE>(actualShape), 0};

        if constexpr (Policy::IS_GMM1) {
            auto tensorBlockUbFirst = l0cOutUbFirst.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto tensorBlockUbSecond = l0cOutUbSecond.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto gmBlockB = workSet.gmB.Slice(
                    Te::MakeCoord(kLoc, nLoc + weightBlock * config.outputN),
                    Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                auto gmBlockScaleB = workSet.gmScaleB.Slice(
                    Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * config.outputN),
                    Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias,
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, singleShape);
            }
            NotifyVector();
            args.vecSetSyncCom = 1;
        } else {
            auto gmBlockB = workSet.gmB.Slice(
                Te::MakeCoord(kLoc, nLoc), Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto gmBlockScaleB = workSet.gmScaleB.Slice(
                Te::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                Te::MakeShape(CeilDiv(Get<K_VALUE>(actualShape), MXFP_SCALE_GROUP_NUM), Get<N_VALUE>(actualShape)));
            if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                auto tensorUb = args.pingpongIdx == 0 ? l0cOutUbFirst : l0cOutUbSecond;
                auto tensorBlockUb = tensorUb.Slice(
                    Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockUb, singleShape);
                NotifyVector(args.pingpongIdx);
                args.vecSetSyncCom2++;
                args.pingpongIdx = 1 - args.pingpongIdx;
            } else {
                auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                    reinterpret_cast<__gm__ ElementC*>(workSet.gmmAddrInfo.gmm2OutGlobal)), workSet.layouts.c);
                auto gmBlockC = gmC.Slice(
                    Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
                NotifyCombineTileComplete(mLoc, config.m, L1_TILE_M_256, config.blockNum * 2,
                    args.groupIdx, (__gm__ int32_t*)workSet.params.workspaceInfo.gmm2CombineSyncCounterPtr);
            }
        }
    }
}

template <typename WorkSet, typename ExtraArgs>
__aicore__ inline void AivGmm1PostGeneric(
    WorkSet& workSet, ExtraArgs& args, uint32_t startLoopIdx, uint32_t tileNum)
{
    const auto& config = workSet.config;
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
            Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + nLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0, 0, 0, 0};
        WaitForCube();
        AscendC::SetCtrlSpr<60, 60>(0);
        args.swigluQuantOp(epilogueShape, epilogueOffset);
        NotifyCube();
    }
}

template <typename ElementC, bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AivGmm2PostGeneric(
    WorkSet& workSet, ExtraArgs& args, uint32_t startLoopIdx, uint32_t tileNum)
{
    constexpr uint32_t ubBufSize = MAX_SINGLE_MN_ALIGN32_NUM_128;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2First(TPosition::VECIN, ubOffsetFirst, L1_TILE_M_128 * L1_TILE_N);
    LocalTensor<ElementC> l0cOutUbGMM2Second(TPosition::VECIN, ubOffsetSecond, L1_TILE_M_128 * L1_TILE_N);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += workSet.config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        auto l0cOutUbGMM2 = args.pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
        WaitForCube(args.pingpongIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::GlobalTensor<int32_t> tripleGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(
            TPosition::VECCALC, TRIPLE_TENSOR_ADDR, lenTile * TRIPLE_SIZE);
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workSet.params.workspaceInfo.tripleInfoPtr +
            (args.groupCnt + mLoc) * TRIPLE_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(tripleTensor, tripleGm, lenTile * TRIPLE_SIZE);
        if constexpr (IsLayered) {
            MegaMoeCombineImpl::CombineTokensLayered<ElementC, decltype(actualShape)>(
                mLoc, nLoc, workSet.config.n, tripleTensor, l0cOutUbGMM2, actualShape, workSet.params);
        } else {
            MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
                mLoc, nLoc, workSet.config.n, tripleTensor, l0cOutUbGMM2, actualShape, workSet.params);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        NotifyCube(args.pingpongIdx);
        args.pingpongIdx = 1 - args.pingpongIdx;
    }
}

template <typename SwigluQuantOp>
struct Gmm1ArgsGeneric {
    SwigluQuantOp& swigluQuantOp;
    int32_t& vecSetSyncCom;
};

struct Gmm2ArgsGeneric {
    int32_t& vecSetSyncCom2;
    uint32_t groupCnt;
    uint16_t& pingpongIdx;
    uint32_t groupIdx;
};

template <typename Scheduler, typename TensorA, typename TensorB,
    typename TensorScaleA, typename TensorScaleB, typename TensorBias,
    typename Config, typename LayoutBundle>
struct WorkSetGeneric {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorBias& gmBias;
    const GMMAddrInfo& gmmAddrInfo;
    const Params& params;
    const Config& config;
    const LayoutBundle& layouts;
};

template <typename Policy, uint8_t CombineQuantMode, typename BlockMmad, typename ElementC,
    bool IsLayered = false, typename WorkSet, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecGeneric(
    WorkSet& workSet, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad;
        bool enableL0CPingPong = false;
        typename BlockMmad::BlockShape l0TileShape{workSet.config.tileM, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{workSet.config.m, workSet.config.n, workSet.config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, workSet.config.l1Params, false, enableL0CPingPong);

        AicComputeGeneric<CombineQuantMode, Policy, BlockMmad, ElementC>(
            blockMmad, workSet, startLoopIdx, tileNum, args);
    } else {
        if constexpr (Policy::IS_GMM1) {
            AivGmm1PostGeneric(workSet, args, startLoopIdx, tileNum);
        } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            AivGmm2PostGeneric<ElementC, IsLayered>(workSet, args, startLoopIdx, tileNum);
        }
        // GMM2 量化模式：AIV 不经此路径，由 ProcessCombine 独立处理
    }
}

template <typename Policy, uint8_t CombineQuantMode, typename ElementA, typename ElementB,
    typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
    bool IsLayered = false, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplGeneric(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, ExtraArgs& args)
{
    using Config = Config<false, Policy, CombineQuantMode, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ>;
    auto config = Config::BuildProblemConfig(problemShape);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;

    auto layouts = Config::BuildLayouts(config);

    if constexpr (Policy::IS_GMM1) {
        if (GetSubBlockIdx() != 0) {
            startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
            return;
        }
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    } else if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (GetSubBlockIdx() != 0) return;
    }
    // GMM2 量化模式：两分支均不匹配，直接往下执行

    using BlockMmad = typename Config::BlockMmad;
    using BiasType = typename Config::BiasType;

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(0UL)), layouts.bias);

    using WorkSetType = WorkSetGeneric<BlockScheduler, decltype(gmA), decltype(gmB),
        decltype(gmScaleA), decltype(gmScaleB), decltype(gmBias),
        decltype(config), decltype(layouts)>;
    WorkSetType workSet{
        scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias,
        gmmAddrInfo, params, config, layouts
    };

    GroupMatmulExecGeneric<Policy, CombineQuantMode, BlockMmad, ElementC, IsLayered>(
        workSet, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}
} // namespace Detail

// =================================================================================================
// GroupMatmulSwigluQuant：GMM1 矩阵乘法 + SwiGLU 激活 + 量化
// =================================================================================================
template <typename ElementA, typename EpilogueElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false>
__aicore__ inline void GroupMatmulSwigluQuant(
    BlockEpilogueSwigluMxQuant<EpilogueElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    using SwigluQuantOpType = std::remove_reference_t<decltype(epilogueOp)>;
    Detail::Gmm1ArgsGeneric<SwigluQuantOpType> args{epilogueOp, vecSetSyncCom};
    Detail::GroupMatmulImplGeneric<Detail::Gmm1Policy, COMBINE_NO_QUANT, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ>(params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// =================================================================================================
// GroupMatmul2：GMM2 矩阵乘法，支持量化和非量化模式
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
           typename ElementMxScaleA, typename ElementMxScaleB, bool IsWeightNZ = false,
           bool IsLayered = false>
__aicore__ inline void GroupMatmul2(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2, uint32_t groupCnt, uint16_t& pingpongIdx, uint32_t groupIdx)
{
    Detail::Gmm2ArgsGeneric args{vecSetSyncCom2, groupCnt, pingpongIdx, groupIdx};
    Detail::GroupMatmulImplGeneric<Detail::Gmm2Policy, CombineQuantMode, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB, IsWeightNZ, IsLayered>(
        params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// ==================================================================================
// A8W4 执行路径 — 共享骨架，基于 Policy 分派 GMM1 / GMM2
// ==================================================================================
namespace Detail {

template <typename SwigluQuantOp>
struct Gmm1ArgsA8W4 {
    SwigluQuantOp& swigluQuantOp;
    uint32_t groupIdx = 0;
};

struct Gmm2ArgsA8W4 {
    uint32_t groupCnt;
    uint16_t& pingpongIdx;
    uint32_t groupIdx = 0;
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename Scheduler, typename TensorA,
    typename TensorScaleA, typename TensorScaleB, typename TensorC, typename Config>
__aicore__ inline void AicComputeA8W4(
    BlockMmad& blockMmad, Scheduler& scheduler, TensorA& gmA, TensorScaleA& gmScaleA, TensorScaleB& gmScaleB,
    TensorC& l0cOutGm, const GMMAddrInfo& gmmAddrInfo, const Config& config,
    uint32_t startLoopIdx, uint32_t tileNum,
    uint32_t groupIdx = 0, __gm__ int32_t* gmm2CombineSyncCounterPtr = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        bool shouldWait = false;
        uint32_t waveIdx = 0;
        if constexpr (Policy::IS_GMM1) {
            waveIdx = mLoc / L1_TILE_M_256;
            shouldWait = waveIdx != lastWaveWaited;
        } else {
            shouldWait = loopIdx == startLoopIdx;
        }
        if (shouldWait) {
            WaitForUpstreamReady<Policy>(gmmAddrInfo, config, mLoc);
            if constexpr (Policy::IS_GMM1) {
                lastWaveWaited = waveIdx;
            }
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                auto gmBlockScaleB =
                    gmScaleB.Slice(Te::MakeCoord(0, nOffset), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
                auto tensorBlockGm = l0cOutGm.Slice(
                    Te::MakeCoord(mLoc, nOffset), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
                blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            }
        } else {
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
            auto tensorBlockGm = l0cOutGm.Slice(
                Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
            if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                NotifyCombineTileComplete<true>(mLoc, config.m, L1_TILE_M_256,
                    config.blockNum * 2, groupIdx, gmm2CombineSyncCounterPtr);
            }
        }
        NotifyVectorToCopyIn();
    }
}

template <typename Policy, typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void AivPrologueA8W4(BlockPrologue& blockPrologue, Scheduler& scheduler, TensorB& gmB,
    const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);

        if constexpr (Policy::IS_GMM1) {
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto nOffset = nLoc + weightBlock * config.outputN;
                blockPrologue(gmB, mL1Size, config.k, nL1Size, nOffset, config.n, config.l1Params.kL1);
            }
        } else {
            blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.l1Params.kL1);
        }
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename SwigluQuantOp,
    typename Config>
__aicore__ inline void AivGmm1PostA8W4(SwigluQuantOp& swigluQuantOp, Scheduler& scheduler,
    TensorC& l0cOutGm, const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        WaitForCubeFinishCopyout();
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
        auto tensorBlockGmFirst = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc + config.outputN),
            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));

        auto layoutL0cUB = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = ubOffsetFirst + MAX_SINGLE_MN_ALIGN32_NUM_256 * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUB);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUB);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGM2UB, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                     Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + nLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0, 0, 0, 0};

        AscendC::SetCtrlSpr<60, 60>(0);
        swigluQuantOp(epilogueShape, epilogueOffset);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    }
}

template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void AivGmm2PostA8W4(Scheduler& scheduler, TensorC& l0cOutGm, const Params& params,
    uint32_t groupCnt, const Config& config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        WaitForCubeFinishCopyout();
        auto tensorBlockGm = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUB = MakeLayoutC{}(L1_TILE_M_256, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        LocalTensor<ElementC> l0cOutUbGMM2 =
            LocalTensor<ElementC>(TPosition::VECIN, ubOffset, L1_TILE_M_256 * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> tripleGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECCALC, 200 * 1024, lenTile * 8);
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.tripleInfoPtr +
            (groupCnt + mLoc) * 32));
        AscendC::DataCopy(tripleTensor, tripleGm, lenTile * 8);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            mLoc, nLoc, config.n, tripleTensor, l0cOutUbGMM2, actualShape, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
}

template <typename Scheduler, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
    typename TensorC>
struct WorkSetA8W4 {
    Scheduler& scheduler;
    TensorA& gmA;
    TensorB& gmB;
    TensorScaleA& gmScaleA;
    TensorScaleB& gmScaleB;
    TensorC& l0cOutGm;
};

template <uint8_t CombineQuantMode, typename Policy, typename BlockMmad, typename BlockPrologue, typename ElementC,
    typename MakeLayoutC, typename WorkSet, typename Config, typename ExtraArgs>
__aicore__ inline void GroupMatmulExecA8W4(WorkSet& workSet, const Params& params, const GMMAddrInfo& gmmAddrInfo,
    const Config& config, uint32_t startLoopIdx, uint32_t tileNum, ExtraArgs& args)
{
    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{L1_TILE_M_256, L1_TILE_N, L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.l1Params);
        AicComputeA8W4<CombineQuantMode, Policy>(blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA,
            workSet.gmScaleB, workSet.l0cOutGm, gmmAddrInfo, config, startLoopIdx, tileNum,
            args.groupIdx, (__gm__ int32_t*)params.workspaceInfo.gmm2CombineSyncCounterPtr);
    } else {
        if (GetSubBlockIdx() == 0) {
            BlockPrologue blockPrologue;
            AivPrologueA8W4<Policy>(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx,
                tileNum);
        } else {
            if constexpr (Policy::IS_GMM1) {
                AivGmm1PostA8W4<ElementC, MakeLayoutC>(
                    args.swigluQuantOp, workSet.scheduler, workSet.l0cOutGm, config, startLoopIdx, tileNum);
            } else {
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    AivGmm2PostA8W4<ElementC, MakeLayoutC>(
                        workSet.scheduler, workSet.l0cOutGm, params, args.groupCnt, config,
                        startLoopIdx, tileNum);
                }
            }
        }
    }
}

template <uint8_t CombineQuantMode, typename Policy, typename ElementA, typename ElementB, typename ElementC,
    typename ElementMxScaleA, typename ElementMxScaleB, typename ExtraArgs>
__aicore__ inline void GroupMatmulImplA8W4(const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape, const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx, ExtraArgs& args)
{
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using Config = Config<true, Policy, 0, ElementA, ElementB, ElementC, ElementMxScaleA,
        ElementMxScaleB, false>;
    auto config = Config::BuildProblemConfig(problemShape);

    if constexpr (Policy::IS_GMM1) {
        args.swigluQuantOp.UpdateNextProblem({config.m, config.outputN, config.k, 0});
    }

    auto layouts = Config::BuildLayouts(config);
    using BlockMmad = typename Config::BlockMmad;
    using BlockPrologue = typename Config::BlockPrologue;
    using MakeLayoutC = typename Config::MakeLayoutC;

    auto l0cOutGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC*>(
        Policy::IS_GMM1 ? gmmAddrInfo.gmm1OutGlobal : gmmAddrInfo.gmm2OutGlobal)), layouts.c);
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
        reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(
            reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(
            reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);

    BlockScheduler scheduler(
        {config.m, config.outputN, config.k},
        BlockScheduler::Params{Te::MakeCoord(static_cast<int64_t>(L1_TILE_M_256), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    using WorkSetType = WorkSetA8W4<
        BlockScheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA), decltype(gmScaleB), decltype(l0cOutGm)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, l0cOutGm};
    GroupMatmulExecA8W4<CombineQuantMode, Policy, BlockMmad, BlockPrologue, ElementC, MakeLayoutC>(
        workSet, params, gmmAddrInfo, config, startLoopIdx, tileNum, args);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}
} // namespace Detail

// GroupMatmulSwigluQuantA8W4 — A8W4 prologue（W4→W8）+ GMM1 + SwiGLU + 量化
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuantA8W4(
    BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& swigluQuantOp,
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom)
{
    (void)vecSetSyncCom;
    using SwigluQuantOpType = std::remove_reference_t<decltype(swigluQuantOp)>;
    Detail::Gmm1ArgsA8W4<SwigluQuantOpType> args{swigluQuantOp};
    Detail::GroupMatmulImplA8W4<COMBINE_NO_QUANT, Detail::Gmm1Policy, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>(params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

// GroupMatmul2CombineA8W4 — A8W4 prologue（W4→W8）+ GMM2 + Combine
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC,
    typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2CombineA8W4(
    const Params& params, const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo, uint32_t& startBlockIdx, int32_t& vecSetSyncCom2, uint32_t groupCnt,
    uint16_t& pingpongIdx, uint32_t groupIdx = 0)
{
    (void)vecSetSyncCom2;
    Detail::Gmm2ArgsA8W4 args{groupCnt, pingpongIdx, groupIdx};
    Detail::GroupMatmulImplA8W4<CombineQuantMode, Detail::Gmm2Policy, ElementA, ElementB, ElementC,
        ElementMxScaleA, ElementMxScaleB>(params, problemShape, gmmAddrInfo, startBlockIdx, args);
}

} // namespace MegaMoeImpl

#endif