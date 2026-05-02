/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_SINGLE_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_SINGLE_SPLITK_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"

#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <class PrologueA_, class PrologueB_, class BlockMmad_, 
    class BlockEpilogue_, class BlockScheduler_, class RemovePaddingNDAndCastC_>
class SingleCoreSplitkMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using RemovePaddingNDAndCastC = RemovePaddingNDAndCastC_;

    using BlockMmad = BlockMmad_;
    using BlockScheduler = BlockScheduler_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;

    template<class T, bool isOutType = false, typename = void>
    struct LayoutHelper;

    template<class T, bool isOutType>
    struct LayoutHelper<T, isOutType, std::enable_if_t<!std::is_void_v<T>>> {
        using type = std::conditional_t<isOutType, typename T::LayoutOut, typename T::LayoutIn>;
    };

    template<class T, bool isOutType>
    struct LayoutHelper<T, isOutType, std::enable_if_t<std::is_void_v<T>>> {
        using type = void;
    };

    /// Process Layout
    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;
    using LayoutAOut = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA, true>::type>;
    using LayoutBOut  = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB, true>::type>;

    template<class T>
    struct ElementHelper {
        using ElementC = typename T::ElementOut;
    };
    template<>
    struct ElementHelper<void> {
        using ElementC = typename BlockMmad::ElementC;
    };
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using ElementC = typename ElementHelper<RemovePaddingNDAndCastC>::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    // Decice whther PaddingA/PaddingB enabled.
    using PaddingA = PrologueA;
    using PaddingB = PrologueB;
    static constexpr bool isPaddingA = !std::is_void_v<PrologueA>;
    static constexpr bool isPaddingB = !std::is_void_v<PrologueB>;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutAOut layoutWA;
        GM_ADDR ptrWB;
        LayoutBOut layoutWB;
        GM_ADDR ptrWC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA& layoutA_, 
            GM_ADDR ptrB_, LayoutB& layoutB_, GM_ADDR ptrC_, LayoutC& layoutC_,
            GM_ADDR ptrWA_, LayoutAOut layoutWA_, GM_ADDR ptrWB_, LayoutBOut layoutWB_,
            GM_ADDR prtWC_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_), 
            ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_),
            ptrWC(prtWC_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t workspace = 0, sizeWC = 0;
        if constexpr (isPaddingA) {
            workspace += GetPaddingSizeA(args);
        }
        if constexpr (isPaddingB) {
            workspace += GetPaddingSizeB(args);
        }

        if constexpr (RemovePaddingNDAndCastC::paddingTag == PaddingTag::NO_PADDING && 
            !std::is_same_v<ElementAccumulator, ElementC>) {
            sizeWC = RemovePaddingNDAndCastC::GetWorkspaceSize(args.problemShape.m(), args.problemShape.n());
        }else if constexpr (RemovePaddingNDAndCastC::paddingTag == PaddingTag::PADDING_ND) {
            sizeWC = RemovePaddingNDAndCastC::GetWorkspaceSize(args.problemShape.m(), args.problemShape.n(),
                 512 / sizeof(ElementAccumulator));
        }
        workspace += sizeWC;

        return workspace;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(args.problemShape.m(), args.problemShape.n());
        
        // Malloc memory @[npu device]
        // Meanwhile, arrange `layoutWA`, `layoutWB` (if padding enabled)
        uint8_t *gmWA = nullptr;
        uint8_t *gmWB = nullptr;
        uint8_t *gmWC = nullptr;
        LayoutAOut layoutWA;
        LayoutBOut layoutWB;
        size_t sizeWA = 0, sizeWB = 0;
        if constexpr (isPaddingA) {
            gmWA = workspace;
            SetPaddingALayout(layoutWA, layoutA);
            sizeWA += GetPaddingSizeA(args);
        }

        if constexpr (isPaddingB) {
            gmWB = workspace + sizeWA;
            SetPaddingBLayout(layoutWB, layoutB);
            sizeWB += GetPaddingSizeB(args);
        }

        gmWC = workspace + sizeWA + sizeWB;
        // Concat params
        Params params{
            args.problemShape, 
            args.ptrA, 
            layoutA, 
            args.ptrB, 
            layoutB, 
            args.ptrC, 
            layoutC,
            gmWA, 
            layoutWA,
            gmWB,
            layoutWB,
            gmWC};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    SingleCoreSplitkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (isPaddingA) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if constexpr (isPaddingB) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));

            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            AscendC::GlobalTensor<ElementC> gmC;
            AscendC::GlobalTensor<ElementAccumulator> gmWC;
            gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
            gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWC));
            if constexpr (RemovePaddingNDAndCastC::paddingTag == PaddingTag::NO_PADDING) {
                RemovePaddingNDAndCastC removePaddingNDAndCastC(resource);
                removePaddingNDAndCastC(gmC, gmWC, params.layoutC, params.layoutC);
            } else {
                LayoutC layoutWC = RemovePaddingNDAndCastC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementAccumulator));
                RemovePaddingNDAndCastC removePaddingNDAndCastC(resource);
                removePaddingNDAndCastC(gmC, gmWC, params.layoutC, layoutWC);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template<>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }


        BlockScheduler matmulBlockScheduler(params.problemShape, 
            GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K));
        uint32_t coreLoops = matmulBlockScheduler.GetSingleCoreLoops();

        LayoutAOut layoutA;
        LayoutBOut layoutB;
        typename BlockMmad::LayoutC layoutC;

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        if constexpr (std::is_void_v<PrologueA>) {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
            layoutA = params.layoutA;
        } else {
            gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
            layoutA = params.layoutWA;
        }
        AscendC::GlobalTensor<ElementB> gmB;

        using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            layoutB = params.layoutWB;
        }
        AscendC::GlobalTensor<ElementAccumulator> gmC;
        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            gmC.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrWC);
            if constexpr (RemovePaddingNDAndCastC::paddingTag == Catlass::Gemm::Kernel::PaddingTag::NO_PADDING) {
                layoutC = params.layoutC;
            } else {
                layoutC = RemovePaddingNDAndCastC::GetWorkspaceLayout(params.layoutC, 512 / sizeof(ElementAccumulator));
            }
        } else {
            gmC.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrC);
            layoutC = params.layoutC;
        }

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        GemmCoord blockCoord;
        GemmCoord actualBlockShape;
        GemmCoord nextBlockCoord;
        GemmCoord nextActualBlockShape;

        for (uint32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {

            bool isFirstBlock = (loopIdx == 0);
            if (isFirstBlock) {
                blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            } else {
                blockCoord = nextBlockCoord;
                actualBlockShape = nextActualBlockShape;
            }

            // when loopIdx < coreLoops - 1, at least one of needLoadNextA/needLoadNextB is true
            // when loopIdx == coreLoops - 1, both are false
            bool needLoadNextA = false, needLoadNextB = false;
            if (loopIdx < coreLoops - 1) {
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + 1);
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
                needLoadNextA = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.m() != nextBlockCoord.m());
                needLoadNextB = (blockCoord.k() != nextBlockCoord.k()) || (blockCoord.n() != nextBlockCoord.n());
            }

            // Compute initial location in logical coordinates
            MatrixCoord coordA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord coordB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord coordC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = layoutC.GetOffset(coordC);

            MatrixCoord coordNextA{
                nextBlockCoord.m() * L1TileShape::M, nextBlockCoord.k() * L1TileShape::K};
            MatrixCoord coordNextB{
                nextBlockCoord.k() * L1TileShape::K, nextBlockCoord.n() * L1TileShape::N};
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA],
                layoutA,
                gmB[gmOffsetB],
                layoutB,
                gmC[gmOffsetC],
                layoutC,
                gmA[gmOffsetNextA],
                gmB[gmOffsetNextB],
                actualBlockShape,
                nextActualBlockShape,
                needLoadNextA,
                needLoadNextB,
                matmulBlockScheduler.IsAtomicAdd(loopIdx));
        }
        if constexpr (!std::is_void_v<RemovePaddingNDAndCastC>) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        }
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

protected:
    static size_t GetPaddingSizeA(const Arguments &args) 
    {
        if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
            return PaddingA::GetWorkspaceSize(args.problemShape.m(), args.problemShape.k(), L1TileShape::M, L1TileShape::K);
        } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_ND) {
            return PaddingA::GetWorkspaceSize(args.problemShape.m(), args.problemShape.k(), 512 / sizeof(ElementA));
        } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_NZ) {
            return PaddingA::GetWorkspaceSize(args.problemShape.m(), args.problemShape.k());
        } else {
            return 0;
        }
    }

    static size_t GetPaddingSizeB(const Arguments &args) 
    {
        if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
            return PaddingB::GetWorkspaceSize(args.problemShape.k(), args.problemShape.n(), L1TileShape::K, L1TileShape::N);
        } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_ND) {
            // Optimal bandwidth for 512 Byte aligned reads
            return PaddingB::GetWorkspaceSize(args.problemShape.k(), args.problemShape.n(), 512 / sizeof(ElementB));
        } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_NZ) {
            return PaddingB::GetWorkspaceSize(args.problemShape.k(), args.problemShape.n());
        } else {
            return 0;  // 默认返回0
        }
    }

    static void SetPaddingALayout(LayoutAOut &layoutWA_, LayoutA &layoutA_) 
    {
        if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
            layoutWA_ = PrologueA::GetWorkspaceLayout(layoutA_, 512 / sizeof(ElementA));
        } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
            layoutWA_ = PrologueA::GetWorkspaceLayout(layoutA_, L1TileShape::M, L1TileShape::K);
        } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
            layoutWA_ = PrologueA::GetWorkspaceLayout(layoutA_);
        }
    }

    static void SetPaddingBLayout(LayoutBOut &layoutWB_, const LayoutB &layoutB_)
    {
        if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
            layoutWB_ = PrologueB::GetWorkspaceLayout(layoutB_, 512 / sizeof(ElementB));
        } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
            layoutWB_ = PrologueB::GetWorkspaceLayout(layoutB_, L1TileShape::K, L1TileShape::N);
        } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
            layoutWB_ = PrologueB::GetWorkspaceLayout(layoutB_);
            // Do nothing
        }
    }

private:
    Arch::Resource<ArchTag> resource;

    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_SINGLE_SPLITK_MATMUL_HPP