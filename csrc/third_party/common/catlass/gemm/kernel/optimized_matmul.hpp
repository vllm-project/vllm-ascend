/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <
    class PrologueA,
    class PrologueB,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class OptimizedMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<
        std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<
        std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct ParamsBase {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        // Methods
        CATLASS_HOST_DEVICE
        ParamsBase() {}

        CATLASS_HOST_DEVICE
        ParamsBase(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_) {}
    };

    template<bool IsPaddingA = true, bool IsPaddingB = true>
    struct KernelParams : public ParamsBase {
        // Data members
        using LayoutWA = typename BlockMmad::LayoutA;
        using LayoutWB = typename BlockMmad::LayoutB;

        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;

        // Methods
        CATLASS_HOST_DEVICE
        KernelParams() {}

        CATLASS_HOST_DEVICE
        KernelParams(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_)
            : ParamsBase(problemShape_, ptrA_, layoutA_, ptrB_, layoutB_, ptrC_, layoutC_),
            ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    template<>
    struct KernelParams<true, false> : public ParamsBase {
        // Data members
        using LayoutWA = typename BlockMmad::LayoutA;

        GM_ADDR ptrWA;
        LayoutWA layoutWA;

        // Methods
        CATLASS_HOST_DEVICE
        KernelParams() {}

        CATLASS_HOST_DEVICE
        KernelParams(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutWA layoutWA_)
            : ParamsBase(problemShape_, ptrA_, layoutA_, ptrB_, layoutB_, ptrC_, layoutC_),
            ptrWA(ptrWA_), layoutWA(layoutWA_) {}
    };

    template<>
    struct KernelParams<false, true> : public ParamsBase {
        // Data members
        using LayoutWB = typename BlockMmad::LayoutB;

        GM_ADDR ptrWB;
        LayoutWB layoutWB;;

        // Methods
        CATLASS_HOST_DEVICE
        KernelParams() {}

        CATLASS_HOST_DEVICE
        KernelParams(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWB_, LayoutWB layoutWB_)
            : ParamsBase(problemShape_, ptrA_, layoutA_, ptrB_, layoutB_, ptrC_, layoutC_),
            ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    template<>
    struct KernelParams<false, false> : public ParamsBase {
        // Methods
        CATLASS_HOST_DEVICE
        KernelParams() {}

        CATLASS_HOST_DEVICE
        KernelParams(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
            : ParamsBase(problemShape_, ptrA_, layoutA_, ptrB_, layoutB_, ptrC_, layoutC_) {}
    };
    
    using Params = KernelParams<!std::is_void_v<PrologueA>, !std::is_void_v<PrologueB>>;

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
        constexpr bool isPaddingA = !std::is_void_v<PrologueA>;
        constexpr bool isPaddingB = !std::is_void_v<PrologueB>;
        size_t workspaceSize = 0;
        if constexpr (isPaddingA) {
            if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                workspaceSize += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k(), L1TileShape::M, L1TileShape::K);
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                workspaceSize += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k(), 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_NZ) {
                workspaceSize += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k());
            }
        }
        if constexpr (isPaddingB) {
            if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                workspaceSize += PrologueB::GetWorkspaceSize(
                        args.problemShape.k(), args.problemShape.n(), L1TileShape::K, L1TileShape::N);
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                workspaceSize += PrologueB::GetWorkspaceSize(
                        args.problemShape.k(), args.problemShape.n(), 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_NZ) {
                workspaceSize += PrologueB::GetWorkspaceSize(
                        args.problemShape.k(), args.problemShape.n());
            }
        }
        return workspaceSize;
    }

    static auto ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        constexpr bool isPaddingA = !std::is_void_v<PrologueA>;
        constexpr bool isPaddingB = !std::is_void_v<PrologueB>;
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(args.problemShape.m(), args.problemShape.n());

        uint8_t *gmWA = nullptr;
        uint8_t *gmWB = nullptr;
        size_t sizeWA = 0;
        if constexpr (isPaddingA) {
            gmWA = workspace;
            if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                sizeWA += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k(), L1TileShape::M, L1TileShape::K);
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                sizeWA += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k(), 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_NZ) {
                sizeWA += PrologueA::GetWorkspaceSize(
                        args.problemShape.m(), args.problemShape.k());
            }
        }
        if constexpr (isPaddingB) {
            gmWB = workspace + sizeWA;
        }

        if constexpr (isPaddingA && isPaddingB) {
            typename PrologueA::LayoutOut layoutWA;
            typename PrologueB::LayoutOut layoutWB;
            if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA, L1TileShape::M, L1TileShape::K);
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA);
            }
            if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB, L1TileShape::K, L1TileShape::N);
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB);
            }
            Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC, 
                gmWA, layoutWA, gmWB, layoutWB};
            return params;
        } else if constexpr (isPaddingA) {
            typename PrologueA::LayoutOut layoutWA;
            if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA, L1TileShape::M, L1TileShape::K);
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(layoutA);
            }
            Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC,
                gmWA, layoutWA};
            return params;
        } else if constexpr (isPaddingB) {
            typename PrologueB::LayoutOut layoutWB;
            if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB, L1TileShape::K, L1TileShape::N);
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_ND) {
                // Optimal bandwidth for 512 Byte aligned reads
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(layoutB);
            }
            Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC,
                gmWB, layoutWB};
            return params;
        } else {
            Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC};
            return params;
        }  
    }

    // Methods
    CATLASS_DEVICE
    OptimizedMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
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

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        typename BlockMmad::LayoutA layoutA;
        typename BlockMmad::LayoutB layoutB;

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
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            layoutB = params.layoutWB;
        }
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockIdCoord);
            }
            MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M, nextBlockIdCoord.k() * L1TileShape::K};
            MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K, nextBlockIdCoord.n() * L1TileShape::N};
            int64_t gmOffsetNextA = layoutA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], layoutA,
                gmB[gmOffsetB], layoutB,
                gmC[gmOffsetC], params.layoutC,
                gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP