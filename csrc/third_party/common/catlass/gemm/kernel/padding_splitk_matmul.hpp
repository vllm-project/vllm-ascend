/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ReduceAdd_
>
class PaddingSplitkMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrixND<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrixND<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    using BlockScheduler = BlockScheduler_;
    using ReduceAdd = ReduceAdd_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        bool aNeedPadding;
        bool bNeedPadding;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutA layoutWA;
        GM_ADDR ptrWB;
        LayoutB layoutWB;
        GM_ADDR ptrWC;
        uint32_t splitkFactor = 1;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, bool aNeedPadding_, bool bNeedPadding_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutA layoutWA_,
               GM_ADDR ptrWB_, LayoutB layoutWB_, GM_ADDR ptrWC_, uint32_t splitkFactor_)
            : problemShape(problemShape_), aNeedPadding(aNeedPadding_), bNeedPadding(bNeedPadding_),
              ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_),
              ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_),
              ptrWC(ptrWC_), splitkFactor(splitkFactor_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        uint32_t align;
        bool aNeedPadding;
        bool bNeedPadding;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t aicCoreNum)
    {
        uint32_t maxSplitkFactor;
        if (k <= 1024) {
            // When k is less than or equal to 1024, it can be divided into at most 2 parts.
            maxSplitkFactor = 2;
        } else if (k <= 2048) {
            // When k is less than or equal to 2048, it can be divided into at most 4 parts.
            maxSplitkFactor = 4;
        } else if (k <= 4096) {
            // When k is less than or equal to 4096, it can be divided into at most 8 parts.
            maxSplitkFactor = 8;
        } else {
            // else it can be divided into at most 16 parts.
            maxSplitkFactor = 16;
        }
        uint32_t splitkFactor = 1;
        uint32_t m0 = L1TileShape::M;
        uint32_t n0 = L1TileShape::N;
        uint32_t k0 = L1TileShape::K;

        uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
        splitkFactor = (aicCoreNum / baseTilesCount < maxSplitkFactor) ? (aicCoreNum / baseTilesCount) : maxSplitkFactor;
        // Prevent the split factor form being less than 1
        splitkFactor = (splitkFactor > static_cast<uint32_t>(1)) ? splitkFactor : static_cast<uint32_t>(1);
        if (baseTilesCount < aicCoreNum) {
            while (splitkFactor + 1 <= maxSplitkFactor &&
                CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >=
                CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
                splitkFactor += 1;
            }
        }
        // Ensure that splitkFactor is less than the number of base tiels in the k direction.
        splitkFactor = (CeilDiv(k, k0) < splitkFactor) ? CeilDiv(k, k0) : splitkFactor;
        // If k is very large, splitting k can lead to better cache utilization.
        // If k is greater than 8192.
        if (k > 8192) {
            // split the k direction into at least 2 parts.
            splitkFactor = (splitkFactor > static_cast<uint32_t>(2)) ? splitkFactor : static_cast<uint32_t>(2);
        }
        // If k is greater than 32768.
        if (k > 32768) {
            // split the k direction into at least 4 parts.
            splitkFactor = (splitkFactor > static_cast<uint32_t>(4)) ? splitkFactor : static_cast<uint32_t>(4);
        }
        return splitkFactor;
    }

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
    {
        // prevent division of 0
        if (align == 0) {
            return layout;
        }
        return layout::RowMajor(layout.shape(0), layout.shape(1),
            (layout.shape(1) + align - 1) / align * align);
    }

    static layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
    {
        // prevent division of 0
        if (align == 0) {
            return layout;
        }
        return layout::ColumnMajor(layout.shape(0), layout.shape(1),
            (layout.shape(0) + align - 1) / align * align);
    }

    static size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }

    static size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        GemmCoord problemShape = args.problemShape;
        LayoutA layoutA{problemShape.m(), problemShape.k()};
        LayoutB layoutB{problemShape.k(), problemShape.n()};
        size_t sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        size_t sizeWB = GetWorkspaceLen(GetWorkspaceLayout(layoutB, args.align)) * args.elementSize;
        size_t sizeWC = args.elementSize * args.problemShape.m() * args.problemShape.n() *
                        GetSplitkFactor(args.problemShape.m(),
                                        args.problemShape.n(),
                                        args.problemShape.k(),
                                        args.aicCoreNum);
        return sizeWA + sizeWB + sizeWC;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};

        uint8_t *workspaceWA = nullptr;
        uint8_t *workspaceWB = nullptr;
        size_t sizeWA = 0;
        size_t sizeWB = 0;

        if (args.aNeedPadding) {
            workspaceWA = workspace;
            sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        } else {
            workspaceWA = args.ptrA;
        }

        if (args.bNeedPadding) {
            workspaceWB = workspace + sizeWA;
            sizeWB = GetWorkspaceLen(GetWorkspaceLayout(layoutB, args.align)) * args.elementSize;
        } else {
            workspaceWB = args.ptrB;
        }

        uint8_t *workspaceWC = workspace + sizeWA + sizeWB;

        Params params{
            args.problemShape,
            args.aNeedPadding,
            args.bNeedPadding,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            workspaceWA,
            GetWorkspaceLayout(layoutA, args.align),
            workspaceWB,
            GetWorkspaceLayout(layoutB, args.align),
            workspaceWC,
            GetSplitkFactor(args.problemShape.m(),
                            args.problemShape.n(),
                            args.problemShape.k(),
                            args.aicCoreNum)};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    PaddingSplitkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes one Matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if (params.aNeedPadding || params.bNeedPadding) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape,
            GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), params.splitkFactor);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWC);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(
                blockCoord, matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            uint64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            uint64_t gmOffsetC = params.layoutC.GetOffset(offsetC)
                + static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n())
                * static_cast<uint64_t>(matmulBlockScheduler.GetSplitkSliceIdx(loopIdx));

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutWA,
                      gmB[gmOffsetB], params.layoutWB,
                      gmC[gmOffsetC], params.layoutC,
                      actualBlockShape);
        }

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if (params.aNeedPadding) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (params.bNeedPadding) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
        }

        if (params.aNeedPadding || params.bNeedPadding) {
            // 0x0 synchronization control between AI Core
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        // reduce add
        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmWC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut*>(params.ptrC));
        gmWC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator*>(params.ptrWC));
        ReduceAdd reduceAdd(resource);
        reduceAdd(gmC, gmWC,
            static_cast<uint64_t>(params.problemShape.m()) * static_cast<uint64_t>(params.problemShape.n()),
            params.splitkFactor);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 0;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 1;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_PADDING_SPLITK_MATMUL_HPP
