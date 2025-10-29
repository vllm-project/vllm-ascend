/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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

    using LayoutA = std::conditional_t<std::is_same_v<LayoutWA, layout::RowMajor> ||
        std::is_same_v<LayoutWA, layout::PaddingRowMajor>, layout::RowMajor, layout::ColumnMajor>;
    using LayoutB = std::conditional_t<std::is_same_v<LayoutWB, layout::RowMajor> ||
        std::is_same_v<LayoutWB, layout::PaddingRowMajor>, layout::RowMajor, layout::ColumnMajor>;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrixBlockND<ArchTag, ElementA, LayoutA, LayoutWA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrixBlockND<ArchTag, ElementB, LayoutB, LayoutWB, COMPUTE_LENGTH_B>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

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
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t align;
        size_t elementSize;
        LayoutWA layoutWA;
        LayoutWB layoutWB;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static bool IfNeedPadding(layout::RowMajor layout, uint32_t align)
    {
        // prevent division by zero
        if (align == 0) {
            return false;
        }
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(0) < 65536) {
            return layout.stride(0) % align != 0;
        } else {
            return true;
        }
    }

    static bool IfNeedPadding(layout::ColumnMajor layout, uint32_t align)
    {
        // prevent division by zero
        if (align == 0) {
            return false;
        }
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(1) < 65536) {
            return layout.stride(1) % align != 0;
        } else {
            return true;
        }
    }

    template<class Layout>
    static size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
    {
        return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
                RoundUp(static_cast<size_t>(layout.shape(1)), blockRows);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t workspaceSize = 0;
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        if (IfNeedPadding(layoutA, args.align)) {
            workspaceSize +=
                GetWorkspaceLen(layoutA, args.layoutWA.shape(0), args.layoutWA.shape(2)) * args.elementSize;
        }
        if (IfNeedPadding(layoutB, args.align)) {
            workspaceSize +=
                GetWorkspaceLen(layoutB, args.layoutWB.shape(0), args.layoutWB.shape(2)) * args.elementSize;
        }
        return workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using LayoutPaddingB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        bool isPaddingA = IfNeedPadding(layoutA, args.align);
        bool isPaddingB = IfNeedPadding(layoutB, args.align);

        uint8_t *gmWA = nullptr;
        uint8_t *gmWB = nullptr;
        size_t sizeWA = 0;

        if (isPaddingA) {
            gmWA = workspace;
            sizeWA = GetWorkspaceLen(layoutA, args.layoutWA.shape(0), args.layoutWA.shape(2)) * args.elementSize;
        } else {
            gmWA = args.ptrA;
        }
        if (isPaddingB) {
            gmWB = workspace + sizeWA;
        } else {
            gmWB = args.ptrB;
        }
        Params params{args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            gmWA,
            args.layoutWA,
            gmWB,
            args.layoutWB};
        return params;
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
        if (params.ptrA != params.ptrWA) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PaddingA paddingA(resource);
            paddingA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if (params.ptrB != params.ptrWB) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PaddingB paddingB(resource);
            paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if ((params.ptrA != params.ptrWA) || (params.ptrB != params.ptrWB)) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if ((params.ptrA != params.ptrWA) || (params.ptrB != params.ptrWB)) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockIdxCoord.k() * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
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
            int64_t gmOffsetNextA = params.layoutWA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutWB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutWA,
                gmB[gmOffsetB], params.layoutWB,
                gmC[gmOffsetC], params.layoutC,
                gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP