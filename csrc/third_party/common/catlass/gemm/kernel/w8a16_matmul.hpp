/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_W8A16_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_W8A16_MATMUL_HPP

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
class W8A16Matmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementPrologueB = typename BlockMmad::PrologueB::ElementSrc;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutPrologueB = typename BlockMmad::PrologueB::LayoutSrc;
    using LayoutB = typename BlockMmad::LayoutB;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using MmadParams = typename BlockMmad::Params;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrPrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        MmadParams mmadParams;

        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrPrologueB_, LayoutPrologueB const &layoutPrologueB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            MmadParams const &mmadParams_,
            GM_ADDR ptrWorkspace_
        ):  problemShape(problemShape_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrPrologueB(ptrPrologueB_), layoutPrologueB(layoutPrologueB_),
            ptrC(ptrC_), layoutC(layoutC_),
            mmadParams(mmadParams_),
            ptrWorkspace(ptrWorkspace_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR deviceA;
        LayoutA layoutA;
        GM_ADDR devicePrologueB;
        LayoutPrologueB layoutPrologueB;
        GM_ADDR deviceC;
        LayoutC layoutC;
        half deqScalar;
        half deqZeroPoint;
        uint32_t aicoreNum;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(Arguments const &args)
    {
        return BlockMmad::STAGES * L1TileShape::K * L1TileShape::N * sizeof(ElementB) * args.aicoreNum;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.problemShape,
            args.deviceA, args.layoutA,
            args.devicePrologueB, args.layoutPrologueB,
            args.deviceC, args.layoutC,
            {{}, {args.deqScalar, args.deqZeroPoint}, {}},
            workspace
        };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    W8A16Matmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        auto aicoreNum = AscendC::GetBlockNum();
        auto aicoreIdx = AscendC::GetBlockIdx();

        BlockMmad blockMmad(resource, params.mmadParams);

        GemmCoord blockShape = L1TileShape::ToCoord();
        BlockScheduler matmulBlockScheduler(params.problemShape, blockShape.GetCoordMN());
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Load A from workspace
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));

        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = aicoreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);
            GemmCoord offsetCoord = blockIdxCoord * blockShape;

            auto gmBlockA = gmA[params.layoutA.GetOffset(offsetCoord.GetCoordMK())];
            auto layoutBlockA = params.layoutA.GetTileLayout(actualBlockShape.GetCoordMK());
            auto gmBlockC = gmC[params.layoutC.GetOffset(offsetCoord.GetCoordMN())];
            auto layoutBlockC = params.layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

            blockMmad(
                gmBlockA, layoutBlockA,
                gmBlockB, layoutBlockB,
                gmBlockC, layoutBlockC,
                actualBlockShape
            );
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        auto aicoreNum = AscendC::GetBlockNum();
        auto aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        BlockMmad blockMmad(resource, params.mmadParams);
        BlockScheduler matmulBlockScheduler(params.problemShape, L1TileShape::ToCoordMN());

        AscendC::GlobalTensor<ElementPrologueB> gmPrologueB;
        gmPrologueB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPrologueB *>(params.ptrPrologueB));

        LayoutB layoutBlockB{L1TileShape::K, L1TileShape::N};
        auto gmOffsetB = aicoreIdx * layoutBlockB.Capacity() * BlockMmad::STAGES;
        AscendC::GlobalTensor<ElementB> gmBlockB;
        gmBlockB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWorkspace) + gmOffsetB);

        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            auto blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            auto offsetCoordB = blockIdxCoord.GetCoordKN() * L1TileShape::ToCoordKN();
            auto gmBlockPrologueB = gmPrologueB[params.layoutPrologueB.GetOffset(offsetCoordB)];
            auto layoutBlockPrologueB = params.layoutPrologueB.GetTileLayout(actualBlockShape.GetCoordKN());

            // Compute block-scoped matrix multiply-add
            blockMmad.Prologue(
                gmBlockPrologueB, layoutBlockPrologueB,
                gmBlockB, layoutBlockB,
                actualBlockShape
            );
        }
    }

private:
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_W8A16_MATMUL_HPP