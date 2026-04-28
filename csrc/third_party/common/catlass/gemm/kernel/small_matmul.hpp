/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_SMALL_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_SMALL_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

// Template for Matmul kernel. Compute C = A * B
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class SmallMatmul {
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

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_) {}
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
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        Params params{args.problemShape, args.ptrA, layoutA, args.ptrB, layoutB, args.ptrC, layoutC};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    SmallMatmul() {}

    /// Executes one Matmul
    CATLASS_DEVICE
    void operator()(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        uint32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;
        uint32_t nLoops = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        uint32_t totalLoops = mLoops * nLoops;

        // For small-shape matmul, each AI Core computes only one block.
        uint32_t loopIdx = AscendC::GetBlockIdx();
        if (loopIdx < totalLoops) {
            // Compute block location
            GemmCoord blockCoord{loopIdx / nLoops, loopIdx % nLoops, 0};
            uint32_t mActual = (blockCoord.m() == (mLoops - 1)) ?
                (params.problemShape.m() - blockCoord.m()  * L1TileShape::M) : L1TileShape::M;
            uint32_t nActual = (blockCoord.n() == (nLoops - 1)) ?
                (params.problemShape.n() - blockCoord.n()  * L1TileShape::N) : L1TileShape::N;
            GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, 0U};
            MatrixCoord offsetB{0U, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutA,
                      gmB[gmOffsetB], params.layoutB,
                      gmC[gmOffsetC], params.layoutC,
                      actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_SMALL_MATMUL_HPP