/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_SMALL_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_SMALL_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class DynamicSmallMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;

    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord l1TileShape;
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
        Params(GemmCoord const &problemShape_, GemmCoord const &l1TileShape_, GM_ADDR ptrA_, LayoutA &layoutA_,
            GM_ADDR ptrB_, LayoutB &layoutB_, GM_ADDR ptrC_, LayoutC &layoutC_)
            : problemShape(problemShape_), l1TileShape(l1TileShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_),
              layoutB(layoutB_), ptrC(ptrC_), layoutC(layoutC_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    DynamicSmallMatmul()
    {}

    /// Executes matmul
    CATLASS_DEVICE void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        uint32_t mLoops = (params.problemShape.m() + params.l1TileShape.m() - 1) / params.l1TileShape.m();
        uint32_t nLoops = (params.problemShape.n() + params.l1TileShape.n() - 1) / params.l1TileShape.n();
        BlockMmad blockMmad(params.l1TileShape, resource);

        uint32_t totalLoops = mLoops * nLoops;

        // For small-shape matmul, each AI Core computes only one block.
        uint32_t loopIdx = AscendC::GetBlockIdx();
        // Compute block location
        GemmCoord blockCoord{loopIdx / nLoops, loopIdx % nLoops, 0};
        uint32_t mActual = (blockCoord.m() == (mLoops - 1))
                               ? (params.problemShape.m() - blockCoord.m() * params.l1TileShape.m())
                               : params.l1TileShape.m();
        uint32_t nActual = (blockCoord.n() == (nLoops - 1))
                               ? (params.problemShape.n() - blockCoord.n() * params.l1TileShape.n())
                               : params.l1TileShape.n();
        GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};

        // Compute initial location in logical coordinates
        MatrixCoord offsetA{blockCoord.m() * params.l1TileShape.m(), 0U};
        MatrixCoord offsetB{0U, blockCoord.n() * params.l1TileShape.n()};
        MatrixCoord offsetC{blockCoord.m() * params.l1TileShape.m(), blockCoord.n() * params.l1TileShape.n()};
        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
        int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
        int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

        // Compute block-scoped matrix multiply-add
        blockMmad(gmA[gmOffsetA],
            params.layoutA,
            gmB[gmOffsetB],
            params.layoutB,
            gmC[gmOffsetC],
            params.layoutC,
            actualBlockShape);

        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_DYNAMIC_SMALL_MATMUL_HPP