/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_AIV_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_AIV_HPP

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"


namespace Catlass::Gemm::Kernel {

template <class PrologueA_, class PrologueB_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class AivMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using TensorCoord = layout::VectorLayout::TensorCoord;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        MatrixCoord taskTileShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            MatrixCoord const &taskTileShape_,
            GM_ADDR ptrA_,
            LayoutA layoutA_,
            GM_ADDR ptrB_,
            LayoutB layoutB_,
            GM_ADDR ptrC_,
            LayoutC layoutC_
        )
            : problemShape(problemShape_)
            , taskTileShape(taskTileShape_)
            , ptrA(ptrA_)
            , layoutA(layoutA_)
            , ptrB(ptrB_)
            , layoutB(layoutB_)
            , ptrC(ptrC_)
            , layoutC(layoutC_)
        {
        }
    };

    struct Arguments {
        GemmCoord problemShape;
        MatrixCoord taskTileShape;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return args.problemShape.k() == 1;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m()};
        LayoutB layoutB{args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        Params params{args.problemShape, args.taskTileShape, args.ptrA, layoutA, args.ptrB,
                      layoutB,           args.ptrC,          layoutC};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    AivMatmul()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params, Arch::Resource<ArchTag> &resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Arch::Resource<ArchTag> &resource)
    {
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params, Arch::Resource<ArchTag> &resource)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, params.taskTileShape);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource, params.taskTileShape);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            TensorCoord coordA{blockCoord.m() * params.taskTileShape.row()};
            TensorCoord coordB{blockCoord.n() * params.taskTileShape.column()};
            MatrixCoord coordC{
                blockCoord.m() * params.taskTileShape.row(), blockCoord.n() * params.taskTileShape.column()
            };
            int64_t gmOffsetA = params.layoutA.GetOffset(coordA);
            int64_t gmOffsetB = params.layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC);
            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC,
                actualBlockShape
            );
        }
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_AIV_HPP