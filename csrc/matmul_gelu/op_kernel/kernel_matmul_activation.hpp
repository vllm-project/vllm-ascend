/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

// Template for matmul add kernel. Compute C(fp32) = A * B, D = Cast(Activation(C))
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class MatmulActivation {
public:
    // BlockMmad的内核
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;

    // 后处理的内核
    using BlockEpilogue = BlockEpilogue_;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    static_assert(std::is_same_v<typename BlockEpilogue::ElementC, ElementC> &&
        std::is_same_v<typename BlockEpilogue::LayoutC, LayoutC>,
        "The CType of Mmad and Epilogue should be consistent.");

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrBias_,
            GM_ADDR ptrWorkspace_, EpilogueParams const &epilogueParams_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_), ptrBias(ptrBias_),
            ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrBias;
        GM_ADDR ptrD;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return args.elementSize * args.problemShape.m() * args.problemShape.n();
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        GemmCoord problemShape = args.problemShape;
        uint32_t m = problemShape.m();
        uint32_t n = problemShape.n();
        uint32_t k = problemShape.k();
        LayoutA layoutA{m, k};
        LayoutB layoutB{k, n};
        LayoutC layoutC{m, n};
        // 传出
        typename BlockEpilogue::Params epilogueParams{workspace, layoutC, args.ptrD, layoutC};
        Params params{problemShape,
            args.ptrA, layoutA, // A矩阵
            args.ptrB, layoutB, // B矩阵
            args.ptrBias, // bias
            workspace,
            epilogueParams};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    MatmulActivation() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            int64_t gmOffsetBias = blockCoord.n() * L1TileShape::N;

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], layoutC,
                gmBias[gmOffsetBias],
                actualBlockShape);

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockEpilogue blockEpilogue(resource, params.epilogueParams);

        // Represent the full gm
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        // Get aicore information
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t subcoreIndex = AscendC::GetSubBlockIdx();

        // Loop through the epilogue calculations of each basic block
        GemmCoord blockShape = L1TileShape::ToCoord();
        for (uint32_t loopIdx = aicoreIndex; loopIdx < coreLoops; loopIdx += aicoreNum) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            // Get the data and layout of C under the current basic block
            auto gmBlockC = gmC[layoutC.GetOffset(blockCoord.GetCoordMN() * blockShape.GetCoordMN())];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());
            // Synchronize cross core
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);
            // Actual calculatioin logic for performing block-scoped epilogue
            blockEpilogue(blockShape, blockCoord, actualBlockShape, gmBlockC, layoutBlockC);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_HPP
