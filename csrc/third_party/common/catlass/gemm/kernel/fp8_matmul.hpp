/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/cross_core_sync.hpp"

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t mScalar, uint32_t nScalar,
    uint32_t splitkLength>
class FP8Matmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementA_ = typename std::conditional_t<
        std::is_same_v<typename BlockMmad::PrologueA, void>, 
        ElementA,
        typename BlockMmad::PrologueA::ElementSrc>;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutA_ = typename std::conditional_t<
        std::is_same_v<typename BlockMmad::PrologueA, void>,
        LayoutA,
        typename BlockMmad::PrologueA::LayoutSrc>; // LayoutPrologueA
    using ElementB = typename BlockMmad::ElementB;
    using ElementB_ = typename std::conditional_t<
        std::is_same_v<typename BlockMmad::PrologueB, void>,
        ElementB,
        typename BlockMmad::PrologueB::ElementSrc>;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutB_ = typename std::conditional_t<
        std::is_same_v<typename BlockMmad::PrologueB, void>,
        LayoutB,
        typename BlockMmad::PrologueB::LayoutSrc>; 
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using MmadParams = typename BlockMmad_::Params;
    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA_ layoutA;
        GM_ADDR ptrB;
        LayoutB_ layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrWC;
        // half scalar;
        // half zeroPoint;

        MmadParams mmadParams;
        // MMadParams: {prologueAParams, prologueBParams, mScalar, nScalar, splitkLength}

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_, 
            GM_ADDR ptrA_, LayoutA layoutA_, 
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrC_, LayoutC layoutC_, 
            GM_ADDR ptrWA_, GM_ADDR ptrWB_, GM_ADDR ptrWC_, 
            MmadParams mmadParams_)
            : problemShape(problemShape_), 
              ptrA(ptrA_), layoutA(layoutA_), 
              ptrB(ptrB_), layoutB(layoutB_), 
              ptrC(ptrC_), layoutC(layoutC_), 
              ptrWA(ptrWA_), ptrWB(ptrWB_), 
              ptrWC(ptrWC_), mmadParams(mmadParams_)
        {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
        half scalar;
        half zeroPoint;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return GetSizeWA(args.aicCoreNum) + GetSizeWB(args.aicCoreNum) + GetSizeWC(args.aicCoreNum);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};

        // Malloc memory @[npu device]
        uint8_t *gmWA = nullptr;
        uint8_t *gmWB = nullptr;
        uint8_t *gmWC = nullptr;
        gmWA = workspace;
        workspace += GetSizeWA(args.aicCoreNum);
        gmWB = workspace;
        workspace += GetSizeWB(args.aicCoreNum);
        gmWC = workspace;

        Params params{args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            gmWA,
            gmWB,
            gmWC,
            {{args.scalar, args.zeroPoint}, 
            {args.scalar, args.zeroPoint}, mScalar, nScalar, splitkLength} };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    FP8Matmul()
    {
        flag0[0].id = flagID0;
        flag0[1].id = flagID1;
    }

    /// Executes one GEMM
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    __attribute__((always_inline)) void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        BlockMmad blockMmad(resource, params.mmadParams);
        BlockScheduler blockScheduler(
            params.problemShape, MakeCoord((L1TileShape::M * mScalar), (L1TileShape::N * nScalar)));

        AscendC::GlobalTensor<int8_t> gmA;
        gmA.SetGlobalBuffer((__gm__ int8_t *)params.ptrA);
        AscendC::GlobalTensor<half> gmWA;
        gmWA.SetGlobalBuffer((__gm__ half *)params.ptrWA);

        AscendC::GlobalTensor<int8_t> gmB;
        gmB.SetGlobalBuffer((__gm__ int8_t *)params.ptrB);
        AscendC::GlobalTensor<half> gmWB;
        gmWB.SetGlobalBuffer((__gm__ half *)params.ptrWB);

        AscendC::GlobalTensor<half> gmC;
        gmC.SetGlobalBuffer((__gm__ half *)params.ptrC);
        AscendC::GlobalTensor<float> gmWC;
        gmWC.SetGlobalBuffer((__gm__ float *)params.ptrWC);

        uint32_t coreLoops = blockScheduler.GetCoreLoops();
        for (uint32_t loopIdx = AscendC::GetBlockIdx() / AIVPERCORE; loopIdx < coreLoops;
             loopIdx += AscendC::GetBlockNum()) {  // 一次for循环完成两个行块或者两个列块的反量化
            // 当前任务块信息
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            // 下一个任务块的信息
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            bool isFirstBlock = (loopIdx == (AscendC::GetBlockIdx() / AIVPERCORE));
            bool hasNextBlock = (loopIdx + AscendC::GetBlockNum() < coreLoops);
            if (hasNextBlock) {
                nextBlockCoord = blockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = blockScheduler.GetActualBlockShape(nextBlockCoord);
            }
            
            blockMmad.Prologue(
                gmWA, gmA, params.layoutA,
                gmWB, gmB, params.layoutB, 
                gmC, gmWC, params.layoutC,
                blockCoord, nextBlockCoord,
                actualBlockShape, nextActualBlockShape,
                params.problemShape,
                isFirstBlock, hasNextBlock,
                bufferIndex
            );
            
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockMmad blockMmad(resource, params.mmadParams);
        BlockScheduler blockScheduler(
            params.problemShape, MakeCoord((L1TileShape::M * mScalar), (L1TileShape::N * nScalar)));

        AscendC::GlobalTensor<half> gmWA;
        gmWA.SetGlobalBuffer((__gm__ half *)params.ptrWA);
        AscendC::GlobalTensor<half> gmWB;
        gmWB.SetGlobalBuffer((__gm__ half *)params.ptrWB);
        AscendC::GlobalTensor<float> gmWC;
        gmWC.SetGlobalBuffer((__gm__ float *)params.ptrWC);

        uint32_t coreLoops = blockScheduler.GetCoreLoops();
        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops;
             loopIdx += AscendC::GetBlockNum()) {  // 一次for循环完成一个大基本结果块(256,512)
            // Compute block location
            // 获取当前大基本结果块的左上角坐标以及实际大小
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBigBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            blockMmad(gmWA, gmWB, gmWC, actualBigBlockShape, params.problemShape);
        }
    }

protected:
    static size_t GetSizeWA(const uint32_t aicCoreNum) 
    {
        size_t lenWA = static_cast<size_t>(splitkLength) * 128 * mScalar;
        return aicCoreNum * lenWA * sizeof(ElementA) * 2; // 双缓冲
    }

    static size_t GetSizeWB(const uint32_t aicCoreNum) 
    {
        size_t lenWB = static_cast<size_t>(splitkLength) * 256 * nScalar;
        return aicCoreNum * lenWB * sizeof(ElementB) * 2; // 双缓冲
    }

    static size_t GetSizeWC(const uint32_t aicCoreNum) 
    {
        size_t lenWC = static_cast<size_t>(128 * mScalar) * 256 * nScalar;
        return aicCoreNum * lenWC * sizeof(ElementC);
    }

protected:
    Arch::Resource<ArchTag> resource;

    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t AIVPERCORE = 2;

    static constexpr Arch::FlagID flagID0 = 0;
    static constexpr Arch::FlagID flagID1 = 1;

    Arch::CrossCoreFlag flag0[STAGES];
    uint32_t bufferIndex{0};
    
};

}  // namespace Catlass::Gemm::Kernel

#endif  // CATLASS_GEMM_KERNEL_FP8_MATMUL_HPP