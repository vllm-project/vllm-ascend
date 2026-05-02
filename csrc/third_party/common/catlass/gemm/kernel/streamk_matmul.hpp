/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_STREAMK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_STREAMK_MATMUL_HPP

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Kernel {

template <class ArchTag_, class BlockScheduler_, class ElementAccumulator_, class ElementOut_, uint32_t COMPUTE_LENGTH>
struct StreamkReduceAdd {
    using ArchTag = ArchTag_;
    using BlockScheduler = BlockScheduler_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOut = ElementOut_;
    using LocalLayout = layout::RowMajor;

    constexpr static uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementOut);

    CATLASS_DEVICE
    StreamkReduceAdd(Arch::Resource<ArchTag> &resource, int64_t bufferOffset = 0)
    {
        accumulatorBuffer = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(bufferOffset);
        outputBuffer = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementOut> const &dst,
        AscendC::GlobalTensor<ElementAccumulator> const &src,
        BlockScheduler &matmulBlockScheduler,
        LocalLayout layoutDst
    )
    {
        constexpr uint32_t ELE_NUM_ALIGN = BYTE_PER_BLK / sizeof(ElementOut);

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t aivNum = blockDim * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t aicId = aivId / AscendC::GetSubBlockNum();

        uint32_t l1TileM = matmulBlockScheduler.tileMNK.m();
        uint32_t l1TileN = matmulBlockScheduler.tileMNK.n();
        uint32_t l1TileK = matmulBlockScheduler.tileMNK.k();

        // The number of tail block using the streamk algorithm
        uint32_t streamkBlockNum = matmulBlockScheduler.GetStreamkBlockNum();
        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        for (uint32_t skBlockId = 0; skBlockId < streamkBlockNum; ++skBlockId) {
            // If the head part of the current block and tail part of previous block are computed by the same core,
            // this flag is set to true.
            bool isHeadCross = matmulBlockScheduler.IsCross(skBlockId);
            bool isTailCross = matmulBlockScheduler.IsCross(skBlockId + 1);
            // Get the ID of first core for the current skBlock computation.
            uint32_t startCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId);
            // Get the ID of first core for the next skBlock computation.
            uint32_t endCoreIdx = matmulBlockScheduler.GetCoreIdx(skBlockId + 1);

            uint32_t laborCoreNum = (endCoreIdx - startCoreIdx) * AscendC::GetSubBlockNum();

            if ((aicId < startCoreIdx) || (aicId >= endCoreIdx)) {
                continue;
            }

            if (isTailCross) {
                endCoreIdx = endCoreIdx + 1;
            }

            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(normalBlockNum + skBlockId);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t tileNum = actualBlockShape.m();
            uint32_t tileLen = actualBlockShape.n();

            uint32_t tilePerCoreMax = (COMPUTE_LENGTH / laborCoreNum) / RoundUp(tileLen, ELE_NUM_ALIGN);
            uint32_t tilePerCore = CeilDiv(tileNum, laborCoreNum);
            if (tilePerCore > tilePerCoreMax) {
                tilePerCore = tilePerCoreMax;
            }
            uint32_t splitkSliceNum = endCoreIdx - startCoreIdx;
            uint32_t loopsNum = CeilDiv(tileNum, tilePerCore);
            uint32_t loopStart = aivId - startCoreIdx * AscendC::GetSubBlockNum();

            for (uint32_t loopIdx = loopStart; loopIdx < loopsNum; loopIdx += laborCoreNum) {
                uint32_t tilesActual = tilePerCore;
                if (loopIdx == loopsNum - 1) {
                    tilesActual = tileNum - loopIdx * tilePerCore;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::DataCopyExtParams dataCopyParamsIn(
                    tilesActual, tileLen * sizeof(ElementAccumulator), (l1TileN - tileLen) * sizeof(ElementAccumulator),
                    (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) * sizeof(ElementAccumulator) / BYTE_PER_BLK, 0
                );
                AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
                if (isHeadCross) {
                    uint64_t skBlockOffset = (startCoreIdx * 2 + 1) * l1TileM * l1TileN;
                    uint64_t srcOffset = skBlockOffset + loopIdx * tilePerCore * l1TileN;
                    AscendC::DataCopyPad(accumulatorBuffer, src[srcOffset], dataCopyParamsIn, padParams);
                }

                uint32_t sliceStart = isHeadCross ? 1 : 0;
                uint32_t computeNum = tilesActual * RoundUp(tileLen, ELE_NUM_ALIGN);
                for (uint32_t sliceIdx = sliceStart; sliceIdx < splitkSliceNum; ++sliceIdx) {
                    uint64_t skBlockOffset = (sliceIdx + startCoreIdx) * 2 * l1TileM * l1TileN;
                    uint64_t srcOffset = skBlockOffset + loopIdx * tilePerCore * l1TileN;
                    uint64_t dstOffset = computeNum * sliceIdx;
                    AscendC::DataCopyPad(accumulatorBuffer[dstOffset], src[srcOffset], dataCopyParamsIn, padParams);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                for (uint32_t sliceIdx = 1; sliceIdx < splitkSliceNum; ++sliceIdx) {
                    AscendC::Add(
                        accumulatorBuffer, accumulatorBuffer, accumulatorBuffer[computeNum * sliceIdx], computeNum
                    );
                    AscendC::PipeBarrier<PIPE_V>();
                }

                if constexpr (!std::is_same_v<ElementAccumulator, ElementOut>) {
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer, accumulatorBuffer, AscendC::RoundMode::CAST_NONE, computeNum);
                    } else {
                        AscendC::Cast(outputBuffer, accumulatorBuffer, AscendC::RoundMode::CAST_RINT, computeNum);
                    }
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                // layoutDst can only be RowMajor
                uint64_t dstStride = layoutDst.stride(0);
                uint64_t dstOffset = (static_cast<uint64_t>(blockCoord.m()) * l1TileM + loopIdx * tilePerCore)
                                         * dstStride
                                     + static_cast<uint64_t>(blockCoord.n()) * l1TileN;
                AscendC::DataCopyExtParams dataCopyParamsOut(
                    tilesActual, tileLen * sizeof(ElementOut),
                    (RoundUp(tileLen, ELE_NUM_ALIGN) - tileLen) / ELE_NUM_ALIGN,
                    (dstStride - tileLen) * sizeof(ElementOut), 0
                );
                AscendC::DataCopyPad(dst[dstOffset], outputBuffer, dataCopyParamsOut);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementAccumulator> accumulatorBuffer;
    AscendC::LocalTensor<ElementOut> outputBuffer;
    static_assert(COMPUTE_LENGTH * sizeof(ElementAccumulator) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ReduceAdd_
>
class StreamkMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ReduceAdd = ReduceAdd_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

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
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_,
            LayoutA &layoutA_,
            GM_ADDR ptrB_,
            LayoutB &layoutB_,
            GM_ADDR ptrC_,
            LayoutC &layoutC_,
            GM_ADDR ptrWorkspace_
        )
            : problemShape(problemShape_)
            , ptrA(ptrA_)
            , layoutA(layoutA_)
            , ptrB(ptrB_)
            , layoutB(layoutB_)
            , ptrC(ptrC_)
            , layoutC(layoutC_)
            , ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t aicCoreNum;
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
        size_t minSpaceSize = 2 * 1024 * 1024; // 2M
        size_t workspaceSize = static_cast<size_t>(L1TileShape::M) * L1TileShape::N * sizeof(ElementAccumulator) *
            args.aicCoreNum * 2;
        return minSpaceSize > workspaceSize ? minSpaceSize : workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(args.problemShape.m(), args.problemShape.n());
        Params params{args.problemShape,
            args.ptrA, layoutA,
            args.ptrB, layoutB,
            args.ptrC, layoutC, workspace};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    StreamkMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    /// Executes matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        BlockScheduler matmulBlockScheduler(
            params.problemShape, GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), blockDim
        );
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementAccumulator> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrWorkspace);

        typename BlockScheduler::StreamkBlockDec streamkBlockDec;
        typename BlockScheduler::StreamkBlockDec nextStreamkBlockDec;

        // The number of normal blocks
        uint32_t normalBlockNum = matmulBlockScheduler.GetNormalBlockNum();
        uint32_t streamkCores = coreLoops - normalBlockNum;
        if (blockIdx >= streamkCores) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
        }

        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += blockDim) {

            uint32_t actualLoopIdx = loopIdx;
            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < streamkCores && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum + blockIdx;
            } else if (loopIdx >= normalBlockNum && normalBlockNum > 0) {
                actualLoopIdx = normalBlockNum - blockDim + blockIdx;
            }
            uint32_t actualNextLoopIdx = loopIdx + blockDim;
            if ((loopIdx + blockDim) == normalBlockNum - blockDim + blockIdx && blockIdx < streamkCores
                && normalBlockNum > 0) {
                actualNextLoopIdx = normalBlockNum + blockIdx;
            } else if ((loopIdx + blockDim) >= normalBlockNum && normalBlockNum > 0) {
                actualNextLoopIdx = normalBlockNum - blockDim + blockIdx;
            }

            bool isFirstBlock = (loopIdx == blockIdx);
            if (isFirstBlock) {
                matmulBlockScheduler.GetStreamkBlockDec(actualLoopIdx, streamkBlockDec);
            } else {
                streamkBlockDec = nextStreamkBlockDec;
            }

            bool hasNextBlock = false;
            if (loopIdx + blockDim < coreLoops) {
                hasNextBlock = true;
                matmulBlockScheduler.GetStreamkBlockDec(actualNextLoopIdx, nextStreamkBlockDec);
            }

            // Compute initial location in logical coordinates
            MatrixCoord coordA{
                streamkBlockDec.blockCoord.m() * L1TileShape::M,
                streamkBlockDec.blockCoord.k() * L1TileShape::K
            };
            MatrixCoord coordB{
                streamkBlockDec.blockCoord.k() * L1TileShape::K,
                streamkBlockDec.blockCoord.n() * L1TileShape::N
            };
            MatrixCoord coordC{
                streamkBlockDec.blockCoord.m() * L1TileShape::M,
                streamkBlockDec.blockCoord.n() * L1TileShape::N
            };
            int64_t gmOffsetA = params.layoutA.GetOffset(coordA);
            int64_t gmOffsetB = params.layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC);
            MatrixCoord coordNextA{
                nextStreamkBlockDec.blockCoord.m() * L1TileShape::M,
                nextStreamkBlockDec.blockCoord.k() * L1TileShape::K
            };
            MatrixCoord coordNextB{
                nextStreamkBlockDec.blockCoord.k() * L1TileShape::K,
                nextStreamkBlockDec.blockCoord.n() * L1TileShape::N
            };
            int64_t gmOffsetNextA = params.layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = params.layoutB.GetOffset(coordNextB);

            int64_t gmOffsetW = L1TileShape::M * L1TileShape::N * 2 * blockIdx;
            LayoutC layoutW = LayoutC{
                streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.n(), L1TileShape::N
            };

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC, gmW[gmOffsetW],
                layoutW, gmA[gmOffsetNextA], gmB[gmOffsetNextB], streamkBlockDec.actualBlockShape,
                nextStreamkBlockDec.actualBlockShape, isFirstBlock, hasNextBlock && (!streamkBlockDec.isCrossBlock),
                streamkBlockDec.isStreamkBlock
            );
            // If the current block is a cross block-meaning it consists partly of the tail k portion of
            // current block and partly of head k portion of the next block-then a second blockmmad
            // computation must be invoked to process the head k segment of the next block.
            if (streamkBlockDec.isCrossBlock) {
                MatrixCoord coordA{
                    streamkBlockDec.streamkBlockCoord.m() * L1TileShape::M,
                    streamkBlockDec.streamkBlockCoord.k() * L1TileShape::K
                };
                MatrixCoord coordB{
                    streamkBlockDec.streamkBlockCoord.k() * L1TileShape::K,
                    streamkBlockDec.streamkBlockCoord.n() * L1TileShape::N
                };
                int64_t gmOffsetA = params.layoutA.GetOffset(coordA);
                int64_t gmOffsetB = params.layoutB.GetOffset(coordB);
                int64_t gmOffsetW = L1TileShape::M * L1TileShape::N * (2 * blockIdx + 1);
                LayoutC layoutW = LayoutC{
                    streamkBlockDec.streamkActualBlockShape.m(), streamkBlockDec.streamkActualBlockShape.n(),
                    L1TileShape::N
                };
                blockMmad(
                    gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], params.layoutC, gmW[gmOffsetW],
                    layoutW, gmA[gmOffsetNextA], gmB[gmOffsetNextB], streamkBlockDec.streamkActualBlockShape,
                    nextStreamkBlockDec.actualBlockShape, true, hasNextBlock, streamkBlockDec.isStreamkBlock
                );
            }

            if (loopIdx == normalBlockNum - blockDim + blockIdx && blockIdx < streamkCores && normalBlockNum > 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
            }
            if (normalBlockNum == 0) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmReduceW;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut *>(params.ptrC));
        gmReduceW.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrWorkspace));
        ReduceAdd reduceAdd(resource);
        BlockScheduler matmulBlockScheduler(
            params.problemShape, GemmCoord(L1TileShape::M, L1TileShape::N, L1TileShape::K), AscendC::GetBlockNum()
        );
        reduceAdd(gmC, gmReduceW, matmulBlockScheduler, params.layoutC);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;

    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_STREAMK_MATMUL_HPP