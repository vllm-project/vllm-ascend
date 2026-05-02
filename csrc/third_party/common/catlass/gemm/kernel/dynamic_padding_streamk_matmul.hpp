/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP

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

template <class PrologueA_, class PrologueB_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, class ReduceAdd_>
class DynamicPaddingStreamkMatmul {
public:
    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;
    using ReduceAdd = ReduceAdd_;

    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using LayoutC = typename BlockMmad::LayoutC;

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::
        conditional_t<std::is_void_v<PrologueA>, typename BlockMmad::LayoutA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::
        conditional_t<std::is_void_v<PrologueB>, typename BlockMmad::LayoutB, typename LayoutHelper<PrologueB>::type>;

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
        GM_ADDR ptrWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrReduceW;
        uint32_t swizzleOffset;
        uint32_t swizzleDirection;

        // Methods
        CATLASS_HOST_DEVICE
        Params()
        {
        }

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GemmCoord const &l1TileShape_,
            GM_ADDR ptrA_,
            LayoutA &layoutA_,
            GM_ADDR ptrB_,
            LayoutB &layoutB_,
            GM_ADDR ptrC_,
            LayoutC &layoutC_,
            GM_ADDR ptrWA_,
            GM_ADDR ptrWB_,
            GM_ADDR ptrReduceW_,
            uint32_t swizzleOffset_,
            uint32_t swizzleDirection_
        )
            : problemShape(problemShape_)
            , l1TileShape(l1TileShape_)
            , ptrA(ptrA_)
            , layoutA(layoutA_)
            , ptrB(ptrB_)
            , layoutB(layoutB_)
            , ptrC(ptrC_)
            , layoutC(layoutC_)
            , ptrWA(ptrWA_)
            , ptrWB(ptrWB_)
            , ptrReduceW(ptrReduceW_)
            , swizzleOffset(swizzleOffset_)
            , swizzleDirection(swizzleDirection_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    DynamicPaddingStreamkMatmul()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params, Catlass::Arch::Resource<ArchTag> &resource);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            typename BlockMmad::LayoutA layoutWA;
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            typename BlockMmad::LayoutB layoutWB;
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutWB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }

        using ElementOut = typename ReduceAdd::ElementOut;
        using ElementAccumulator = typename ReduceAdd::ElementAccumulator;

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        AscendC::GlobalTensor<ElementOut> gmC;
        AscendC::GlobalTensor<ElementAccumulator> gmReduceW;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOut *>(params.ptrC));
        gmReduceW.SetGlobalBuffer(reinterpret_cast<__gm__ ElementAccumulator *>(params.ptrReduceW));
        ReduceAdd reduceAdd(resource);
        BlockScheduler matmulBlockScheduler(
            params.problemShape, params.l1TileShape, AscendC::GetBlockNum(), params.swizzleOffset,
            params.swizzleDirection
        );
        reduceAdd(gmC, gmReduceW, matmulBlockScheduler, params.layoutC);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params, Catlass::Arch::Resource<ArchTag> &resource)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        uint32_t blockDim = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        BlockScheduler matmulBlockScheduler(
            params.problemShape, params.l1TileShape, blockDim, params.swizzleOffset, params.swizzleDirection
        );
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
            if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, 512 / sizeof(ElementA));
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA, params.l1TileShape.m(), params.l1TileShape.k());
            } else if constexpr (PrologueA::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutA = PrologueA::GetWorkspaceLayout(params.layoutA);
            }
        }
        AscendC::GlobalTensor<ElementB> gmB;
        if constexpr (std::is_void_v<PrologueB>) {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
            layoutB = params.layoutB;
        } else {
            gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
            if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, 512 / sizeof(ElementB));
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_BLOCK_ND) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB, params.l1TileShape.k(), params.l1TileShape.n());
            } else if constexpr (PrologueB::paddingTag == Catlass::Gemm::Kernel::PaddingTag::PADDING_NZ) {
                layoutB = PrologueB::GetWorkspaceLayout(params.layoutB);
            }
        }
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementAccumulator> gmW;
        gmW.SetGlobalBuffer((__gm__ ElementAccumulator *)params.ptrReduceW);

        BlockMmad blockMmad(params.l1TileShape, resource);

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
                streamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                streamkBlockDec.blockCoord.k() * params.l1TileShape.k()
            };
            MatrixCoord coordB{
                streamkBlockDec.blockCoord.k() * params.l1TileShape.k(),
                streamkBlockDec.blockCoord.n() * params.l1TileShape.n()
            };
            MatrixCoord coordC{
                streamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                streamkBlockDec.blockCoord.n() * params.l1TileShape.n()
            };
            int64_t gmOffsetA = layoutA.GetOffset(coordA);
            int64_t gmOffsetB = layoutB.GetOffset(coordB);
            int64_t gmOffsetC = params.layoutC.GetOffset(coordC);
            MatrixCoord coordNextA{
                nextStreamkBlockDec.blockCoord.m() * params.l1TileShape.m(),
                nextStreamkBlockDec.blockCoord.k() * params.l1TileShape.k()
            };
            MatrixCoord coordNextB{
                nextStreamkBlockDec.blockCoord.k() * params.l1TileShape.k(),
                nextStreamkBlockDec.blockCoord.n() * params.l1TileShape.n()
            };
            int64_t gmOffsetNextA = layoutA.GetOffset(coordNextA);
            int64_t gmOffsetNextB = layoutB.GetOffset(coordNextB);

            int64_t gmOffsetW = params.l1TileShape.m() * params.l1TileShape.n() * 2 * blockIdx;
            LayoutC layoutW = LayoutC{
                streamkBlockDec.actualBlockShape.m(), streamkBlockDec.actualBlockShape.n(), params.l1TileShape.n()
            };

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], params.layoutC, gmW[gmOffsetW],
                layoutW, gmA[gmOffsetNextA], gmB[gmOffsetNextB], streamkBlockDec.actualBlockShape,
                nextStreamkBlockDec.actualBlockShape, isFirstBlock, hasNextBlock && (!streamkBlockDec.isCrossBlock),
                streamkBlockDec.isStreamkBlock
            );
            // If the current block is a cross block-meaning it consists partly of the tail k portion of
            // current block and partly of head k portion of the next block-then a second blockmmad
            // computation must be invoked to process the head k segment of the next block.
            if (streamkBlockDec.isCrossBlock) {
                MatrixCoord coordA{
                    streamkBlockDec.streamkBlockCoord.m() * params.l1TileShape.m(),
                    streamkBlockDec.streamkBlockCoord.k() * params.l1TileShape.k()
                };
                MatrixCoord coordB{
                    streamkBlockDec.streamkBlockCoord.k() * params.l1TileShape.k(),
                    streamkBlockDec.streamkBlockCoord.n() * params.l1TileShape.n()
                };
                int64_t gmOffsetA = layoutA.GetOffset(coordA);
                int64_t gmOffsetB = layoutB.GetOffset(coordB);
                int64_t gmOffsetW = params.l1TileShape.m() * params.l1TileShape.n() * (2 * blockIdx + 1);
                LayoutC layoutW = LayoutC{
                    streamkBlockDec.streamkActualBlockShape.m(), streamkBlockDec.streamkActualBlockShape.n(),
                    params.l1TileShape.n()
                };
                blockMmad(
                    gmA[gmOffsetA], layoutA, gmB[gmOffsetB], layoutB, gmC[gmOffsetC], params.layoutC, gmW[gmOffsetW],
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

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIC_FINISH = 1;
    Arch::CrossCoreFlag flagAicFinish{FLAG_AIC_FINISH};
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_DYNAMIC_PADDING_STREAMK_MATMUL_HPP