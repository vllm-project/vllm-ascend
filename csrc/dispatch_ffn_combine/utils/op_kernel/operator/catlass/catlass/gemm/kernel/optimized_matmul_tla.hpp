/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_TLA_HPP
#define CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
namespace Catlass::Gemm::Kernel {

template<
    class ArchTag_,
    class TensorIn_,
    class TensorOut_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixBlockND {
public:
    using ArchTag = ArchTag_;
    using TensorIn = TensorIn_;
    using TensorOut = TensorOut_;
    using Element = typename TensorIn::Element;
    using LayoutIn = typename TensorIn::Layout;
    using LayoutOut = typename TensorOut::Layout;

    using LayoutInner = tla::Layout<tla::Shape<uint32_t, uint32_t>, tla::Stride<int64_t, tla::Int<1>>>;
    using TensorInnerUb = tla::Tensor<AscendC::LocalTensor<Element>, LayoutInner, AscendC::TPosition::VECCALC>;
    using TensorInnerSrcGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInner, AscendC::TPosition::GM>;

    using LayoutInnerDstGm = tla::Layout<
        tla::Shape<tla::Shape<uint32_t, uint32_t>, tla::Shape<uint32_t, uint32_t>>,
        tla::Stride<tla::Stride<int64_t, int64_t>, tla::Stride<tla::Int<1>, int64_t>>>;
    using TensorInnerDstGm = tla::Tensor<AscendC::GlobalTensor<Element>, LayoutInnerDstGm, AscendC::TPosition::GM>;

    using CopyGm2Ub = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorInnerSrcGm, TensorInnerUb>;
    using CopyUb2Gm = Catlass::Gemm::Tile::TileCopyTlaExt<ArchTag, TensorInnerUb,
        TensorInnerDstGm, layout::RowMajor, layout::PaddingRowMajor>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrixBlockND(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            //在ub上分配空间
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            //每一片UB上的开均分到BUFFER_NUM的空间
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    template<class Tensor>
    CATLASS_DEVICE
    auto GetPaddingTensorSrc(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInner>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    template<class Tensor>
    CATLASS_DEVICE
    auto GetPaddingTensorDst(Tensor const &tensor)
    {
        if constexpr (std::is_same_v<typename Tensor::Layout, LayoutInnerDstGm>) {
            return tensor;
        } else {
            auto shape = tla::MakeShape(tla::get<1>(tensor.shape()), tla::get<0>(tensor.shape()));
            auto stride = tla::MakeStride(tla::get<1>(tensor.stride()), tla::get<0>(tensor.stride()));
            return tla::MakeTensor(tensor.data(), MakeLayout(shape, stride), Arch::PositionGM{});
        }
    }

    CATLASS_DEVICE
    void operator()(TensorOut &tensorDst, TensorIn const& tensorSrc)
    {
        auto paddingTensorSrc = GetPaddingTensorSrc(tensorSrc);
        auto paddingTensorDst = GetPaddingTensorDst(tensorDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // 按照行切块，每行为一个tile块
        uint32_t tilesNum = tla::get<0>(paddingTensorSrc.shape());
        uint32_t tileLen = tla::get<1>(paddingTensorSrc.shape());
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(tla::get<1>(paddingTensorSrc.shape()));
        // 计算每一个aiv要计算的大小，对于剩余的工作从前向后增加
        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        //因为前面进行了工作重分配，所以相应后面的aiv处理的偏移量要后移
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);
        // 配置UB到GM的信号量
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{ 0 };
        if (roundTileLen > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                auto offset = tla::MakeCoord(mIdx + tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );
                auto tensorTileDst = GetTile(
                    paddingTensorDst,
                    offset,
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum)
                );

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(static_cast<uint32_t>(1), actualDataNum),
                    tla::MakeStride(static_cast<int64_t>(COMPUTE_LENGTH), tla::Int<1>{})
                );
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    tla::MakeShape(
                        CeilDiv(actualDataNum, tla::get<1, 0>(paddingTensorDst.shape())),
                        tla::get<1, 0>(paddingTensorDst.shape())
                    ),
                    tla::MakeStride(static_cast<int64_t>(tla::get<1, 0>(paddingTensorDst.shape())), tla::Int<1>{})
                );
                auto tensorSrcUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutSrcUb, Arch::PositionUB{});
                copyUb2Gm(tensorTileDst, tensorSrcUb);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                auto offset = tla::MakeCoord(mIdx + tileIdx, static_cast<uint32_t>(0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                auto tensorTileSrc = GetTile(
                    paddingTensorSrc,
                    offset,
                    tla::MakeShape(actualTilesNum, tileLen)
                );

                auto layoutDstUb = MakeLayout(
                    tla::MakeShape(actualTilesNum, tileLen),
                    tla::MakeStride(static_cast<int64_t>(roundTileLen), tla::Int<1>{})
                );
                auto tensorDstUb = tla::MakeTensor(inputBuffer[bufferIndex], layoutDstUb, Arch::PositionUB{});

                copyGm2Ub(tensorDstUb, tensorTileSrc);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                auto layoutSrcUb = MakeLayout(
                    tla::MakeShape(
                        CeilDiv(tileLen, tla::get<1, 0>(paddingTensorDst.shape())),
                        tla::get<1, 0>(paddingTensorDst.shape())
                    ),
                    tla::MakeStride(static_cast<int64_t>(tla::get<1, 0>(paddingTensorDst.shape())), tla::Int<1>{})
                );
                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    auto tensorTileDst = GetTile(
                        paddingTensorDst,
                        tla::MakeCoord(mIdx + tileIdx + i, static_cast<uint32_t>(0)),
                        tla::MakeShape(static_cast<uint32_t>(1), tileLen)
                    );
                    auto tensorSrcUb = tla::MakeTensor(
                        inputBuffer[bufferIndex][i * roundTileLen],
                        layoutSrcUb,
                        Arch::PositionUB{}
                    );
                    copyUb2Gm(tensorTileDst, tensorSrcUb);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~PaddingMatrixBlockND() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class PaddingA,
    class PaddingB
>
class OptimizedMatmulTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };
    using LayoutA = std::conditional_t<std::is_void_v<PaddingA>, LayoutWA, typename LayoutHelper<PaddingA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PaddingB>, LayoutWB, typename LayoutHelper<PaddingB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

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
        uint8_t *ptrA; LayoutA layoutA;
        uint8_t *ptrB; LayoutB layoutB;
        uint8_t *ptrC; LayoutC layoutC;
        uint8_t *ptrWA; LayoutWA layoutWA;
        uint8_t *ptrWB; LayoutWB layoutWB;
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
        Params params{args.problemShape,
            args.ptrA, args.layoutA,
            args.ptrB, args.layoutB,
            args.ptrC, args.layoutC,
            args.ptrWA, args.layoutWA,
            args.ptrWB, args.layoutWB
            };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    OptimizedMatmulTla() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PaddingA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            auto tensorA = tla::MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutA, AscendC::TPosition::GM>(
                gmA, params.layoutA);
            auto tensorWA = tla::MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutWA, AscendC::TPosition::GM>(
                gmWA, params.layoutWA);
            PaddingA paddingA(resource);
            paddingA(tensorWA, tensorA);
        }

        if constexpr (!std::is_void_v<PaddingB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            auto tensorB = tla::MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutB, AscendC::TPosition::GM>(
                gmB, params.layoutB);
            auto tensorWB = tla::MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutWB, AscendC::TPosition::GM>(
                gmWB, params.layoutWB);
            PaddingB paddingB(resource);
            paddingB(tensorWB, tensorB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        if (!std::is_void_v<PaddingA> || !std::is_void_v<PaddingB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        auto tensorA = tla::MakeTensor<AscendC::GlobalTensor<ElementA>, LayoutWA, AscendC::TPosition::GM>(
            gmA, params.layoutWA);
        auto tensorB = tla::MakeTensor<AscendC::GlobalTensor<ElementB>, LayoutWB, AscendC::TPosition::GM>(
            gmB, params.layoutWB);
        auto tensorC = tla::MakeTensor<AscendC::GlobalTensor<ElementC>, LayoutC, AscendC::TPosition::GM>(
            gmC, params.layoutC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            auto tensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
            );
            auto tensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
            );
            auto tensorBlockC = GetTile(
                tensorC,
                tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
            );

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockCoord);
            }

            auto nextTensorBlockA = GetTile(
                tensorA,
                tla::MakeCoord(nextBlockCoord.m() * L1_TILE_M, nextBlockCoord.k() * L1_TILE_K),
                tla::MakeShape(nextActualBlockShape.m(), nextActualBlockShape.k())
            );
            auto nextTensorBlockB = GetTile(
                tensorB,
                tla::MakeCoord(nextBlockCoord.k() * L1_TILE_K, nextBlockCoord.n() * L1_TILE_N),
                tla::MakeShape(nextActualBlockShape.k(), nextActualBlockShape.n())
            );

            // Compute block-scoped matrix multiply-add
            blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockC, nextTensorBlockA, nextTensorBlockB,
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock
            );
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_TLA_HPP
