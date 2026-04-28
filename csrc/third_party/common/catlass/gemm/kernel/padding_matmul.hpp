/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_PADDING_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_PADDING_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catlass::Gemm::Kernel {

enum class PaddingTag {
    NO_PADDING,
    PADDING_ND,
    PADDING_BLOCK_ND,
    PADDING_NZ    
};

template<
    class ArchTag_,
    class Element_,
    class LayoutIn_,
    class LayoutOut_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixNZ {
    using ArchTag = ArchTag_;
    using Element = Element_;
    using LayoutIn = LayoutIn_;
    using LayoutOut = LayoutOut_;
    using ComputeLayoutSrc = Catlass::layout::RowMajor;
    using ComputeLayoutDst = Catlass::layout::zN;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<Element, ComputeLayoutSrc>>;

    constexpr static uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    constexpr static uint32_t VNCHW_SIZE = 16;

    static const PaddingTag paddingTag = PaddingTag::PADDING_NZ;
    CATLASS_HOST_DEVICE static LayoutOut GetWorkspaceLayout(const LayoutIn& layout)
    {
        return LayoutOut::template MakeLayout<Element>(layout.shape(0), layout.shape(1));
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols)
    {
        if constexpr (std::is_same_v<LayoutIn, Catlass::layout::RowMajor>) {
            return static_cast<size_t>(
                RoundUp(rows, Catlass::C0_NUM_PER_FRACTAL)) * RoundUp(cols, ELE_NUM_PER_C0) * sizeof(Element);
        } else {
            return static_cast<size_t>(
                RoundUp(rows, ELE_NUM_PER_C0)) * RoundUp(cols, Catlass::C0_NUM_PER_FRACTAL) * sizeof(Element);
        }
    }

    CopyGm2Ub copyGm2Ub;

    CATLASS_DEVICE
    PaddingMatrixNZ(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
            outputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    void CopyUb2Ub(AscendC::LocalTensor<Element> const &dst, AscendC::LocalTensor<Element> const &src,
        ComputeLayoutDst const &layoutDst, ComputeLayoutSrc const &layoutSrc)
    {
       uint32_t loops = CeilDiv(layoutSrc.shape(0), BLK_NUM_PER_VECTOR_FRACTAL); 
       for (uint32_t i = 0; i < loops; ++i) {
            uint64_t offsetSrc = i * BLK_NUM_PER_VECTOR_FRACTAL * layoutSrc.stride(0);
            uint64_t offsetDst = i * BLK_NUM_PER_VECTOR_FRACTAL * layoutDst.stride(0);
            uint32_t repeatTimes = CeilDiv(layoutSrc.shape(1), ELE_NUM_PER_C0);
            uint64_t mask = 256 / sizeof(Element);
            AscendC::CopyRepeatParams copyRepeatParams {
                1, static_cast<uint16_t>(layoutSrc.stride(0) / ELE_NUM_PER_C0),
                static_cast<uint16_t>(layoutDst.stride(3) / ELE_NUM_PER_C0), 1
            };
            AscendC::Copy(dst[offsetDst], src[offsetSrc], mask, repeatTimes, copyRepeatParams);
       }
    }

    CATLASS_DEVICE
    void CopyUb2Gm(AscendC::GlobalTensor<Element> const &dst, AscendC::LocalTensor<Element> const &src,
        ComputeLayoutDst const &layoutDst, ComputeLayoutDst const &layoutSrc)
    {
        if (layoutDst.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            AscendC::DataCopyParams dataCopyParams(
                layoutDst.shape(3),
                layoutDst.shape(0) * layoutDst.shape(1),
                (layoutSrc.stride(3) - layoutSrc.shape(0) * layoutSrc.shape(1) * layoutSrc.shape(2)) / ELE_NUM_PER_C0,
                (layoutDst.stride(3) - layoutDst.shape(0) * layoutDst.shape(1) * layoutDst.shape(2)) / ELE_NUM_PER_C0
            );
            AscendC::DataCopy(dst, src, dataCopyParams);
        } else {
            uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1));
            uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0));
            AscendC::DataCopyParams dataCopyParams(1, blockLen, 0, 0);
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dst[dstOffset], src[srcOffset], dataCopyParams);
            }
        }
    }

    CATLASS_DEVICE
    ComputeLayoutSrc GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return layout;
    }

    CATLASS_DEVICE
    ComputeLayoutSrc GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayoutSrc(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CATLASS_DEVICE
    ComputeLayoutDst GetPaddingComputeLayout(layout::zN const &layout)
    {
        return layout;
    }
    CATLASS_DEVICE
    ComputeLayoutDst GetPaddingComputeLayout(layout::nZ const &layout)
    {
        return ComputeLayoutDst::MakeLayout<Element>(layout.orgShape(1), layout.orgShape(0));
    }

    CATLASS_DEVICE
    void CommonND2NZ(AscendC::GlobalTensor<Element> const &dst, AscendC::GlobalTensor<Element> const &src,
        ComputeLayoutDst const &computeLayoutDst, ComputeLayoutSrc const &computeLayoutSrc)
    {
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        uint32_t refTileRows = 16;
        uint32_t refTileCols = COMPUTE_LENGTH / refTileRows;
        uint32_t tileRows = refTileRows;
        uint32_t tileCols = RoundUp(
            computeLayoutSrc.shape(1) / CeilDiv(computeLayoutSrc.shape(1), refTileCols), ELE_NUM_PER_C0);
        
        uint32_t rowTiles = CeilDiv(computeLayoutSrc.shape(0), tileRows);
        uint32_t colTiles = CeilDiv(computeLayoutSrc.shape(1), tileCols);
        uint32_t totalTiles = rowTiles * colTiles;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);

        for (uint32_t loopIdx = aivId; loopIdx < totalTiles; loopIdx += aivNum) {
            uint32_t rowTileIdx = loopIdx / colTiles;
            uint32_t colTileIdx = loopIdx % colTiles;
            MatrixCoord tileCoord{rowTileIdx * tileRows, colTileIdx * tileCols};
            uint32_t rowTileActual = (rowTileIdx == rowTiles - 1) ?
                (computeLayoutSrc.shape(0) - rowTileIdx * tileRows) : tileRows;
            uint32_t colTileActual = (colTileIdx == colTiles - 1) ?
                (computeLayoutSrc.shape(1) - colTileIdx * tileCols) : tileCols;
            
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            uint64_t tileOffsetSrc = computeLayoutSrc.GetOffset(tileCoord);
            ComputeLayoutSrc tileLayoutSrc = computeLayoutSrc.GetTileLayout(MatrixCoord(rowTileActual, colTileActual));
            ComputeLayoutSrc ubLayoutIn = ComputeLayoutSrc{rowTileActual, RoundUp(colTileActual, ELE_NUM_PER_C0)};
            if ((tileCols < computeLayoutSrc.shape(1))
                || (computeLayoutSrc.shape(1) % ELE_NUM_PER_C0) 
                || (computeLayoutSrc.shape(1) != computeLayoutSrc.stride(0))) {
                copyGm2Ub(inputBuffer[bufferIndex], src[tileOffsetSrc], ubLayoutIn, tileLayoutSrc);
            } else {
                // Continuous copy to ub
                ComputeLayoutSrc tileLayoutSrcConti = 
                    computeLayoutSrc.GetTileLayout(MatrixCoord(1, rowTileActual * colTileActual));
                ComputeLayoutSrc ubLayoutInConti = ComputeLayoutSrc{1, rowTileActual * colTileActual};
                copyGm2Ub(inputBuffer[bufferIndex], src[tileOffsetSrc], ubLayoutInConti, tileLayoutSrcConti);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);
            ComputeLayoutDst ubLayoutOut = ComputeLayoutDst::template MakeLayout<Element>(rowTileActual, colTileActual);
            CopyUb2Ub(outputBuffer[bufferIndex], inputBuffer[bufferIndex], ubLayoutOut, ubLayoutIn);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);

            uint64_t tileOffsetDst = computeLayoutDst.GetOffset(tileCoord);
            ComputeLayoutDst tileLayoutDst = computeLayoutDst.GetTileLayout(MatrixCoord(rowTileActual, colTileActual));
            CopyUb2Gm(dst[tileOffsetDst], outputBuffer[bufferIndex], tileLayoutDst, ubLayoutOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);
            bufferIndex = 1 - bufferIndex;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);
    }

    CATLASS_DEVICE
    void VnchwND2NZ(AscendC::GlobalTensor<Element> const &dst, AscendC::GlobalTensor<Element> const &src,
        ComputeLayoutDst const &computeLayoutDst, ComputeLayoutSrc const &computeLayoutSrc)
    {
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        uint64_t dstLocalList[VNCHW_SIZE];
        uint64_t srcLocalList[VNCHW_SIZE];

        uint32_t vnchwMaxInnerAxisLen = COMPUTE_LENGTH / VNCHW_SIZE / ELE_NUM_PER_C0;
        // Only perform core separation in the outer axis direction.
        uint32_t innerAxisLen = computeLayoutSrc.shape(1);
        uint32_t transMatrixCols = innerAxisLen * ELE_NUM_PER_C0;
        uint32_t transMatrixRows = VNCHW_SIZE;
        bool cond1 = (computeLayoutSrc.shape(0) > 8192) && (innerAxisLen < vnchwMaxInnerAxisLen);
        bool cond2 = 
            (computeLayoutSrc.shape(0) > 4096) && (innerAxisLen < vnchwMaxInnerAxisLen * 2 && innerAxisLen % 2 == 0);
        bool cond3 = 
            (computeLayoutSrc.shape(0) > 2048) && (innerAxisLen < vnchwMaxInnerAxisLen * 4 && innerAxisLen % 4 == 0);
        if (cond2 && !cond1) {
            transMatrixCols /= 2;
        } else if (cond3 && !cond1) {
            transMatrixCols /= 4;
        } else if (!cond1) {
            if (innerAxisLen % 4 == 0) {
                transMatrixCols /= 4;
            } else if (innerAxisLen % 2 == 0) {
                transMatrixCols /= 2;
            }
        }
        uint32_t innerAxisPerCols = transMatrixCols / innerAxisLen;
        uint32_t tileRows = innerAxisPerCols * transMatrixRows;

        AscendC::TransDataTo5HDParams paramsRepeatOne{false, false, 1, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);

        uint32_t rowTiles = CeilDiv(computeLayoutSrc.shape(0), tileRows);
        for (uint32_t loopIdx = aivId; loopIdx < rowTiles; loopIdx += aivNum) {
            uint32_t rowTileIdx = loopIdx;
            uint32_t rowTileActual = (rowTileIdx == rowTiles - 1) ?
                (computeLayoutSrc.shape(0) - rowTileIdx * tileRows) : tileRows;
            MatrixCoord tileCoord {rowTileIdx * tileRows, 0};
            uint64_t tileOffsetSrc = computeLayoutSrc.GetOffset(tileCoord);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            ComputeLayoutSrc tileLayoutSrc = 
                computeLayoutSrc.GetTileLayout(MatrixCoord(1, rowTileActual * innerAxisLen));
            ComputeLayoutSrc ubLayoutIn = ComputeLayoutSrc{1, RoundUp(rowTileActual * innerAxisLen, ELE_NUM_PER_C0)};
            copyGm2Ub(inputBuffer[bufferIndex], src[tileOffsetSrc], ubLayoutIn, tileLayoutSrc);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);
            for (uint32_t i = 0; i < VNCHW_SIZE; ++i) {
                srcLocalList[i] = i * transMatrixCols * sizeof(Element) + inputBuffer[bufferIndex].GetPhyAddr();
                dstLocalList[i] = i * ELE_NUM_PER_C0 * sizeof(Element) + outputBuffer[bufferIndex].GetPhyAddr();
            }
            uint8_t repeatTimes = transMatrixCols / ELE_NUM_PER_C0;
            if (repeatTimes > 1) {
                AscendC::TransDataTo5HDParams params{false, false, repeatTimes, VNCHW_SIZE, 1};
                AscendC::TransDataTo5HD<Element>(dstLocalList, srcLocalList, params);
            } else {
                AscendC::TransDataTo5HD<Element>(dstLocalList, srcLocalList, paramsRepeatOne);
            }

            uint32_t vnchwPerInnerAxis = CeilDiv(innerAxisLen, ELE_NUM_PER_C0);
            for (uint32_t i = 0; i < vnchwPerInnerAxis; ++i) {
                if constexpr (std::is_same_v<Element, half>) {
                    for (uint32_t j = 0; j < VNCHW_SIZE; ++j) {
                        srcLocalList[j] = (j * ELE_NUM_PER_C0 + i * VNCHW_SIZE * ELE_NUM_PER_C0) * sizeof(Element)
                            + outputBuffer[bufferIndex].GetPhyAddr();
                        dstLocalList[j] = (i * tileRows * ELE_NUM_PER_C0 + j * innerAxisPerCols * ELE_NUM_PER_C0)
                            * sizeof(Element) + inputBuffer[bufferIndex].GetPhyAddr();
                    }
                } 
                if constexpr (std::is_same_v<Element, float>) {
                    for (uint32_t j = 0; j < VNCHW_SIZE; ++j) {
                        srcLocalList[j] = ((2 * (j % 8) + j / 8) * ELE_NUM_PER_C0 + i * VNCHW_SIZE * ELE_NUM_PER_C0) 
                            * sizeof(Element) + outputBuffer[bufferIndex].GetPhyAddr();
                        dstLocalList[j] = (i * tileRows * ELE_NUM_PER_C0 + (j / 2 + 8 * (j % 2)) * innerAxisPerCols
                            * ELE_NUM_PER_C0) * sizeof(Element) + inputBuffer[bufferIndex].GetPhyAddr();
                    }
                }
                uint8_t repeatTimes = innerAxisPerCols;
                if (repeatTimes > 1) {
                    AscendC::TransDataTo5HDParams params{false, false, repeatTimes, 1, 
                        static_cast<uint16_t>(innerAxisLen * VNCHW_SIZE / ELE_NUM_PER_C0)};
                    AscendC::TransDataTo5HD<Element>(dstLocalList, srcLocalList, params);
                } else {
                    AscendC::TransDataTo5HD<Element>(dstLocalList, srcLocalList, paramsRepeatOne);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventIds[bufferIndex]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);

            ComputeLayoutDst ubLayoutOut = ComputeLayoutDst::template MakeLayout<Element>(tileRows, innerAxisLen);
            ComputeLayoutDst tileLayoutUb = ubLayoutOut.GetTileLayout(MatrixCoord(rowTileActual, innerAxisLen));
            uint64_t tileOffsetDst = computeLayoutDst.GetOffset(tileCoord);
            ComputeLayoutDst tileLayoutDst = computeLayoutDst.GetTileLayout(MatrixCoord(rowTileActual, innerAxisLen));
            CopyUb2Gm(dst[tileOffsetDst], inputBuffer[bufferIndex], tileLayoutDst, tileLayoutUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIds[bufferIndex]);

            auto tmpBuffer = inputBuffer[bufferIndex];
            inputBuffer[bufferIndex] = outputBuffer[bufferIndex];
            outputBuffer[bufferIndex] = tmpBuffer;

            bufferIndex = 1 - bufferIndex;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventIds[1]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIds[1]);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst, AscendC::GlobalTensor<Element> const &src,
        LayoutOut const &layoutDst, LayoutIn const &layoutSrc)
    {
        auto computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        auto computeLayoutDst = GetPaddingComputeLayout(layoutDst);
        uint32_t vnchwMaxInnerAxisLen = COMPUTE_LENGTH / VNCHW_SIZE / ELE_NUM_PER_C0;
        bool cond0 = computeLayoutSrc.shape(1) % ELE_NUM_PER_C0 != 0;
        bool cond1 = computeLayoutSrc.shape(1) < vnchwMaxInnerAxisLen && computeLayoutSrc.shape(1);
        bool cond2 = computeLayoutSrc.shape(1) < vnchwMaxInnerAxisLen * 2 && computeLayoutSrc.shape(1) % 2 == 0;
        bool cond3 = computeLayoutSrc.shape(1) < vnchwMaxInnerAxisLen * 4 && computeLayoutSrc.shape(1) % 4 == 0;
        bool cond4 = computeLayoutSrc.shape(0) > 1024; // Empirical parameters
        // VnchwND2NZ requires that the shape must be equal the stride
        bool cond5 = computeLayoutSrc.shape(1) == computeLayoutSrc.stride(0);
        if (cond0 && cond4 && (cond1 || cond2 || cond3) && cond5) {
            VnchwND2NZ(dst, src, computeLayoutDst, computeLayoutSrc);
        } else {
            CommonND2NZ(dst, src, computeLayoutDst, computeLayoutSrc);
        }
    }

private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<Element> outputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) * 2 <= ArchTag::UB_SIZE, "Excedding the UB space!");
    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> ||
        std::is_same_v<LayoutIn, layout::ColumnMajor>, "Unsported layout for PaddingMatrixNZ!");
};

template<
    class ArchTag_,
    class Element_,
    class LayoutIn_,
    class LayoutOut_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixBlockND {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using LayoutIn = LayoutIn_;
    using LayoutOut = LayoutOut_;
    using ComputeLayout = Catlass::layout::RowMajor;
    using ComputeLayoutDst = Catlass::layout::PaddingRowMajor;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, ComputeLayout>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, ComputeLayout>>;

    static const PaddingTag paddingTag = PaddingTag::PADDING_BLOCK_ND;
    CATLASS_HOST_DEVICE static
    LayoutOut GetWorkspaceLayout(const LayoutIn& layout, uint32_t rowAlign, uint32_t colAlign)
    {
        return LayoutOut(layout.shape(0), layout.shape(1), rowAlign, colAlign);
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols, uint32_t rowAlign, uint32_t colAlign)
    {
        return static_cast<size_t>(RoundUp(rows, rowAlign)) * RoundUp(cols, colAlign) * sizeof(Element);
    }

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrixBlockND(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CATLASS_DEVICE
    ComputeLayoutDst GetPaddingComputeLayout(layout::PaddingRowMajor const &layout)
    {
        return ComputeLayoutDst(
            layout.shape(0) * layout.shape(1),
            layout.shape(2) * layout.shape(3),
            layout.shape(0),
            layout.shape(2)
        );
    }

    CATLASS_DEVICE
    ComputeLayoutDst GetPaddingComputeLayout(layout::PaddingColumnMajor const &layout)
    {
        return ComputeLayoutDst(
            layout.shape(2) * layout.shape(3),
            layout.shape(0) * layout.shape(1),
            layout.shape(2),
            layout.shape(0)
        );
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    LayoutOut const &layoutDst, LayoutIn const &layoutSrc)
    {
        auto computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        auto computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(computeLayoutSrc.shape(1));

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

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
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout ubLayout = ComputeLayout{1, actualDataNum};
                ComputeLayout dstLayout = ComputeLayout(CeilDiv(actualDataNum, computeLayoutDst.shape(2)),
                    computeLayoutDst.shape(2), computeLayoutDst.stride(3));

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout ubLayout = ComputeLayout{actualTilesNum, tileLen, roundTileLen};
                ComputeLayout dstLayout = ComputeLayout{CeilDiv(tileLen, computeLayoutDst.shape(2)),
                    computeLayoutDst.shape(2), computeLayoutDst.stride(3)};

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset + MatrixCoord(i, 0));
                    uint64_t ubOffset = ubLayout.GetOffset(MatrixCoord(i, 0));
                    copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex][ubOffset], dstLayout, ubLayout);
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
    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> ||
        std::is_same_v<LayoutIn, layout::ColumnMajor>, "Unsported layout for PaddingMatrixBlockNd!");
};

template<
    class ArchTag_,
    class Element_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixND {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor;

    constexpr static uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    static const PaddingTag paddingTag = PaddingTag::PADDING_ND;
    using LayoutIn = Layout_;
    using LayoutOut = Layout_;
    CATLASS_HOST_DEVICE static
    LayoutOut GetWorkspaceLayout(const LayoutIn& layout, uint32_t align)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align)};
        } else {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align)};
        }
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols, uint32_t align)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return static_cast<size_t>(rows) * RoundUp(cols, align) * sizeof(Element);
        } else {
            return static_cast<size_t>(cols) * RoundUp(rows, align) * sizeof(Element);
        }
    }

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrixND(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    Layout const &layoutDst, Layout const &layoutSrc)
    {
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t roundTileLen = RoundUp(computeLayoutSrc.shape(1), ELE_NUM_PER_C0);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

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
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout ubLayout = ComputeLayout{actualTilesNum, tileLen, roundTileLen};

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~PaddingMatrixND() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> || 
        std::is_same_v<LayoutIn, layout::ColumnMajor>, "Unsported layout for PaddingMatrixND!");
};

// The PaddingBuilder structure can construct the required padding class by specifying the PaddingTag 
// and the basic information of the matrix, thereby unifying the use of various paddings.
// Moreover, it allows for quick retrieval of the layout information after padding.
template <PaddingTag, class ArchTag, class Element, class LayoutIn, uint32_t COMPUTE_LENGTH = 0>
struct PaddingBuilder {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Padding is not implemented for this layout");
};

template <class ArchTag, class Element, class LayoutIn, uint32_t COMPUTE_LENGTH>
struct PaddingBuilder<PaddingTag::NO_PADDING, ArchTag, Element, LayoutIn, COMPUTE_LENGTH> {
    using LayoutAfterPadding = LayoutIn;
    using Padding = void;
};

template <class ArchTag, class Element, class LayoutIn, uint32_t COMPUTE_LENGTH>
struct PaddingBuilder<PaddingTag::PADDING_ND, ArchTag, Element, LayoutIn, COMPUTE_LENGTH> {
    using LayoutAfterPadding = LayoutIn;
    using Padding = std::conditional_t<
        (COMPUTE_LENGTH > 0),
        Catlass::Gemm::Kernel::PaddingMatrixND<ArchTag, Element, LayoutIn, COMPUTE_LENGTH>,
        Catlass::Gemm::Kernel::PaddingMatrixND<ArchTag, Element, LayoutIn, 96 * 1024 / sizeof(Element)>
    >;
};

template <class ArchTag, class Element, class LayoutIn, uint32_t COMPUTE_LENGTH>
struct PaddingBuilder<PaddingTag::PADDING_BLOCK_ND, ArchTag, Element, LayoutIn, COMPUTE_LENGTH> {
    using LayoutAfterPadding = std::conditional_t<std::is_same_v<LayoutIn, layout::RowMajor>,
        layout::PaddingRowMajor, layout::PaddingColumnMajor>;
    using Padding = std::conditional_t<
        (COMPUTE_LENGTH > 0),
        Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, Element, LayoutIn, LayoutAfterPadding, COMPUTE_LENGTH>,
        Catlass::Gemm::Kernel::PaddingMatrixBlockND<
            ArchTag, Element, LayoutIn, LayoutAfterPadding, 96 * 1024 / sizeof(Element)>
    >;
};

template <class ArchTag, class Element, class LayoutIn, uint32_t COMPUTE_LENGTH>
struct PaddingBuilder<PaddingTag::PADDING_NZ, ArchTag, Element, LayoutIn, COMPUTE_LENGTH> {
    using LayoutAfterPadding = std::conditional_t<std::is_same_v<LayoutIn, layout::RowMajor>,
        layout::zN, layout::nZ>;
    using Padding = std::conditional_t<
        (COMPUTE_LENGTH > 0),
        Catlass::Gemm::Kernel::PaddingMatrixNZ<ArchTag, Element, LayoutIn, LayoutAfterPadding, COMPUTE_LENGTH>,
        Catlass::Gemm::Kernel::PaddingMatrixNZ<
            ArchTag, Element, LayoutIn, LayoutAfterPadding, 48 * 1024 / sizeof(Element)>
    >;
};


template<
    PaddingTag paddingTag_,
    class ArchTag_,
    class ElementIn_,
    class ElementOut_,
    class Layout_
>
struct RemovePaddingNDAndCast {
public:
    using ArchTag = ArchTag_;
    using ElementIn = ElementIn_;
    using ElementOut = ElementOut_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<ElementIn, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<ElementOut, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor;

    constexpr static uint32_t ELE_NUM_PER_C0 = 
        (BYTE_PER_C0 / sizeof(ElementIn) > BYTE_PER_C0 / sizeof(ElementOut))
            ? BYTE_PER_C0 / sizeof(ElementIn)
            : BYTE_PER_C0 / sizeof(ElementOut);

    static const PaddingTag paddingTag = paddingTag_;
    using LayoutIn = Layout_;
    using LayoutOut = Layout_;
    CATLASS_HOST_DEVICE static
    LayoutOut GetWorkspaceLayout(const LayoutIn& layout, uint32_t align)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(1), align)};
        } else {
            return LayoutOut{layout.shape(0), layout.shape(1), RoundUp(layout.shape(0), align)};
        }
    }
    static size_t GetWorkspaceSize(uint32_t rows, uint32_t cols, uint32_t align = 1)
    {
        if constexpr (std::is_same_v<LayoutIn, layout::RowMajor>) {
            return static_cast<size_t>(rows) * RoundUp(cols, align) * sizeof(ElementIn);
        } else {
            return static_cast<size_t>(cols) * RoundUp(rows, align) * sizeof(ElementIn);
        }
    }

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    RemovePaddingNDAndCast(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        constexpr uint32_t inputBufSize = COMPUTE_LENGTH * sizeof(ElementIn);
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(bufferOffset + inputBufSize * i);
        }
        if constexpr (!std::is_same_v<ElementIn, ElementOut>) {
            constexpr uint32_t outputBufSize = COMPUTE_LENGTH * sizeof(ElementOut);
            bufferOffset = inputBufSize * BUFFER_NUM;
            for (uint32_t i = 0; i < BUFFER_NUM; i++) {
                outputBuffer[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(bufferOffset + outputBufSize * i);
            }
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementOut> const &dst,
                    AscendC::GlobalTensor<ElementIn> const &src,
                    Layout const &layoutDst, Layout const &layoutSrc, bool useSingleCore = false)
    {
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum, aivId;
        if (useSingleCore) {
            aivNum = AscendC::GetSubBlockNum();
            aivId = AscendC::GetSubBlockIdx();
        } else {
            aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
            aivId = AscendC::GetBlockIdx();
        }

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t roundTileLen = RoundUp(computeLayoutSrc.shape(1), ELE_NUM_PER_C0);

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

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
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                if constexpr (!std::is_same_v<ElementIn, ElementOut>) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer[bufferIndex], inputBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, actualDataNum);
                    } else {
                        AscendC::Cast(outputBuffer[bufferIndex], inputBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, actualDataNum);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);

                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                    copyUb2Gm(dst[gmDstOffset], outputBuffer[bufferIndex], dstLayout, ubLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                    copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                }


                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout ubLayout = ComputeLayout{actualTilesNum, tileLen, roundTileLen};

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                if constexpr (!std::is_same_v<ElementIn, ElementOut>) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIds[bufferIndex]);
                    uint32_t castDataNum = actualTilesNum * roundTileLen;
                    if constexpr (std::is_same_v<ElementOut, half>) {
                        AscendC::Cast(outputBuffer[bufferIndex], inputBuffer[bufferIndex], AscendC::RoundMode::CAST_NONE, castDataNum);
                    } else {
                        AscendC::Cast(outputBuffer[bufferIndex], inputBuffer[bufferIndex], AscendC::RoundMode::CAST_RINT, castDataNum);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIds[bufferIndex]);

                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                    copyUb2Gm(dst[gmDstOffset], outputBuffer[bufferIndex], dstLayout, ubLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                    copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                }

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~RemovePaddingNDAndCast() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<ElementIn> inputBuffer[BUFFER_NUM];
    AscendC::LocalTensor<ElementOut> outputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static constexpr uint32_t COMPUTE_LENGTH = 
        std::is_same_v<ElementIn, ElementOut> ? 
        ArchTag::UB_SIZE / sizeof(ElementIn) / BUFFER_NUM :
        ArchTag::UB_SIZE / (sizeof(ElementIn) + sizeof(ElementOut)) / BUFFER_NUM;
    static_assert(
        std::is_same_v<ElementIn, ElementOut> ? 
        BUFFER_NUM * COMPUTE_LENGTH * sizeof(ElementIn) <= ArchTag::UB_SIZE :
        BUFFER_NUM * COMPUTE_LENGTH * (sizeof(ElementIn) + sizeof(ElementOut)) <= ArchTag::UB_SIZE,
        "Excedding the UB space!");

    static_assert(std::is_same_v<LayoutIn, layout::RowMajor> || 
        std::is_same_v<LayoutIn, layout::ColumnMajor>, "Unsported layout for RemovePaddingNDAndCast!");
};


template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class PaddingMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingA = PaddingMatrixND<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrixND<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

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
        LayoutA layoutWA;
        GM_ADDR ptrWB;
        LayoutB layoutWB;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
            GM_ADDR ptrWA_, LayoutA layoutWA_, GM_ADDR ptrWB_, LayoutB layoutWB_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t align;
        size_t elementSize;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
    {
        // prevent division of 0
        if (align == 0) {
            return 0;
        }
        return layout::RowMajor(layout.shape(0), layout.shape(1),
            (layout.shape(1) + align - 1) / align * align);
    }

    static layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
    {
        return layout::ColumnMajor(layout.shape(0), layout.shape(1),
            (layout.shape(0) + align - 1) / align * align);
    }

    static size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }

    static size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        GemmCoord problemShape = args.problemShape;
        LayoutA layoutA{problemShape.m(), problemShape.k()};
        LayoutB layoutB{problemShape.k(), problemShape.n()};
        size_t sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        size_t sizeWB = GetWorkspaceLen(GetWorkspaceLayout(layoutB, args.align)) * args.elementSize;
        return sizeWA + sizeWB;
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
        size_t sizeWA = GetWorkspaceLen(GetWorkspaceLayout(layoutA, args.align)) * args.elementSize;
        uint8_t *workspaceWB = workspace + sizeWA;
        Params params{problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            workspace,
            GetWorkspaceLayout(layoutA, args.align),
            workspaceWB,
            GetWorkspaceLayout(layoutB, args.align)};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    PaddingMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
        PaddingA paddingA(resource);
        paddingA(gmWA, gmA, params.layoutWA, params.layoutA);

        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementB> gmWB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
        PaddingB paddingB(resource);
        paddingB(gmWB, gmB, params.layoutWB, params.layoutB);
        // 0x0 synchronization control between AI Core
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Executes matmul
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);

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

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutWA,
                gmB[gmOffsetB], params.layoutWB,
                gmC[gmOffsetC], params.layoutC,
                actualBlockShape);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_PADDING_MATMUL_HPP