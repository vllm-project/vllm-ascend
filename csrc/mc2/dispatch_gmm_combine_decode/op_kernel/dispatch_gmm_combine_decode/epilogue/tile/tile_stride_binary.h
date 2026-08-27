/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ElementCompute_, class TileShape_, int64_t DST_STRIDE_, int64_t SRC0_STRIDE_,
          int64_t SRC1_STRIDE_>
struct TileStrideBinary {
    using ArchTag = ArchTag_;
    using ElementCompute = ElementCompute_;
    using TileShape = TileShape_;
    static constexpr int64_t DST_STRIDE = DST_STRIDE_;
    static constexpr int64_t SRC0_STRIDE = SRC0_STRIDE_;
    static constexpr int64_t SRC1_STRIDE = SRC1_STRIDE_;

    static constexpr uint32_t MAX_REPEAT_TIMES = 255;
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementCompute);

    static constexpr uint32_t DST_BLK_NUM_PER_COLUMN = DST_STRIDE / ELE_NUM_PER_BLK;
    static constexpr uint32_t SRC0_BLK_NUM_PER_COLUMN = SRC0_STRIDE / ELE_NUM_PER_BLK;
    static constexpr uint32_t SRC1_BLK_NUM_PER_COLUMN = SRC1_STRIDE / ELE_NUM_PER_BLK;

    static constexpr uint32_t ROW_NUM_PER_COMPUTE = MAX_REPEAT_TIMES;
    static constexpr uint32_t COL_NUM_PER_COMPUTE = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);

    CATLASS_DEVICE
    TileStrideBinary()
    {
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = DST_BLK_NUM_PER_COLUMN;
        repeatParams.src0RepStride = SRC0_BLK_NUM_PER_COLUMN;
        repeatParams.src1RepStride = SRC1_BLK_NUM_PER_COLUMN;
    }

    AscendC::BinaryRepeatParams repeatParams;
};

template <class ArchTag_, class ElementCompute_, class TileShape_, int64_t DST_STRIDE_, int64_t SRC0_STRIDE_,
          int64_t SRC1_STRIDE_>
struct TileStrideMul
    : TileStrideBinary<ArchTag_, ElementCompute_, TileShape_, DST_STRIDE_, SRC0_STRIDE_, SRC1_STRIDE_> {
    using Base = TileStrideBinary<ArchTag_, ElementCompute_, TileShape_, DST_STRIDE_, SRC0_STRIDE_, SRC1_STRIDE_>;

    CATLASS_DEVICE
    TileStrideMul() : Base() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<typename Base::ElementCompute> const &ubDst,
                    AscendC::LocalTensor<typename Base::ElementCompute> const &ubSrc0,
                    AscendC::LocalTensor<typename Base::ElementCompute> const &ubSrc1)
    {
        for (uint32_t rowOffset = 0; rowOffset < Base::TileShape::ROW; rowOffset += Base::ROW_NUM_PER_COMPUTE) {
            uint32_t residueM = Base::TileShape::ROW - rowOffset;
            uint8_t repeatTimes =
                static_cast<uint8_t>((residueM > Base::ROW_NUM_PER_COMPUTE) ? Base::ROW_NUM_PER_COMPUTE : residueM);
            for (uint32_t colOffset = 0; colOffset < Base::TileShape::COLUMN; colOffset += Base::COL_NUM_PER_COMPUTE) {
                uint32_t residueN = Base::TileShape::COLUMN - colOffset;
                uint64_t mask = (residueN > Base::COL_NUM_PER_COMPUTE) ? Base::COL_NUM_PER_COMPUTE : residueN;
                AscendC::Mul(ubDst[rowOffset * Base::DST_STRIDE + colOffset],
                             ubSrc0[rowOffset * Base::SRC0_STRIDE + colOffset],
                             ubSrc1[rowOffset * Base::SRC1_STRIDE + colOffset], mask, repeatTimes, this->repeatParams);
            }
        }
    }
};

template <class ArchTag_, class ElementCompute_, class TileShape_, int64_t DST_STRIDE_, int64_t SRC0_STRIDE_,
          int64_t SRC1_STRIDE_>
struct TileStrideDiv
    : TileStrideBinary<ArchTag_, ElementCompute_, TileShape_, DST_STRIDE_, SRC0_STRIDE_, SRC1_STRIDE_> {
    using Base = TileStrideBinary<ArchTag_, ElementCompute_, TileShape_, DST_STRIDE_, SRC0_STRIDE_, SRC1_STRIDE_>;

    CATLASS_DEVICE
    TileStrideDiv() : Base() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<typename Base::ElementCompute> const &ubDst,
                    AscendC::LocalTensor<typename Base::ElementCompute> const &ubSrc0,
                    AscendC::LocalTensor<typename Base::ElementCompute> const &ubSrc1)
    {
        for (uint32_t rowOffset = 0; rowOffset < Base::TileShape::ROW; rowOffset += Base::ROW_NUM_PER_COMPUTE) {
            uint32_t residueM = Base::TileShape::ROW - rowOffset;
            uint8_t repeatTimes =
                static_cast<uint8_t>((residueM > Base::ROW_NUM_PER_COMPUTE) ? Base::ROW_NUM_PER_COMPUTE : residueM);
            for (uint32_t colOffset = 0; colOffset < Base::TileShape::COLUMN; colOffset += Base::COL_NUM_PER_COMPUTE) {
                uint32_t residueN = Base::TileShape::COLUMN - colOffset;
                uint64_t mask = (residueN > Base::COL_NUM_PER_COMPUTE) ? Base::COL_NUM_PER_COMPUTE : residueN;
                AscendC::Div(ubDst[rowOffset * Base::DST_STRIDE + colOffset],
                             ubSrc0[rowOffset * Base::SRC0_STRIDE + colOffset],
                             ubSrc1[rowOffset * Base::SRC1_STRIDE + colOffset], mask, repeatTimes, this->repeatParams);
            }
        }
    }
};

}  // namespace Catlass::Epilogue::Tile
