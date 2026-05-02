/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_in_with_params.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_IN_COPY_IN_WITH_PARAMS_H
#define MATMUL_TILE_COPY_IN_COPY_IN_WITH_PARAMS_H

#include "../tile_copy_policy.h"
#include "../../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Define a Copy structure for data copying operations from global memory to local memory
 * @param [in] ArchTag: architecture tag
 * @param [in] InputType: input type
 * @param [in] COPY_CFG: the copy configuration
 * @note This structure is only valid when the input data is located in global memory and has the ND format
 */
template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<
    ArchTag, CopyWithParams, void, void, InputType,
    AscendC::Std::enable_if_t<PosIsGM<InputType::pos>() && InputType::format == CubeFormat::ND>, // GM ND
    COPY_CFG
> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    /**
     * @brief Overloaded operator() function to perform the copy operation
     * @param [in] dst: parameters for the problem
     * @param [in] srcND: the source global tensor
     * @param [in] curRow: the current row index
     * @param [in] curCol: the current column index
     * @param [in] tileHeight: the height of the tile
     * @param [in] tileWidth: the width of the tile
     * @param [in] baseHeight: the base height
     * @param [in] baseWidth: the base width
     * @param [in] orgHeight: the original height
     * @param [in] orgWidth: the original width
     * @param [in] iskRowDirec: whether to copy in the row direction
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dst, AscendC::GlobalTensor<SrcT> srcND,
                                      int curRow, int curCol, int tileHeight, int tileWidth,
                                      int baseHeight, int baseWidth, int orgHeight, int orgWidth, bool iskRowDirec)
    {
        if constexpr (sizeof(TransT) == sizeof(int8_t)) {
            CopyND2NZ(dst, srcND, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth, 1,
                      0, 0, iskRowDirec);
        } else {
            CopyND2NZ(dst, srcND, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth);
        }
    }

private:
    /**
     * @brief Copy ND2NZ data from global memory to local memory
     * @param [in] dst: destination local tensor
     * @param [in] src: source global tensor
     * @param [in] row: matrix row count
     * @param [in] col: matrix column count
     * @param [in] height: matrix height
     * @param [in] width: matrix width
     * @param [in] gCol: global matrix column count
     * @param [in] ndNum: number of NDs, default is 1
     * @param [in] srcNdMatrixStride: source ND matrix stride, default is 0
     * @param [in] dstNzMatrixStride: destination NZ matrix stride, default is 0
     * @param [in] kAlignToC0Size: whether to align to C0 size, default is false
     */
    __aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<TransT>& dst, const AscendC::GlobalTensor<SrcT>& src,
                                     const int32_t row, const int32_t col, const int32_t height, const int32_t width,
                                     const int32_t gCol, const int32_t ndNum = 1, const int32_t srcNdMatrixStride = 0,
                                     const int32_t dstNzMatrixStride = 0, const bool kAlignToC0Size = false)
    {
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        int32_t dstNzC0Stride = 0;
        int64_t srcOffset;
        if constexpr (AscendC::IsSameTypeV<TransT, AscendC::int4b_t>) {
            srcOffset = ((int64_t)row * (int64_t)gCol * AscendC::INT4_TWO + (int64_t)col);
        } else {
            srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
        }
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;

        if (dstNzC0Stride) {
            nd2nzParams.dstNzC0Stride = dstNzC0Stride;
        } else {
            // when k is row(height) axis, int8 type gm->l1 nd2nz should be aligned to 32(c0Size)
            // while float/half type should be aligned to 16
            if (kAlignToC0Size) {
                nd2nzParams.dstNzC0Stride = AscendC::Ceil(height, c0Size) * c0Size;
            } else {
                nd2nzParams.dstNzC0Stride = AscendC::Ceil(height, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
            }
        }
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        if constexpr (!AscendC::ToMatmulConfig(COPY_CFG).intrinsicsCheck) {
            AscendC::DataCopy(dst, src[srcOffset], nd2nzParams);
        } else {
            if (gCol >= UINT16_MAX) {
                nd2nzParams.nValue = 1;
                nd2nzParams.srcDValue = width;
                for (int32_t i = 0; i < height; ++i) {
                    AscendC::DataCopy(dst[i * c0Size], src[srcOffset + gCol * i], nd2nzParams);
                }
            } else {
                AscendC::DataCopy(dst, src[srcOffset], nd2nzParams);
            }
        }
    }
};

/**
 * @struct Copy
 * @brief Template struct for copying data from GM to local tensor in NZ format
 * @param [in] ArchTag: architecture tag
 * @param [in] InputType: input type
 * @param [in] COPY_CFG: the copy configuration
 */
template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<
    ArchTag, CopyWithParams, void, void, InputType,
    AscendC::Std::enable_if_t<PosIsGM<InputType::pos>() && IsNz<InputType>()>, // GM NZ
    COPY_CFG
> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    /**
     * @brief Copy operator for transferring data from global tensor to local tensor
     * @param [in] dst: parameters for the problem
     * @param [in] srcNz: the source global tensor
     * @param [in] curRow: the current row index
     * @param [in] curCol: the current column index
     * @param [in] tileHeight: the height of the tile
     * @param [in] tileWidth: the width of the tile
     * @param [in] baseHeight: the base height
     * @param [in] baseWidth: the base width
     * @param [in] orgHeight: the original height
     * @param [in] orgWidth: the original width
     * @param [in] iskRowDirec: whether to copy in the row direction
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dst, AscendC::GlobalTensor<SrcT> srcNz,
                                      int curRow, int curCol, int tileHeight, int tileWidth,
                                      int baseHeight, int baseWidth, int orgHeight, int orgWidth, bool iskRowDirec)
    {
        CopyNZ2NZ(dst, srcNz, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgHeight, iskRowDirec);
    }

private:
    /**
     * @brief Copy NZ2NZ data from global memory to local memory
     * @param [in] dst: destination local tensor
     * @param [in] src: source global tensor
     * @param [in] row: matrix row count
     * @param [in] col: matrix column count
     * @param [in] height: matrix height
     * @param [in] width: matrix width
     * @param [in] gCol: global matrix column count
     * @param [in] ndNum: number of NDs, default is 1
     * @param [in] srcNdMatrixStride: source ND matrix stride, default is 0
     * @param [in] dstNzMatrixStride: destination NZ matrix stride, default is 0
     * @param [in] kAlignToC0Size: whether to align to C0 size, default is false
     */
    __aicore__ inline void CopyNZ2NZ(const AscendC::LocalTensor<TransT>& dst, const AscendC::GlobalTensor<SrcT>& src,
                                     const int32_t row, const int32_t col, const int32_t height, const int32_t width,
                                     const int32_t gRow, const bool kAlignToC0Size = false)
    {
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        int32_t alignedGRow = AscendC::Ceil(gRow, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        int64_t srcOffset = (int64_t)row * (int64_t)c0Size + (int64_t)col * (int64_t)alignedGRow;
        // height direction need to be 16 aligned
        auto alignHeight = AscendC::Ceil(height, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        int32_t blockLen = alignHeight * c0Size * sizeof(TransT) / AscendC::ONE_BLK_SIZE;
        int32_t srcStride = (alignedGRow - alignHeight) * (c0Size * sizeof(TransT) / AscendC::ONE_BLK_SIZE);

        if (srcStride >= UINT16_MAX) {
            for (int32_t i = 0; i < AscendC::Ceil(width, c0Size); ++i) {
                AscendC::DataCopy(dst[i * alignHeight * c0Size], src[srcOffset + i * gRow * c0Size],
                                  {1, static_cast<uint16_t>(blockLen), 0, 0});
            }
        } else {
            uint16_t nburst = AscendC::Ceil(width, c0Size);
            int32_t dstStride = 0;
            if constexpr (AscendC::IsSameTypeV<TransT, int8_t>) {
                if (kAlignToC0Size) {
                    auto alignHeightC0Size = AscendC::Ceil(height, c0Size) * c0Size;
                    dstStride = alignHeightC0Size - alignHeight;
                }
            }
            AscendC::DataCopy(dst, src[srcOffset],
                              {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride),
                               static_cast<uint16_t>(dstStride)});
        }
    }
};

/**
 * @struct Copy
 * @brief Template struct for copying data from GM to local tensor in VECTOR format
 * @param [in] ArchTag: architecture tag
 * @param [in] InputType: input type
 * @param [in] COPY_CFG: the copy configuration
 */
template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<
    ArchTag, CopyWithParams, void, void, InputType,
    AscendC::Std::enable_if_t<PosIsGM<InputType::pos>() && InputType::format == CubeFormat::VECTOR>, // GM VECTOR
    COPY_CFG
> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    /**
     * @brief Copy operator for transferring data from global tensor to local tensor
     * @param [in] dst: parameters for the problem
     * @param [in] srcVec: the source global tensor
     * @param [in] curRow: the current row index
     * @param [in] curCol: the current column index
     * @param [in] tileHeight: the height of the tile
     * @param [in] tileWidth: the width of the tile
     * @param [in] baseHeight: the base height
     * @param [in] baseWidth: the base width
     * @param [in] orgHeight: the original height
     * @param [in] orgWidth: the original width
     * @param [in] iskRowDirec: whether to copy in the row direction
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dst, AscendC::GlobalTensor<SrcT> srcVec,
                                      int curRow, int curCol, int tileHeight, int tileWidth,
                                      int baseHeight, int baseWidth, int orgHeight, int orgWidth, bool iskRowDirec)
    {
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        CopyVector2A1(dst, srcVec, curCol * baseWidth, AscendC::Ceil(tileWidth, c0Size));
    }

private:
    /**
     * @brief Copy data from VECTOR format global tensor to A1 format local tensor
     * @param [in] dst: destination local tensor
     * @param [in] src: source global tensor
     * @param [in] col: matrix column count
     * @param [in] blockLen: block length
     */
    __aicore__ inline void CopyVector2A1(const AscendC::LocalTensor<TransT>& dst,
                                         const AscendC::GlobalTensor<SrcT>& src, const int32_t col,
                                         const int32_t blockLen)
    {
        AscendC::DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = blockLen;
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(dst, src[col], dataCopyInfo, enhancedParams);
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
