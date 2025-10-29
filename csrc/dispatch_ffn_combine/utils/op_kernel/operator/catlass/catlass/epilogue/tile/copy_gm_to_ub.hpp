/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Epilogue::Tile {

template <
    class ArchTag,
    class GmType
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0),
            layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            1,
            layoutSrc.shape(0) * sizeof(Element),
            0,
            0,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

/// @brief This copy instruction used to copy per token scale from GM to UB.
/// Copy the scale of shape (m,1) on GM to the first column of shape (m,n) on UB,
/// and pad the first block of each row (i.e. pad to shape (m,8) when element type is float).
/// @tparam ArchTag: Architecture tag.
/// @tparam GmType: Type of data on GM.
template <
    class ArchTag,
    class GmType
>
struct CopyPerTokenScale2Ub {
    static_assert(std::is_same_v<typename GmType::Layout, layout::ColumnMajor>,
        "Unsupported layout for CopyPerTokenScale2Ub.");

    using Element = typename GmType::Element;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyPerTokenScale2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<Element> padParams;

        dataCopyParams.blockCount = layoutSrc.shape(0);
        dataCopyParams.blockLen = layoutSrc.shape(1) * sizeof(Element);  // per token scale has only one column
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;
        // Pad the data to the complete block
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    }
};

template <
    class ArchTag,
    class GmType
>
struct CopyGm2UbAligned {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub aligned, can not find the specialization.");
};

template <typename Element>
struct CopyGm2UbAligned<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    CATLASS_DEVICE
    CopyGm2UbAligned() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    };
};

}  // Catlass::Epilogue::Tile

#endif
