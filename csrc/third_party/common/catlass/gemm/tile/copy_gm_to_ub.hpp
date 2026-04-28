/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<Arch::AtlasA2,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::isRowMajor<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(tla::detail::isRowMajor<typename TensorSrc::Layout>::value &&
                      tla::detail::isRowMajor<typename TensorDst::Layout>::value &&
                      TensorSrc::position == AscendC::TPosition::GM &&
                      TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, "
            "while TensorDst must be UB and RowMajor");

        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()),
            tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopyPad(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], dataCopyParams, padParams);
    };
};

template <
    class ArchTag,
    class GmType
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
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

}  // Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP
