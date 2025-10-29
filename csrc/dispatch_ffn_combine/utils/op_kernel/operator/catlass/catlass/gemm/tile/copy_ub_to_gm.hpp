/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP
#define CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(dstTensor.shape()),
            tla::get<1>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) * sizeof(ElementSrc),
            0
        );
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

/// Partial specialization for AtlasA2, RowMajor in and PaddingRowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Arch::AtlasA2, tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
    tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>,
    layout::RowMajor, layout::PaddingRowMajor> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = tla::Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTlaExt() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<1, 1>(dstTensor.shape()),
            tla::get<1, 0>(dstTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (tla::get<1, 1>(dstTensor.stride()) - tla::get<1, 0>(dstTensor.shape())) * sizeof(ElementSrc),
            0
        );
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

}  // Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_UB_TO_GM_HPP