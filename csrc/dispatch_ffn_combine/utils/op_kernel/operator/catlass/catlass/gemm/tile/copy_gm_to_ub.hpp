/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<Arch::AtlasA2, tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::VECCALC>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value &&
                     tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, AscendC::TPosition::VECCALC>;
    using TensorSrc = tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::GM>;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(ElementSrc);

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            tla::get<0>(srcTensor.shape()),
            tla::get<1>(srcTensor.shape()) * sizeof(ElementSrc),
            (tla::get<0>(srcTensor.stride()) - tla::get<1>(srcTensor.shape())) * sizeof(ElementSrc),
            (tla::get<0>(dstTensor.stride()) - tla::get<1>(dstTensor.shape())) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams, padParams);
    };
};

}  // Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_UB_HPP
