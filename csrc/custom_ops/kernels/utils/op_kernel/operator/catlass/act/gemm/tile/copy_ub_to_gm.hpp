/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef ACT_GEMM_TILE_COPY_UB_TO_GM_HPP
#define ACT_GEMM_TILE_COPY_UB_TO_GM_HPP

#include "../../../act/act.hpp"
#include "../../../tla/tensor.hpp"

namespace Act::Gemm::Tile {

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTla<
    Arch::AtlasA2, Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
    Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc_>::value && tla::detail::isRowMajor<LayoutDst_>::value>> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    ACT_DEVICE
    TileCopyTla() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            get<0>(dstTensor.shape()), get<1>(dstTensor.shape()) * sizeof(ElementSrc),
            (get<0>(srcTensor.stride()) - get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (get<0>(dstTensor.stride()) - get<1>(dstTensor.shape())) * sizeof(ElementSrc), 0);
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

/// Partial specialization for AtlasA2, RowMajor in and PaddingRowMajor out.
template <class ElementSrc, class ElementDst, class LayoutSrc_, class LayoutDst_>
struct TileCopyTlaExt<Arch::AtlasA2, Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc_, AscendC::TPosition::VECCALC>,
                      Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst_, AscendC::TPosition::GM>, layout::RowMajor,
                      layout::PaddingRowMajor> {
    using LayoutDst = LayoutDst_;
    using LayoutSrc = LayoutSrc_;
    using TensorDst = Tensor<AscendC::GlobalTensor<ElementDst>, LayoutDst, AscendC::TPosition::GM>;
    using TensorSrc = Tensor<AscendC::LocalTensor<ElementSrc>, LayoutSrc, AscendC::TPosition::VECCALC>;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    ACT_DEVICE
    TileCopyTlaExt() {};

    ACT_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            get<1, 1>(dstTensor.shape()), get<1, 0>(dstTensor.shape()) * sizeof(ElementSrc),
            (get<0>(srcTensor.stride()) - get<1>(srcTensor.shape())) / ELE_NUM_PER_C0,
            (get<1, 1>(dstTensor.stride()) - get<1, 0>(dstTensor.shape())) * sizeof(ElementSrc), 0);
        AscendC::DataCopyPad(dstTensor.data(), srcTensor.data(), dataCopyParams);
    };
};

}  // namespace Act::Gemm::Tile

#endif  // ACT_GEMM_TILE_COPY_UB_TO_GM_HPP
