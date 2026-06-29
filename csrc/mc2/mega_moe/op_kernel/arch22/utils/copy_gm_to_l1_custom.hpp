/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_l1_custom.hpp
 * \brief
 */

#ifndef COPY_GM_TO_L1_CUSTOM_HPP
#define COPY_GM_TO_L1_CUSTOM_HPP

namespace Catlass::Gemm::Tile {
/// Partial specialization for nZ in and nZ out.
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); // int64, 32/8=4

    template <class T>
    __aicore__ static constexpr T CeilDiv(const T dividend, const T divisor)
    {
        return (dividend + divisor - 1) / divisor;
    }

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::GlobalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = 1;
        uint32_t blockLen = CeilDiv(layoutSrc.shape(0), ELE_NUM_PER_C0);

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
    }
};
} // namespace Catlass::Gemm::Tile
#endif // COPY_GM_TO_L1_CUSTOM_HPP