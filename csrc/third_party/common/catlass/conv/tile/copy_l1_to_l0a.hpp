/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_COPY_L1_TO_L0A_HPP
#define CATLASS_CONV_TILE_COPY_L1_TO_L0A_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Conv::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template<class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<Element, layout::NC1HWC0, AscendC::TPosition::A1>>{
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::NC1HWC0;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    Conv2dFilterParams params;

    CATLASS_DEVICE
    CopyL1ToL0A(const Conv2dFilterParams& params_) : params(params_) {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor, // (ho, wo, cin1, Kh, Kw, C0)
        AscendC::LocalTensor<Element> srcTensor, // (cin1, hi, wi, C0)
        LayoutDst const &layoutDst, // {mPartRound, kPartActual} rowsInFractal, rowsByFractal, colsInFractal, colsByFractal
        LayoutSrc const &layoutSrc, // {1, FmapL1TileShape::Cin1, hiActual, wiActual, ELE_NUM_A_PER_C0}
        uint8_t* blockPadList)
    {
        uint32_t hiActual = layoutSrc.shape(2);
        uint32_t wiActual = layoutSrc.shape(3);
        uint32_t mPartRound = layoutDst.orgShape(0);
        uint32_t kPartActual = layoutDst.orgShape(1);
        uint32_t cin1L0Actual = kPartActual / (params.kh() * params.kw() * ELE_NUM_PER_C0);

        AscendC::LoadData(
            dstTensor,
            srcTensor,
            { // load3dv2
                blockPadList, // {padLeft, padRight, padTop, padBottom}
                static_cast<uint16_t>(hiActual), // 源操作数 height
                static_cast<uint16_t>(wiActual), // 源操作数 width
                static_cast<uint16_t>(cin1L0Actual * ELE_NUM_PER_C0), // 源操作数的通道数(channelSize为 4, 8, N*16, N*16+4, N*16+8)
                static_cast<uint16_t>(kPartActual), // 目的操作数Width维度的传输长度(16的倍数)
                static_cast<uint16_t>(mPartRound), // 目的操作数height维度的传输长度(16的倍数)
                0, // 目的操作数Width维度的起点
                0, // 目的操作数height维度的起点
                params.strideW(), params.strideH(),
                params.kw(), params.kh(),
                params.dilationW(), params.dilationH(),
                false, // 是否启用转置
                false, // 是否使能small k特性
                (half)(0) // Pad填充值的数值
            }
        );
    }
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_COPY_L1_TO_L0A_HPP
