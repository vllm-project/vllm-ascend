/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_COPY_L1_TO_L0B_HPP
#define CATLASS_CONV_TILE_COPY_L1_TO_L0B_HPP

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
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::CI1KHKWCOCI0, AscendC::TPosition::A1>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::CI1KHKWCOCI0;

    static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(
                dstTensor[i * layoutDst.stride(1)],
                srcTensor[i * layoutSrc.shape(3) * ELE_NUM_PER_C0],
                loadDataParams
            );
        }
    }
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_COPY_L1_TO_L0B_HPP
