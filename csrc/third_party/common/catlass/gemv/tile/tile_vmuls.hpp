/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_VMULS_HPP
#define CATLASS_GEMV_TILE_VMULS_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemv::Tile {

template <
    class ArchTag,
    class VType_
>
struct TileVmuls
{
    using Element = typename VType_::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    TileVmuls() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        Element scalar,
        uint32_t len)
    {
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(len);
        AscendC::Muls<Element,false>(
            dstTensor,
            srcTensor,
            scalar,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
    }
};
} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_VMULS_HPP
