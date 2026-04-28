/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_MULS_HPP
#define CATLASS_GEMM_TILE_TILE_MULS_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/helper.hpp"
#include "tla/tensor.hpp"
namespace Catlass::Gemm::Tile {

///////////////////////////////////////////////////////
template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_>
struct TileMuls {
    using ArchTag = ArchTag_;
    using Element = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileMuls()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        Element scalar,
        uint32_t len
    )
    {
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<Element, AscendC::MaskMode::COUNTER>(len);
        AscendC::Muls<Element, false>(
            dstTensor, srcTensor, scalar, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};
///////////////////////////////////////////////////////
} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_MULS_HPP