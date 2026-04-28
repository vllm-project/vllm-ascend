/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SILU_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_SILU_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ComputeType_,
    // / COMPUTE_LENGTH of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileElemWiseSilu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    TileElemWiseSilu() {}

    CATLASS_DEVICE
    void operator () (AscendC::LocalTensor<ElementCompute> const & dstLocal,
        AscendC::LocalTensor<ElementCompute> const & srcLocal)
    {
        using namespace AscendC;
        // d: -x, s: x
        Muls(dstLocal, srcLocal, (ElementCompute)-1, COMPUTE_LENGTH);
        // d: exp(-x), s: x
        Exp(dstLocal, dstLocal, COMPUTE_LENGTH);
        // d: 1 + exp(-x), s: x
        Adds(dstLocal, dstLocal, (ElementCompute)1, COMPUTE_LENGTH);
        // d: x / 1 + exp(-x), s: x
        Div(dstLocal, srcLocal, dstLocal, COMPUTE_LENGTH);
    }
};
} // namespace Catlass::Epilogue::Tile

#endif
