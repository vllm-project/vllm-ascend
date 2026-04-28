/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_GELU_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ELEMWISE_GELU_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {
template <
    // / Tag indicating architecture
    class ArchTag_,
    // / Compute data type
    class ComputeType_,
    // / Length of the compute buffer
    uint32_t COMPUTE_LENGTH_>
struct TileElemWiseGelu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
    const float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
    const float TANH_APPROX_FACTOR = 1 / 0.044715;

    CATLASS_DEVICE
    TileElemWiseGelu() {}

    CATLASS_DEVICE
    void operator () (AscendC::LocalTensor<ElementCompute> const & dstLocal,
        AscendC::LocalTensor<ElementCompute> const & srcLocal)
    {
        using namespace AscendC;
        
        // current realization: x / (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3)))
        Mul(dstLocal, srcLocal, srcLocal, COMPUTE_LENGTH); // d: x^2 , s:x
        Mul(dstLocal, dstLocal, srcLocal, COMPUTE_LENGTH); // d: x^3 ,.s:x
        Axpy(dstLocal, srcLocal, TANH_APPROX_FACTOR, COMPUTE_LENGTH); // d: x / 0.044715 + x^3 , s: x
        // d: -1.5957691*0.044715(x/0.044715 + x^3), s: x
        Muls(dstLocal, dstLocal, NEG_SQRT_EIGHT_OVER_PI, COMPUTE_LENGTH);
        Exp(dstLocal, dstLocal, COMPUTE_LENGTH); // d: e^(-1.5957691*0.044715(x/0.044715 + x^3))
        // d: (1 + e^(-1.5957691*0.044715(x/0.044715 + x^3))
        Adds(dstLocal, dstLocal, (ElementCompute)1, COMPUTE_LENGTH);
        Div(dstLocal, srcLocal, dstLocal, COMPUTE_LENGTH);
    }
};
} // namespace Catlass::Epilogue::Tile

#endif
