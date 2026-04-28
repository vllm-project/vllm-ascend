/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_H
#define MATMUL_BLOCK_BLOCK_MMAD_H

#include <type_traits>
#include "../utils/arch.h"
#include "../utils/integral_constant.h"
#include "../utils/matmul_layout_type.h"
#include "./block_mmad_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
* @class BlockMmad
* @brief Block matrix multiplication class for performing block matrix multiplication operations
*/
template <
    /// The dispatch policy type, which determines the computational pipeline
    class DispatchPolicy,
    /// The shape of L1 tile
    class L1TileShape,
    /// The shape of L0 tile
    class L0TileShape,
    /// Type of matrix A
    class AType,
    /// Type of matrix B
    class BType,
    /// Type of matrix C
    class CType,
    /// Type of the bias term, defaulting to the same type of CType
    class BiasType = CType,
    /// The tile copy strategy type
    class TileCopy = void,
    /// Support specialization via the DispatchPolicy type
    typename = void
>
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/**
* @class BlockMmadBase
* @brief Base class of Block matrix multiplication class, serving as the base class for the CRTP pattern
*/
template <
    /// AsDerived class in CRTP
    class Derived,
    /// The dispatch policy type
    class DispatchPolicy_,
    /// The shape of L1 tile
    class L1TileShape,
    /// The shape of L0 tile
    class L0TileShape,
    /// Type of matrix A
    class AType_,
    /// Type of matrix B
    class BType_,
    /// Type of matrix C
    class CType_,
    /// Type of the bias term
    class BiasType_,
    /// The tile copy strategy type
    class TileCopy_
>
class BlockMmadBase {
public:
    using DispatchPolicy = DispatchPolicy_;
    using L1Shape = L1TileShape;
    using L0Shape = L0TileShape;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy = TileCopy_;

protected:
    /**
    * @brief Obtain a reference to the derived class
    */
    __aicore__ inline Derived& AsDerived()
    {
        return static_cast<Derived&>(*this);
    }

public:
    // Common static assertion
    static_assert(IsTileShapeValid<AType, BType, L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
