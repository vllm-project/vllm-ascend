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
 * \file block_mmad_multi_block_bias.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_BIAS_H
#define MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_BIAS_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "../policy/dispatch_policy.h"
#include "./matmul_impl_traits.h"
#include "./block_mmad_with_params.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<MatmulMultiBlockBias<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulMultiBlockBias<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulMultiBlockBias<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief A template class BlockMmad for performing multi-block matrix multiplication operations
*
* The class is specialized base on MatmulMultiBlockBias<>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy_>
class BlockMmad<MatmulMultiBlockBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_,
    AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType>>>
    : public BlockMmadWithParams<
        BlockMmad<MatmulMultiBlockBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_>,
        MatmulMultiBlockBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_
    > {
public:
    using DispatchPolicy = MatmulMultiBlockBias<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_>;
    using Base = BlockMmadWithParams<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_>;
    friend class BlockMmadWithParams<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy_>;

    using TileCopy = AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                                 Tile::TileCopy<Arch::Ascend910B, Tile::CopyWithParams>, TileCopy_>;
    using MM = MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(
        IsF16OrBf16AB<AType, BType, CType>() || IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
        "Unsupported dtype"
    );
    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<CType>(), "Only support ND format");

public:
    /**
    * @brief Set tensor A for matrix multiplication
    * @param [in] gm: global tensor for matrix A
    * @param [in] isTransposeA: whether to transpose matrix A, default is false
    */
    __aicore__ inline void SetTensorA(const AscendC::GlobalTensor<typename AType::T>& gm, bool isTransposeA = false)
    {
        matmul_.SetTensorA(gm, isTransposeA);
    }
    /**
    * @brief Set tensor B for matrix multiplication
    * @param [in] gm: global tensor for matrix B
    * @param [in] isTransposeB: whether to transpose matrix B, default is false
    */
    __aicore__ inline void SetTensorB(const AscendC::GlobalTensor<typename BType::T>& gm, bool isTransposeB = false)
    {
        matmul_.SetTensorB(gm, isTransposeB);
    }
    /**
    * @brief Set tensor bias for matrix multiplication
    * @param [in] biasGlobal: global tensor for matrix bias
    */
    __aicore__ inline void SetBias(const AscendC::GlobalTensor<typename BiasType::T>& biasGlobal)
    {
        matmul_.SetBias(biasGlobal);
    }
    /**
    * @brief Iterate over all elements and perform matrix multiplication
    * @param [out] cGm: global memory tensor C
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<typename CType::T>& cGm, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(cGm, enAtomic);
    }
    /**
    * @brief Iterate over all elements and store the result in local memory
    * @param [out] ubCmatrix: local memory tensor C matrix
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::LocalTensor<typename CType::T>& ubCmatrix, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(ubCmatrix, enAtomic);
    }

private:
    MM matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
