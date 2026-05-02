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
 * \file block_mmad_with_params.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_WITH_PARAMS_H
#define MATMUL_BLOCK_BLOCK_MMAD_WITH_PARAMS_H

#include <type_traits>
#include "./block_mmad.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
* @class BlockMmadWithLayout
* @brief The intermediate of Block matrix multiplication class
*
* Inheriting from BlockMmadBase, encapsulates and adapts common functions for params representation
*/
template <
    class Derived,
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    class TileCopy
>
class BlockMmadWithParams
    : public BlockMmadBase<Derived, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> {
public:
    using Base = BlockMmadBase<Derived, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base::AsDerived;

    /**
    * @brief Initialize the matrix multiplication object
    * @param [in] cubeTiling: configuration parameters for matrix multiplication
    * @param [in] tpipe: the pipe object, default is nullptr
    */
    __aicore__ inline void Init(TCubeTiling* __restrict cubeTiling, AscendC::TPipe* tpipe = nullptr)
    {
        this->AsDerived().matmul_.Init(cubeTiling, tpipe);
    }

    /**
    * @brief Set the original shape of the matrix
    * @param [in] orgM: original M value
    * @param [in] orgN: original N value
    * @param [in] orgK: original K value
    */
    __aicore__ inline void SetOrgShape(int32_t orgM, int32_t orgN, int32_t orgK)
    {
        this->AsDerived().matmul_.SetOrgShape(orgM, orgN, orgK);
    }

    /**
    * @brief Set the single shape of the matrix
    * @param [in] singleM: single M value
    * @param [in] singleN: single N value
    * @param [in] singleK: single K value
    */
    __aicore__ inline void SetSingleShape(int32_t singleM, int32_t singleN, int32_t singleK)
    {
        this->AsDerived().matmul_.SetSingleShape(singleM, singleN, singleK);
    }

    /**
    * @brief Set the sub-block index for matrix multiplication
    * @param [in] subBlockIdx: index of the sub-block
    */
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
        this->AsDerived().matmul_.SetSubBlockIdx(subBlockIdx);
    }

    /**
    * @brief Iterate
    * @param [in] enPartialSum: whether to enable partial sum, default is false
    * @return whether all iterations are completed
    */
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        return this->AsDerived().matmul_.Iterate(enPartialSum);
    }

    /**
    * @brief Get tensor C
    * @param [out] gm: output tensor for matrix c
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        this->AsDerived().matmul_.GetTensorC(gm, enAtomic);
    }

    /**
    * @brief End the matrix multiplication operation
    */
    __aicore__ inline void End()
    {
        this->AsDerived().matmul_.End();
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
