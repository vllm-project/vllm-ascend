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
 * \file block_mmad_multi_block.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_H
#define MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "./block_mmad.h"
#include "./block_mmad_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
* @class BlockMmad
* @brief A template class BlockMmad for performing multi-block matrix multiplication operations
*
* The class is templated with parameters for L1 and L0 shapes, input/output data type, and more
*
* @param [in] MatmulMultiBlock<>: policy type
* @param [in] L1TileShape_: the shape of the L1 cache
* @param [in] L0TileShape_: the shape of the L0 cache
* @param [in] AType_: the data type of matrix A
* @param [in] BType_: the data type of matrix B
* @param [in] CType_: the data type of matrix C
* @param [in] BiasType_: the data type of the bias
* @param [in] TileCopy_: tile copy strategy
*/
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulMultiBlock<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using DispatchPolicy = MatmulMultiBlock<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy = AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                                 Tile::TileCopy<Arch::Ascend910B, Tile::CopyWithParams>, TileCopy_>;

public:
    static_assert(IsF16F16F16<AType, BType, CType>() || IsF16F16F32<AType, BType, CType>() ||
                      IsBf16Bf16Bf16<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>() ||
                      IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>() ||
                      IsHIF8HIF8F32<AType, BType, CType>(),
                  "Unsupported dtype");
    static_assert(IsND<AType>() && IsND<CType>(), "Only support ND format");
    static_assert(IsTileShapeValid<AType, BType, L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad() = default;

    /**
    * @struct MatmulPolicyCustom
    * @brief A custom matrix multiplication policy structure
    * @param [in] MM_CFG: matrix multiplication configuration
    * @param [in] Impl: implementation type
    * @param [in] InputAType: input data type of matrix A
    * @param [in] InputBType: input data type of matrix B
    * @param [out] OutputCType: output data type of matrix C
    * @param [in] InputBiasType: input data type of the bias
    */
    template <const auto& MM_CFG, typename Impl, typename InputAType, typename InputBType, typename OutputCType,
              typename InputBiasType>
    struct MatmulPolicyCustom
        : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, Impl, InputAType, InputBType, OutputCType, InputBiasType> {
    public:
        template <class InputType, const auto& COPY_CFG>
        using AdaptedCubeInA = typename TileCopy::template CopyGmToA1<InputType, COPY_CFG>;
        using CopyCubeInA =
            AscendC::Impl::Detail::CopyCubeIn<Impl, AscendC::MatmulInputAType<InputAType, typename InputAType::T>,
                                              MM_CFG, void, AdaptedCubeInA>;

        template <class InputType, const auto& COPY_CFG>
        using AdaptedCubeInB = typename TileCopy::template CopyGmToB1<InputType, COPY_CFG>;
        using CopyCubeInB =
            AscendC::Impl::Detail::CopyCubeIn<Impl, AscendC::MatmulInputBType<InputBType, typename InputAType::T>,
                                              MM_CFG, void, AdaptedCubeInB>;

        template <class InputType, class OutputType, typename T = void>
        using AdaptedCubeOut = typename TileCopy::template CopyCo1ToOut<InputType, OutputType>;
        using CopyCubeOut =
            AscendC::Impl::Detail::CopyCubeOut<Impl, InputAType, InputBType, OutputCType, MM_CFG,
                                               AscendC::McgShfMode::SINGLE_DST_MODE, void, AdaptedCubeOut>;
    };

    constexpr static MatmulShapeParams shapeParams =
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>();
    constexpr static MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(
        shapeParams, GetFuncParams(DispatchPolicy::ENABLE_INTRINSICS_CHECK), GetBiasParams(false));
    constexpr static MatmulApiStaticTiling staticTiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                    L0Shape>(cfg);

    using MM = AscendC::MatmulImpl<AType, BType, CType, BiasType, staticTiling,
                                   AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyCustom>;

public:
    /**
    * @brief Initialize the matrix multiplication
    * @param [in] cubeTiling: configuration parameters for matrix multiplication
    * @param [in] tpipe: the pipe object, default is nullptr
    */
    __aicore__ inline void Init(TCubeTiling* __restrict cubeTiling, AscendC::TPipe* tpipe = nullptr)
    {
        matmul_.Init(cubeTiling, tpipe);
    }
    /**
    * @brief Set the original shape of the matrix
    * @param [in] orgM: numbers of rows in the original matrix A
    * @param [in] orgN: numbers of columns in the original matrix B
    * @param [in] orgK: depth of the original matrix B
    */
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
    {
        matmul_.SetOrgShape(orgM, orgN, orgK);
    }
    /**
    * @brief Set the single shape of the matrix
    * @param [in] singleM: numbers of rows in the single matrix
    * @param [in] singleN: numbers of columns in the single matrix
    * @param [in] singleK: depth of the single matrix
    */
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
    {
        matmul_.SetSingleShape(singleM, singleN, singleK);
    }
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
    * @brief Set the sub-block index for matrix multiplication
    * @param [in] subBlockIdx: index of the sub-block
    */
    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
        matmul_.SetSubBlockIdx(subBlockIdx);
    }
    /**
    * @brief Iterate over all elements and perform matrix multiplication
    * @param [in] gm: global memory tensor
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(gm, enAtomic);
    }
    /**
    * @brief Iterate over all elements and store the result in local memory
    * @param [in] ubCmatrix: local memory tensor
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::LocalTensor<typename CType::T>& ubCmatrix, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(ubCmatrix, enAtomic);
    }
    /**
    * @brief Iterate
    * @param [in] enPartialSum: whether to enable partial sum, default is false
    * @return whether all iterations are completed
    */
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        return matmul_.Iterate(enPartialSum);
    }
    /**
    * @brief Get tensor C
    * @param [in] gm: global memory tensor
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.GetTensorC(gm, enAtomic);
    }
    /**
    * @brief Perform matrix multiplication
    * @param [in] aGlobal: global memory tensor of matrix A
    * @param [in] bGlobal: global memory tensor of matrix B
    * @param [in] ubCmatrix: local tensor
    * @param [in] singleShape: shape of the single matrix (rows, columns, depth)
    * @param [in] isTransposeA: whether to transpose matrix A
    * @param [in] isTransposeB: whether to transpose matrix B
    */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<typename AType::T>& aGlobal,
                                      const AscendC::GlobalTensor<typename BType::T>& bGlobal,
                                      const AscendC::LocalTensor<typename CType::T>& ubCmatrix,
                                      const AscendC::Std::tuple<int32_t, int32_t, int32_t>& singleShape,
                                      bool isTransposeA = false, bool isTransposeB = false)
    {
        matmul_.SetSingleShape(Get<0>(singleShape), Get<1>(singleShape), Get<2>(singleShape)); // 2: idx of k
        matmul_.SetTensorA(aGlobal, isTransposeA);
        matmul_.SetTensorB(bGlobal, isTransposeB);
        matmul_.Iterate();
        matmul_.GetTensorC(ubCmatrix, 0, true);
    }
    /**
    * @brief End the matrix multiplication operation
    */
    __aicore__ inline void End()
    {
        matmul_.End();
    }

private:
    MM matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
