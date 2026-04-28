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
 * \file block_quant_with_tile_mmad_multi_block.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_QUANT_WITH_TILE_MMAD_MULTI_BLOCK_H
#define MATMUL_BLOCK_BLOCK_QUANT_WITH_TILE_MMAD_MULTI_BLOCK_H

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
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<QuantMatmulWithTileMultiBlock<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using DispatchPolicy = QuantMatmulWithTileMultiBlock<>;
    using L1Shape = L1TileShape_;
    using L0Shape = L0TileShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using TileCopy = AscendC::Std::conditional_t<AscendC::Std::is_same_v<TileCopy_, void>,
                                                 Tile::TileCopy<Arch::Ascend910B, Tile::CopyWithParams>, TileCopy_>;

public:
    static_assert(IsFp8Fp8F32<AType, BType, CType>() || IsFp4Fp4F32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsND<AType>() && IsND<CType>(), "Only support ND format");
    static_assert(IsTileShapeValid<AType, BType, L1Shape, L0Shape>(), "L1Shape or L0Shape is invalid");
    static_assert(IsL1BufferValid<AType, BType, L1Shape>(), "L1 buffer overflow");
    static_assert(IsL0BufferValid<AType, BType, L0Shape>(), "L0 buffer overflow");

    __aicore__ BlockMmad() = default;
    __aicore__ ~BlockMmad() = default;

    template <const auto& MM_CFG, typename Impl, typename InputAType, typename InputBType, typename OutputCType,
              typename InputBiasType>
    struct MatmulPolicyNew : public AscendC::Impl::Detail::MatmulWithScalePolicy<MM_CFG, Impl, InputAType, InputBType,
                                                                                 OutputCType, InputBiasType> {
    public:
        template <class InputType, class OutputType, typename T = void>
        using AdaptedCubeOut = typename TileCopy::template CopyCo1ToOut<InputType, OutputType>;
        using CopyCubeOut =
            AscendC::Impl::Detail::CopyCubeOut<Impl, InputAType, InputBType, OutputCType, MM_CFG,
                                               AscendC::McgShfMode::SINGLE_DST_MODE, void, AdaptedCubeOut>;
    };
    constexpr static MatmulShapeParams shapeParams =
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>();
    constexpr static MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(
        shapeParams, GetFuncParams(DispatchPolicy::enableInputDataLenCheck), GetBiasParams(false));
    constexpr static MatmulApiStaticTiling staticTiling =
        AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                    L0Shape>(cfg);

public:
    __aicore__ inline void Init(TCubeTiling* __restrict cubeTiling, AscendC::TPipe* tpipe = nullptr)
    {
        matmul_.Init(cubeTiling, tpipe);
    }
    __aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
    {
        matmul_.SetOrgShape(orgM, orgN, orgK);
    }
    __aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
    {
        matmul_.SetSingleShape(singleM, singleN, singleK);
    }
    __aicore__ inline void SetTensorA(const AscendC::GlobalTensor<typename AType::T>& gm, bool isTransposeA = false)
    {
        matmul_.SetTensorA(gm, isTransposeA);
    }
    __aicore__ inline void SetTensorB(const AscendC::GlobalTensor<typename BType::T>& gm, bool isTransposeB = false)
    {
        matmul_.SetTensorB(gm, isTransposeB);
    }
    __aicore__ inline void SetTensorScaleA(const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleAGlobal,
                                           bool isTransposeA = false)
    {
        matmul_.SetTensorScaleA(scaleAGlobal, isTransposeA);
    }
    __aicore__ inline void SetTensorScaleB(const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleBGlobal,
                                           bool isTransposeB = false)
    {
        matmul_.SetTensorScaleB(scaleBGlobal, isTransposeB);
    }

    __aicore__ inline void SetBias(const AscendC::GlobalTensor<typename BiasType::T>& biasGlobal)
    {
        matmul_.SetBias(biasGlobal);
    }

    __aicore__ inline void SetSubBlockIdx(uint8_t subBlockIdx)
    {
        matmul_.SetSubBlockIdx(subBlockIdx);
    }
    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(gm, enAtomic);
    }
    __aicore__ inline void IterateAll(const AscendC::LocalTensor<typename CType::T>& ubCmatrix, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(ubCmatrix, enAtomic);
    }
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        return matmul_.Iterate(enPartialSum);
    }
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.GetTensorC(gm, enAtomic);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<typename AType::T>& aGlobal,
                                      const AscendC::GlobalTensor<typename BType::T>& bGlobal,
                                      const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleAGlobal,
                                      const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleBGlobal,
                                      const AscendC::LocalTensor<typename CType::T>& ubCmatrix,
                                      const AscendC::Std::tuple<int32_t, int32_t, int32_t>& singleShape,
                                      bool isTransposeA = false, bool isTransposeB = false)
    {
        matmul_.SetSingleShape(Get<0>(singleShape), Get<1>(singleShape), Get<2>(singleShape)); // 2: idx of k
        matmul_.SetTensorA(aGlobal, isTransposeA);
        matmul_.SetTensorB(bGlobal, isTransposeB);
        matmul_.SetTensorScaleA(scaleAGlobal, isTransposeA);
        matmul_.SetTensorScaleB(scaleBGlobal, isTransposeB);
        matmul_.Iterate();
        matmul_.GetTensorC(ubCmatrix, 0, true);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<typename AType::T>& aGlobal,
                                    const AscendC::GlobalTensor<typename BType::T>& bGlobal,
                                    const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleAGlobal,
                                    const AscendC::GlobalTensor<AscendC::fp8_e8m0_t>& scaleBGlobal,
                                    const AscendC::GlobalTensor<typename BiasType::T>& biasGlobal,
                                    const AscendC::LocalTensor<typename CType::T>& ubCmatrix,
                                    const AscendC::Std::tuple<int32_t, int32_t, int32_t>& singleShape,
                                    bool isTransposeA = false, bool isTransposeB = false)
    {
        matmul_.SetSingleShape(Get<0>(singleShape), Get<1>(singleShape), Get<2>(singleShape)); // 2: idx of k
        matmul_.SetTensorA(aGlobal, isTransposeA);
        matmul_.SetTensorB(bGlobal, isTransposeB);
        matmul_.SetTensorScaleA(scaleAGlobal, isTransposeA);
        matmul_.SetTensorScaleB(scaleBGlobal, isTransposeB);
        matmul_.SetBias(biasGlobal);
        matmul_.Iterate();
        matmul_.GetTensorC(ubCmatrix, 0, true);
    }

    __aicore__ inline void End()
    {
        matmul_.End();
    }

private:
    using MM = AscendC::MatmulImpl<AType, BType, CType, BiasType, staticTiling,
                                   AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyNew>;
    MM matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
