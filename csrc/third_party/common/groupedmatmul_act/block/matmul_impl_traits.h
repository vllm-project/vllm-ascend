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
 * \file matmul_impl_traits.h
 * \brief
 */

#ifndef MATMUL_BLOCK_MATMUL_IMPL_TRAITS_H
#define MATMUL_BLOCK_MATMUL_IMPL_TRAITS_H

#include "../utils/common_utils.h"
#include "../utils/tuple_utils.h"
#include "../tile/tile_copy.h"

namespace Act {
namespace Gemm {
namespace Block {
namespace Detail {
/**
 * @brief Get matrix shape parameters
 * @param [in] SingleShape: single shape
 * @param [in] L0TileShape: L0TileShape type
 * @return Return matrix shape parameters
 */
template <class SingleShape, class L0TileShape>
__aicore__ inline constexpr MatmulShapeParams GetMatmulShapeParams()
{
    return {GetIntegralConstant<MNK_M, SingleShape>(), GetIntegralConstant<MNK_N, SingleShape>(),
            GetIntegralConstant<MNK_K, SingleShape>(), GetIntegralConstant<MNK_M, L0TileShape>(),
            GetIntegralConstant<MNK_N, L0TileShape>(), GetIntegralConstant<MNK_K, L0TileShape>()};
}

/**
 * @brief Get function parameters
 * @param [in] intrinsicsCheck: whether to perform intrinsic checks
 * @return Return function parameters
 */
template <bool intrinsicsCheck>
__aicore__ inline constexpr auto GetFuncParams()
{
    MatmulFuncParams params{};
    params.intrinsicsCheck = intrinsicsCheck;
    return params;
}

/**
 * @brief Get bias parameters
 * @param [in] enableSetBias: whether to enable bias setting
 * @return Return bias parameters
 */
template <bool enableSetBias>
__aicore__ inline constexpr auto GetBiasParams()
{
    MatmulBiasParams params{};
    params.enableSetBias = enableSetBias;
    return params;
}

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType
>
__aicore__ inline constexpr auto GetStaticTiling()
{
    constexpr auto CONFIG = GetMMConfig<DispatchPolicy::CONFIG>(
        GetMatmulShapeParams<typename DispatchPolicy::SingleShape, L0Shape>(),
        GetFuncParams<DispatchPolicy::ENABLE_INTRINSICS_CHECK>(),
        GetBiasParams<DispatchPolicy::ENABLE_SET_BIAS>()
    );
    return AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType, typename DispatchPolicy::SingleShape, L1Shape,
                                       L0Shape>(CONFIG);
}
} // namespace Detail

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    class TileCopy,
    template <const auto&, typename, typename, typename, typename, typename> class MatmulPolicyClass,
    typename Enable = void
>
struct MatmulImplTraits {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>,
                  "MatmulImplTraits is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType
>
struct MatmulImplTraits<
    DispatchPolicy,
    L1Shape, L0Shape,
    AType, BType, CType, BiasType,
    void,
    AscendC::Impl::Detail::MatmulPolicy
> {
    constexpr static auto STATIC_TILING =
        Detail::GetStaticTiling<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType>();

    using Type = AscendC::MatmulImpl<AType, BType, CType, BiasType, STATIC_TILING>;
};

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    template <const auto&, typename, typename, typename, typename, typename> class MatmulPolicyClass
>
struct MatmulImplTraits<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, void, MatmulPolicyClass> {
    constexpr static auto STATIC_TILING =
        Detail::GetStaticTiling<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType>();

    template <const auto& MM_CFG, typename Impl, typename AT, typename BT, typename CT, typename BiasT>
    struct MatmulPolicyCustom : public MatmulPolicyClass<MM_CFG, Impl, AT, BT, CT, BiasT> {};

    using Type = AscendC::MatmulImpl<AType, BType, CType, BiasType, STATIC_TILING,
                                     AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyCustom>;
};

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    class TileCopy,
    template <const auto&, typename, typename, typename, typename, typename> class MatmulPolicyClass
>
struct MatmulImplTraits<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy, MatmulPolicyClass> {
    /**
    * @brief Define a custom matrix multiplication policy
    * @param [in] MM_CFG: matrix multiplication configuration
    * @param [in] Impl: implementation type
    * @param [in] InputAType: input data type of matrix A
    * @param [in] InputBType: input data type of matrix B
    * @param [out] OutputCType: output data type of matrix C
    * @param [in] InputBiasType: input data type of the bias
    */
    template <const auto& MM_CFG, typename Impl, typename AT, typename BT, typename CT, typename BiasT>
    struct MatmulPolicyCustom : public MatmulPolicyClass<MM_CFG, Impl, AT, BT, CT, BiasT> {
    private:
        using Base = MatmulPolicyClass<MM_CFG, Impl, AT, BT, CT, BiasT>;
        struct AdaptedCopyInA {
            using CopyInAType = AscendC::Impl::Detail::CopyCubeIn<Impl, AscendC::MatmulInputAType<AT, typename AT::T>,
                                                                  MM_CFG, void, TileCopy::template CopyGmToA1>;
        };
        struct AdaptedCopyInB {
            using CopyInBType = AscendC::Impl::Detail::CopyCubeIn<Impl, AscendC::MatmulInputBType<BT, typename AT::T>,
                                                                  MM_CFG, void, TileCopy::template CopyGmToB1>;
        };
        struct AdaptedCopyOut {
            using CopyOutType = AscendC::Impl::Detail::CopyCubeOut<Impl, AT, BT, CT, MM_CFG,
                                                                   AscendC::McgShfMode::RESERVED, void,
                                                                   TileCopy::template CopyCo1ToOut>;
        };
        struct BaseCopyType {
            using CopyInAType = typename Base::CopyCubeInA;
            using CopyInBType = typename Base::CopyCubeInB;
            using CopyOutType = typename Base::CopyCubeOut;
        };

    public:
        using CopyCubeInA = typename AscendC::Std::conditional_t<
            Tile::HasCopyGmToA1V<TileCopy>,
            AdaptedCopyInA,
            BaseCopyType
        >::CopyInAType;

        using CopyCubeInB = typename AscendC::Std::conditional_t<
            Tile::HasCopyGmToB1V<TileCopy>,
            AdaptedCopyInB,
            BaseCopyType
        >::CopyInBType;

        using CopyCubeOut = typename AscendC::Std::conditional_t<
            Tile::HasCopyCo1ToOutV<TileCopy>,
            AdaptedCopyOut,
            BaseCopyType
        >::CopyOutType;
    };

    constexpr static auto STATIC_TILING =
        Detail::GetStaticTiling<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType>();
    using Type = AscendC::MatmulImpl<AType, BType, CType, BiasType, STATIC_TILING,
                                     AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>, MatmulPolicyCustom>;
};

template <
    class DispatchPolicy,
    class L1Shape, class L0Shape,
    class AType, class BType, class CType, class BiasType,
    class TileCopy = void,
    template <const auto&, typename, typename, typename, typename, typename> class MatmulPolicyClass =
        AscendC::Impl::Detail::MatmulPolicy
>
using MatmulImplTraitsT = typename MatmulImplTraits<
    DispatchPolicy,
    L1Shape, L0Shape,
    AType, BType, CType, BiasType,
    TileCopy,
    MatmulPolicyClass
>::Type;
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
