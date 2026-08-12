/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH35_CATLASS_CONVERTOR_H
#define ARCH35_CATLASS_CONVERTOR_H
#include "catlass/block/david_ub_antiquant_scmc_load_in_advance_aic_tail_resplit.h"
#include "catlass/block/david_ub_antiquant_scmc_load_in_advance_aiv_tail_resplit.h"
#include "catlass/dispatch_policy.h"
#include "catlass/kernel/david_wqbmm_load_in_advance.h"
#include "catlass/scheduler/tile_scheduler_tail_resplit.h"
#include "catlass/utils/constant.h"
#include "catlass/utils/math_utils.h"

using AscendC::int4b_t;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <bool innerK, bool isNz, typename DtypeA, typename DtypeB>
struct XWeightStride {};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<false, false, DtypeA, DtypeB> {
    // k,m
    // k,n
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(_1{}, outer);
    }
};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<true, false, DtypeA, DtypeB> {
    // m,k
    // n,k
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(k, _1{});
    }
};

template <bool innerK, typename DtypeA, typename DtypeB>
struct XWeightStride<innerK, true, DtypeA, DtypeB> {
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        if constexpr ((AscendC::Std::is_same_v<DtypeA, half> || AscendC::Std::is_same_v<DtypeA, bfloat16_t>) &&
                      AscendC::Std::is_same_v<DtypeB, AscendC::int4b_t>) {
            if constexpr (innerK) {
                // n, k -> k1, n1, n0(16), k0(16)
                // m, k -> k1, m1, m0(16), k0(16)
                // stride(以m,k为例) m1*m0*k0, m0*k0(256), k0(16), 1
                return AscendC::Std::make_tuple(CeilDiv(outer, 16UL) * 256, _256{}, _16{}, _1{});
            } else {
                // k, n -> n1, k1, k0(16), n0(16)
                // k, m -> m1, k1, k0(16), m0(16)
                // stride(以k,m为例) k1*k0*m0, k0*m0(256), m0(16), 1
                return AscendC::Std::make_tuple(CeilDiv(k, 16UL) * 256, _256{}, _16{}, _1{});
            }
        }
    }
};

template <bool trans, Mc2QuantType antiquantType>
struct ScaleOffsetStride {};

template <bool trans>
struct ScaleOffsetStride<trans, Mc2QuantType::PER_TENSOR> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(_1{});
    }
};

template <bool trans>
struct ScaleOffsetStride<trans, Mc2QuantType::PER_CHANNEL> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(_1{}, _1{});
    }
};

template <>
struct ScaleOffsetStride<false, Mc2QuantType::PER_GROUP> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(_1{}, n);
    }
};

template <>
struct ScaleOffsetStride<true, Mc2QuantType::PER_GROUP> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(groupSize == 0 ? k : (k + groupSize - 1) / groupSize, _1{});
    }
};

struct TilingKeyParams {
    uint32_t ubMte2InnerSize;
    uint32_t ubMte2BufNum;
    bool transA;
    bool transB;
    Mc2QuantType antiquantType;
    Mc2QuantType quantType;
    bool hasAntiquantOffset;
    bool biasTypeSameAsX;
    bool isWeightNz;

    template <uint64_t TILING_KEY>
    DEVICE static constexpr TilingKeyParams Build()
    {
        constexpr uint32_t customization = TILING_KEY / 1000000000UL % 1000UL;
        constexpr uint8_t trans = TILING_KEY / 10000UL % 10UL;
        constexpr uint8_t optional = TILING_KEY / 10UL % 10UL;

        constexpr uint32_t ubMte2InnerSize = customization == 0 || customization == 1 || customization == 4 ? 512 :
                                             customization == 2 || customization == 6                       ? 1024 :
                                                                                                              256;
        constexpr uint32_t ubMte2BufNum = customization == 0 || customization == 2 || customization == 4 ? 2 : 4;

        return {.ubMte2InnerSize = ubMte2InnerSize,
                .ubMte2BufNum = ubMte2BufNum,
                .transA = trans == 2 || trans == 3,
                .transB = trans == 1 || trans == 3,
                .antiquantType = static_cast<Mc2QuantType>(TILING_KEY / 1000UL % 10UL),
                .quantType = static_cast<Mc2QuantType>(TILING_KEY / 100UL % 10UL),
                .hasAntiquantOffset = optional == 2 || optional == 6,
                .biasTypeSameAsX = optional == 0 || optional == 2,
                .isWeightNz = (TILING_KEY % 10UL) == 1};
    }
};

// PS bias在编译期不清楚，通过tiling获取
using StrideBias = AscendC::Std::tuple<_1>;
using StrideC = AscendC::Std::tuple<uint64_t, _1>;

using ProblemShape = AscendC::Std::tuple<uint64_t, uint64_t, uint64_t>;          // m, n, k
using TileShapeL1 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>; // m, n, ka, kb
using TileShapeL0 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t>;           // m, n, k
using TileShapeUb = AscendC::Std::tuple<uint32_t, uint32_t>;                     // n, k

template <bool weightNz, bool innerK>
struct StrideXWeight {
    // PS delay errors until template instantiation
    static_assert(innerK != innerK, "Unsupported (weightNz, innerK) combination");
};

template <bool innerK>
struct StrideXWeight<true, innerK> {
    using type = AscendC::Std::tuple<uint64_t, _256, _16, _1>;
};

template <>
struct StrideXWeight<false, true> {
    using type = AscendC::Std::tuple<uint64_t, _1>;
};

template <>
struct StrideXWeight<false, false> {
    using type = AscendC::Std::tuple<_1, uint64_t>;
};

template <Mc2QuantType antiquantType, bool innerK>
struct StrideAntiquant {
    // PS delay errors until template instantiation
    static_assert(innerK != innerK, "Unsupported (antiquantType, innerK) combination");
};

template <bool innerK>
struct StrideAntiquant<Mc2QuantType::PER_TENSOR, innerK> {
    using type = AscendC::Std::tuple<_1>;
};

template <bool innerK>
struct StrideAntiquant<Mc2QuantType::PER_CHANNEL, innerK> {
    using type = AscendC::Std::tuple<_1, _1>;
};

template <>
struct StrideAntiquant<Mc2QuantType::PER_GROUP, true> {
    using type = AscendC::Std::tuple<uint64_t, _1>;
};

template <>
struct StrideAntiquant<Mc2QuantType::PER_GROUP, false> {
    using type = AscendC::Std::tuple<_1, uint64_t>;
};

template <int32_t UB_MTE2_INNER_SIZE>
struct TileShapeReg {
    // PS delay errors until template instantiation
    static_assert(UB_MTE2_INNER_SIZE != UB_MTE2_INNER_SIZE, "Unsupported UB_MTE2_INNER_SIZE");
};

template <>
// 寄存器k切分为256
struct TileShapeReg<256> {
    using type = AscendC::Std::tuple<_64, _256>;
};

template <>
// 寄存器k切分为512
struct TileShapeReg<512> {
    using type = AscendC::Std::tuple<_32, _512>;
};

template <>
// 寄存器k切分为1024
struct TileShapeReg<1024> {
    using type = AscendC::Std::tuple<_16, _1024>;
};

template <uint64_t TILING_KEY>
DEVICE void InvokeActKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(Mc2WeightQuantBatchMatmulV2ASTilingData, tilingDataIn, tiling);

    constexpr TilingKeyParams params = TilingKeyParams::Build<TILING_KEY>();
    using StrideA = typename StrideXWeight<false, !params.transA>::type;
    using StrideB = typename StrideXWeight<params.isWeightNz, params.transB>::type;
    using StrideAntiquantScale = typename StrideAntiquant<params.antiquantType, params.transB>::type;
    using StrideBOptionalTuple =
        AscendC::Std::conditional_t<params.hasAntiquantOffset,
                                    AscendC::Std::tuple<StrideB, StrideAntiquantScale, StrideAntiquantScale>,
                                    AscendC::Std::tuple<StrideB, StrideAntiquantScale>>;

    using DtypeBias = AscendC::Std::conditional_t<params.biasTypeSameAsX, DTYPE_X, float>;
#if defined(__DAV_310R6__)
    constexpr int SUB_BLOCK_NUM = 1;
#else
    constexpr int SUB_BLOCK_NUM = 2;
#endif

    using DispatchPolicy =
        MainloopDavidWqbmmUbAntiquantScmc<2, TileShapeUb, typename TileShapeReg<params.ubMte2InnerSize>::type,
                                          g_coreType, SUB_BLOCK_NUM, params.ubMte2BufNum, 2, params.ubMte2InnerSize>;
    using BlockMmad = BlockMmad<DispatchPolicy, TileShapeL1, TileShapeL0, DTYPE_X, StrideA, DTYPE_WEIGHT,
                                StrideBOptionalTuple, DtypeBias, StrideBias, DTYPE_Y, StrideC>;
    using Kernel = wqbmmv2<ProblemShape, BlockMmad, TileSchedulerTailResplit<SUB_BLOCK_NUM>>;

    typename BlockMmad::Arguments args{
        .aGmAddr = (__gm__ DTYPE_X *)x,
        .strideA =
            XWeightStride<!params.transA, false, DTYPE_X, DTYPE_WEIGHT>{}(tilingDataIn.mSize, tilingDataIn.kSize),
        .bGmAddr = (__gm__ DTYPE_WEIGHT *)weight,
        .strideB = XWeightStride<params.transB, params.isWeightNz, DTYPE_X, DTYPE_WEIGHT>{}(tilingDataIn.nSize,
                                                                                            tilingDataIn.kSize),
        .antiquantScaleGmAddr = (__gm__ DTYPE_X *)antiquantScale,
        .strideAntiquantScale = ScaleOffsetStride<params.transB, params.antiquantType>{}(
            tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize),
        .antiquantOffsetGmAddr = (__gm__ DTYPE_X *)antiquantOffset,
        .biasGmAddr = (__gm__ DtypeBias *)bias,
        .strideBias = AscendC::Std::make_tuple(_1{}),
        .cGmAddr = (__gm__ DTYPE_Y *)y,
        .strideC = AscendC::Std::make_tuple(tilingDataIn.nSize, _1{}),
        .groupSize = tilingDataIn.groupSize,
    };

    auto problemShape = AscendC::Std::make_tuple(tilingDataIn.mSize, tilingDataIn.nSize, tilingDataIn.kSize);
    typename BlockMmad::Params mainloopParams = BlockMmad::GetParams(problemShape, args, tilingDataIn);
    typename Kernel::Params kernelParams{problemShape, mainloopParams, mainloopParams.tileShapeL1};
    Kernel op;
    op(kernelParams);
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass

#define KERNEL_PARAMS x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling

#endif
