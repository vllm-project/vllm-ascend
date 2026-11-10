/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file basic_block_vf_nd.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "weight_quant_tool.h"

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType>
struct LocalAddressParam {
    __local_mem__ xType *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantScaleBasePhyAddr1;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1;
    __local_mem__ wType *weightLowBitPhyAddr0;
    __local_mem__ wType *weightLowBitPhyAddr1;
    __local_mem__ xType *weightF16PhyAddr0;
    __local_mem__ xType *weightF16PhyAddr1;
};

template <typename xType>
struct CalculateParam {
    xType offsetValue;
    xType scaleValue;
    uint64_t ubLoop;
};

static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP8_TO_FP32_TRAIT_2 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait FP32_TO_F16_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_ROUND};
static constexpr MicroAPI::CastTrait FP32_TO_F16_EVEN = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                         MicroAPI::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};

static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};

template <typename xType, typename wType, bool hasAntiQuantOffset, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVfLoadScaleOffset(RegTensor<xType> &antiQuantScaleVreg,
                                                         RegTensor<xType> &antiQuantOffsetVreg,
                                                         LocalAddressParam<xType, wType> &localAddressParam,
                                                         const CalculateParam<xType> &calculateParam,
                                                         uint16_t nIdx)
{
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(
            antiQuantOffsetVreg, localAddressParam.antiQuantOffsetBasePhyAddr + nIdx);
    }
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, calculateParam.scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                    localAddressParam.antiQuantScaleBasePhyAddr + nIdx);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdNkVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t nIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + nIdx * ubMte2InnerSize);
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + nIdx * ubMte2InnerSize);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < calculateParam.ubLoop; ubLoopNIdx++) {
        AntiQuantFP8NdNkVfLoadScaleOffset<xType, wType, hasAntiQuantOffset, antiQuantType>(
            antiQuantScaleVreg, antiQuantOffsetVreg, localAddressParam, calculateParam, ubLoopNIdx);
        AntiQuantFP8NdNkVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopNIdx);
        // 奇数、偶数位置分散到2个fp32寄存器存储
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);

        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg, maskAll);
            }
        }

        MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
        MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg, maskAll);

        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset>
__aicore__ inline void AntiQuantFP8NdKnVfLoadOffset(RegTensor<xType> &antiQuantOffsetVreg0,
                                                    RegTensor<xType> &antiQuantOffsetVreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam)
{
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg0,
                                                                 localAddressParam.antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1,
                                                                 localAddressParam.antiQuantOffsetBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVfLoadScale(RegTensor<xType> &antiQuantScaleVreg0,
                                                   RegTensor<xType> &antiQuantScaleVreg1,
                                                   LocalAddressParam<xType, wType> &localAddressParam,
                                                   const CalculateParam<xType> &calculateParam)
{
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg0, calculateParam.scaleValue);
        MicroAPI::Duplicate(antiQuantScaleVreg1, calculateParam.scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg0,
                                                                 localAddressParam.antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1,
                                                                 localAddressParam.antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdKnVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t kIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + kIdx * ubMte2InnerSize);
    MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + kIdx * ubMte2InnerSize);
}

template <typename xType, typename wType>
__aicore__ inline void AntiQuantFP8NdKnVfStoreWeight(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg2,
                                                     LocalAddressParam<xType, wType> &localAddressParam, MaskReg &mask)
{
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg0, antiQuantScaleVreg1, antiQuantOffsetVreg0, antiQuantOffsetVreg1;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AntiQuantFP8NdKnVfLoadOffset<xType, wType, hasAntiQuantOffset>(antiQuantOffsetVreg0, antiQuantOffsetVreg1,
                                                                   localAddressParam);
    AntiQuantFP8NdKnVfLoadScale<xType, wType, hasAntiQuantOffset, antiQuantType>(
        antiQuantScaleVreg0, antiQuantScaleVreg1, localAddressParam, calculateParam);
    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < calculateParam.ubLoop; ubLoopKIdx++) {
        AntiQuantFP8NdKnVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopKIdx);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        MicroAPI::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        MicroAPI::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg0,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);
        MicroAPI::Or<uint16_t, MicroAPI::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg2,
                                                                 (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                MicroAPI::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg0, maskAll);
                MicroAPI::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg1, maskAll);
            }
        }
        MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
        MicroAPI::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg1, maskAll);
        AntiQuantFP8NdKnVfStoreWeight<xType, wType>(weightF16Vreg0, weightF16Vreg2, localAddressParam, maskAll);
    }
}

template <typename xType, typename wType>
__aicore__ inline void CastS8RegTensorToF16RegTensor(RegTensor<xType> &weightF16Vreg, RegTensor<wType> &weightS8Vreg,
                                                     RegTensor<half> &weightFp16Vreg, MicroAPI::MaskReg &maskAll)
{
    // PART_EVEN 表示按照如下形式处理做cast：
    // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
    if constexpr (!IsSameType<typename MicroAPI::TypeGet<xType>::T, vector_f16>::value) {
        MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightS8Vreg, maskAll);
        MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
    } else {
        MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS8Vreg, maskAll);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdKn(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg,
                                                    RegTensor<xType> &antiQuantScaleVreg1,
                                                    RegTensor<xType> &antiQuantOffsetVreg1, MicroAPI::MaskReg &maskAll,
                                                    const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
            MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
        }
    }

    MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
}

template <typename xType, typename wType>
__aicore__ inline void WeightF16NdRegTensorToNzUb(__local_mem__ xType *&weightF16PhyAddr0,
                                                  __local_mem__ xType *&weightF16PhyAddr1,
                                                  RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                  MicroAPI::MaskReg &maskAll)
{
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdNk(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg, MicroAPI::MaskReg &maskAll,
                                                    const xType &scaleValue, const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
            MicroAPI::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            MicroAPI::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            MicroAPI::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            MicroAPI::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
        }
    }

    MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdNkLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           RegTensor<xType> &antiQuantScaleVreg,
                                           RegTensor<xType> &antiQuantOffsetVreg,
                                           xType scaleValue, uint64_t ubLoopNIdx)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg,
                                                                    antiQuantOffsetBasePhyAddr + ubLoopNIdx);
    }
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                                    antiQuantScaleBasePhyAddr + ubLoopNIdx);
    }
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdKnLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantScaleBasePhyAddr1,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr1,
                                           RegTensor<xType> &antiQuantScaleVreg,
                                           RegTensor<xType> &antiQuantScaleVreg1,
                                           RegTensor<xType> &antiQuantOffsetVreg,
                                           RegTensor<xType> &antiQuantOffsetVreg1,
                                           xType scaleValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
    }
    if constexpr (wqmmConfig.antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(antiQuantScaleVreg, scaleValue);
        MicroAPI::Duplicate(antiQuantScaleVreg1, scaleValue);
    } else {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<xType> antiQuantScaleVreg1;
        RegTensor<xType> antiQuantOffsetVreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1,
                                               antiQuantOffsetVreg, antiQuantOffsetVreg1,
                                               scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg1, maskAll, offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopN)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopN)
{
    // int4每次处理128个数即为64B, 256>>2=64
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1, antiQuantOffsetVreg1;
        RegTensor<int4x2_t> weightS4Vreg0, weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0, weightF16Vreg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1,
                                               antiQuantOffsetVreg, antiQuantOffsetVreg1,
                                               scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            MicroAPI::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(
                weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1,
                antiQuantOffsetVreg1, maskAll, offsetValue);

            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_ND_H