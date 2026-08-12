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
 * \file weight_quant_batch_matmul_v2_vf.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_VF_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_VF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename TIN>
struct RegTensorActualT {
    using T = TIN;
};
template <>
struct RegTensorActualT<AscendC::int4b_t> {
    using T = int4x2_t;
};

static constexpr MicroAPI::CastTrait castTraitNorm = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

// CAST_RINT表示采用四舍六入五成双的舍入模式
static constexpr MicroAPI::CastTrait castTraitF162Bf16 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename XType>
struct ParamsW8NK {
    uint32_t groupNum;
    uint32_t groupSize;
    uint32_t groupTail;
    uint32_t vlNum;
    uint32_t tailVlNum;
    uint32_t nBub;
    uint32_t kBubXTypeAlign;
    uint32_t kBubWTypeAlign;
    uint32_t vfElemB16;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    uint32_t scaleNStride;
    int32_t weightOutFixStride;
    int32_t weightOutGroupFixStride;
    int32_t weightOutTailFixStride;

    __local_mem__ XType *offsetBaseAddr;
    __local_mem__ XType *scaleBaseAddr;
    __local_mem__ XType *offsetTailAddr;
    __local_mem__ XType *scaleTailAddr;
    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ int8_t *weightInTailAddr;
    __local_mem__ XType *weightOutBaseAddr;
    __local_mem__ XType *weightOutTailAddr;
};

template <typename XType>
struct ParamsKN {
    uint32_t kBub;
    uint32_t nBub;
    uint32_t nBubXTypeAlign;
    uint32_t nBubWTypeAlign;
    uint32_t nBubTail;
    uint32_t groupSize;
    uint32_t groupNum;
    uint32_t groupTail;
    uint32_t nLoop;
    uint32_t vfElemB16;
    uint32_t wNStride;
    uint32_t wKStride;
    uint32_t wGroupStride;
    int32_t weightOutStride;
    uint32_t dataBlockStride;
    uint32_t repeatStride;

    __local_mem__ XType *offsetBaseAddr;
    __local_mem__ XType *offsetTailAddr;

    __local_mem__ XType *scaleBaseAddr;
    __local_mem__ XType *scaleTailAddr;

    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ int8_t *weightInGroupTailAddr;

    __local_mem__ XType *weightOutBaseAddr;
};

template <typename XType, typename WType>
struct ParamsGroupSizeGt128 {
    int32_t innerExtend;
    int32_t bubNLen;
    int32_t weightOutAddrOffset;
    uint32_t groupNumBub;
    uint32_t dataBlockStride;
    uint32_t repeatStride0;
    uint32_t repeatStride1;
    uint32_t tailKLen;
    uint32_t groupSize;
    uint32_t groupSizeInByte;
    uint32_t numVLInKLen;
    uint32_t numVLInGroup;
    uint32_t numVLInRemainGroup;
    uint32_t oriGroupSize;
    uint32_t resGrpMod128;
    uint32_t resRemainGrpMod128;
    uint32_t tailGroupInBubKLen;
    uint32_t tailVLInGroup;
    uint32_t tailVLInTailGroup;
    uint32_t vlB4SizeInByte;
    uint32_t tailVLInKLen;
    uint64_t mainGroupNum;
    uint64_t mainGroupSize;
    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ XType *offsetBaseAddr;
    __local_mem__ XType *scaleBaseAddr;
    __local_mem__ XType *weightOutBaseAddr0;
    __local_mem__ XType *oriWeightOutBaseAddr0;
    __local_mem__ XType *weightOutBaseAddr1;
    __local_mem__ XType *oriWeightOutBaseAddr1;
};

template <typename XType, typename WType>
struct ParamsGroupSize128 {
    int32_t innerExtend;
    int32_t bubNLenUnrollTwo;
    int32_t weightOutAddrOffset;
    uint32_t groupNumBub;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    uint32_t tailKLen;
    uint32_t groupSize;
    uint32_t groupSizeInByte;
    uint32_t oriGroupSize;
    uint64_t mainGroupNum;
    uint64_t mainGroupSize;
    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ XType *offsetBaseAddr;
    __local_mem__ XType *scaleBaseAddr;
    __local_mem__ XType *weightOutBaseAddr0;
    __local_mem__ XType *weightOutBaseAddr1;
};

template <typename XType, typename WType>
struct ParamsGroupSize64 {
    uint16_t outerExtend;
    uint16_t innerExtend;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    uint32_t outerStride;
    int32_t outerStrideScale;
    int32_t outerStrideWeight;
    uint32_t maskWeight;

    __local_mem__ XType *offsetBaseAddr00;
    __local_mem__ XType *offsetBaseAddr01;

    __local_mem__ XType *scaleBaseAddr00;
    __local_mem__ XType *scaleBaseAddr01;

    __local_mem__ int8_t *weightInBaseAddr0;

    __local_mem__ XType *weightOutBaseAddr0;
};

template <typename XType, typename WType>
struct ParamsGroupSize128And256 {
    uint32_t weightOutAddrOffset;
    uint32_t maskWeight;
    uint32_t maskWeight1;

    int32_t bubNLen;
    uint16_t groupNum;
    uint32_t outerStrideScale;
    uint32_t outerStrideWeight;
    uint32_t innerStrideWeight;

    uint32_t dataBlockStride;
    uint32_t repeatStride;

    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ int8_t *weightInBaseAddr1;
    __local_mem__ XType *weightOutBaseAddr;
    __local_mem__ XType *weightOutBaseAddr1;
    __local_mem__ XType *scaleBaseAddr;
    __local_mem__ XType *offsetBaseAddr;
};

template <typename XType, typename WType>
struct ParamsGroupSize32 {
    uint32_t maskWeight;
    uint32_t maskWeight1;
    uint16_t outerExtend;
    uint32_t outerStrideScale;
    uint32_t outerStrideWeight;
    uint16_t innerExtend;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    int32_t outDimOffset;
    __local_mem__ XType *offsetBaseAddr0;
    __local_mem__ XType *scaleBaseAddr0;
    __local_mem__ int8_t *weightInBaseAddr;
    __local_mem__ XType *weightOutBaseAddr0;
    __local_mem__ XType *weightOutBaseAddr1;
};

template <typename XType>
struct ParamsGroupSize32OddNK {
    uint32_t bubNLen;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    uint32_t maskWeightOdd;
    uint32_t maskWeightEven;
    uint32_t scaleNStride;
    uint32_t weightNStride;
    uint32_t scaleGroupPairStride;
    uint32_t weightGroupPairStride;
    uint32_t weightVLStride;
    uint32_t oddGroupVLNum;
    uint32_t evenGroupVLNum;
    uint32_t groupPairNum;
    uint32_t offsetNWeightOutOdd;
    uint32_t offsetNWeightOutBorder;
    uint32_t offsetNWeightOutEven;
    uint32_t offsetKWeightOut;

    __local_mem__ int8_t *weightInOddAddr;
    __local_mem__ int8_t *weightInTailAddr;
    __local_mem__ int8_t *weightInEvenAddr;
    __local_mem__ int8_t *weightInBorderAddr;

    __local_mem__ XType *offsetOddAddr;
    __local_mem__ XType *scaleOddAddr;
    __local_mem__ XType *offsetEvenAddr;
    __local_mem__ XType *scaleEvenAddr;
    __local_mem__ XType *weightOutOddAddr;
    __local_mem__ XType *weightOutEvenAddr;
    __local_mem__ XType *weightOutBorderAddr;
};

template <typename XType, typename WType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantW8PerGroupNK(ParamsW8NK<XType> &param)
{
    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightIn;
    MicroAPI::RegTensor<XType> weightOut;
    uint32_t maskWeightValue;
    MicroAPI::MaskReg maskWeight;

    for (uint16_t gIdx = 0; gIdx < param.groupNum; ++gIdx) {
        for (uint16_t nIdx = 0; nIdx < param.nBub; ++nIdx) {
            MicroAPI::AddrReg addRegScale = MicroAPI::CreateAddrReg<XType>(gIdx, 1, nIdx, param.scaleNStride);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offset, param.offsetBaseAddr, addRegScale);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, param.scaleBaseAddr, addRegScale);

            maskWeightValue = param.groupSize;
            for (uint16_t kIdx = 0; kIdx < param.vlNum; ++kIdx) {
                maskWeight = MicroAPI::UpdateMask<XType>(maskWeightValue);
                MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    gIdx, param.groupSize, nIdx, param.kBubWTypeAlign, kIdx, param.vfElemB16);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInBaseAddr),
                    addRegWeight);

                if constexpr (AscendC::IsSameType<XType, half>::value) {
                    MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                              maskWeight);
                } else {
                    MicroAPI::RegTensor<half> weightF16;
                    MicroAPI::Cast<half, typename RegTensorActualT<WType>::T, castTraitNorm>(weightF16, weightIn,
                                                                                             maskWeight);
                    MicroAPI::Cast<XType, half, castTraitF162Bf16>(weightOut, weightF16, maskWeight);
                }

                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
                }
                MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    param.weightOutBaseAddr, weightOut, param.dataBlockStride, param.repeatStride, maskWeight);
            }
            param.weightOutBaseAddr += param.weightOutFixStride;
        }
        param.weightOutBaseAddr += param.weightOutGroupFixStride;
    }

    // 处理groupTail部分
    for (uint16_t nIdx = 0; nIdx < param.nBub; ++nIdx) {
        MicroAPI::AddrReg addRegScaleTail = MicroAPI::CreateAddrReg<XType>(nIdx, param.scaleNStride);
        if constexpr (hasAntiquantOffset) {
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offset, param.offsetTailAddr, addRegScaleTail);
        }
        MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, param.scaleTailAddr, addRegScaleTail);

        maskWeightValue = param.groupTail;
        for (uint16_t kIdx = 0; kIdx < param.tailVlNum; ++kIdx) {
            maskWeight = MicroAPI::UpdateMask<XType>(maskWeightValue);
            MicroAPI::AddrReg addRegWeightTail = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                nIdx, param.kBubWTypeAlign, kIdx, param.vfElemB16);
            MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInTailAddr),
                addRegWeightTail);

            if constexpr (AscendC::IsSameType<XType, half>::value) {
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                          maskWeight);
            } else {
                MicroAPI::RegTensor<half> weightF16;
                MicroAPI::Cast<half, typename RegTensorActualT<WType>::T, castTraitNorm>(weightF16, weightIn,
                                                                                         maskWeight);
                MicroAPI::Cast<XType, half, castTraitF162Bf16>(weightOut, weightF16, maskWeight);
            }

            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
            }
            MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                param.weightOutTailAddr, weightOut, param.dataBlockStride, param.repeatStride, maskWeight);
        }
        param.weightOutTailAddr += param.weightOutTailFixStride;
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantPerGroupKN(ParamsKN<XType> &param)
{
    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightIn;
    MicroAPI::RegTensor<XType> weightOut;
    MicroAPI::MaskReg maskWeight = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    uint32_t maskWeightValue = param.nBubXTypeAlign;

    for (uint16_t nIdx = 0; nIdx < param.nLoop; ++nIdx) {
        maskWeight = MicroAPI::UpdateMask<XType>(maskWeightValue);
        for (uint16_t gIdx = 0; gIdx < param.groupNum; ++gIdx) {
            MicroAPI::AddrReg addRegScale =
                MicroAPI::CreateAddrReg<XType>(nIdx, param.vfElemB16, gIdx, param.nBubXTypeAlign);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(offset, param.offsetBaseAddr, addRegScale);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(scale, param.scaleBaseAddr, addRegScale);

            for (uint16_t kIdx = 0; kIdx < param.groupSize; ++kIdx) {
                if constexpr (AscendC::IsSameType<WType, int8_t>::value) {
                    MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                        nIdx, param.wNStride, gIdx, param.wGroupStride, kIdx, param.wKStride);
                    MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                        weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInBaseAddr),
                        addRegWeight);
                    if constexpr (AscendC::IsSameType<XType, half>::value) {
                        MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                                  maskWeight);
                    } else {
                        MicroAPI::RegTensor<half> weightF16;
                        MicroAPI::Cast<half, typename RegTensorActualT<WType>::T, castTraitNorm>(weightF16, weightIn,
                                                                                                 maskWeight);
                        MicroAPI::Cast<XType, half, castTraitF162Bf16>(weightOut, weightF16, maskWeight);
                    }
                } else {
                    MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                        nIdx, param.wNStride, gIdx, param.wGroupStride, kIdx, param.wKStride);
                    MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInBaseAddr),
                        addRegWeight);
                    MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                              maskWeight);
                }

                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
                }
                MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    param.weightOutBaseAddr, weightOut, param.dataBlockStride, param.repeatStride, maskWeight);
            }
        }

        // 处理groupTail部分
        MicroAPI::AddrReg addRegScaleTail = MicroAPI::CreateAddrReg<XType>(nIdx, param.vfElemB16);
        if constexpr (hasAntiquantOffset) {
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(offset, param.offsetTailAddr, addRegScaleTail);
        }
        MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_NORM>(scale, param.scaleTailAddr, addRegScaleTail);

        for (uint16_t kIdx = 0; kIdx < param.groupTail; ++kIdx) {
            if constexpr (AscendC::IsSameType<WType, int8_t>::value) {
                MicroAPI::AddrReg addRegWeightTail = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    nIdx, param.wNStride, kIdx, param.wKStride);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInGroupTailAddr),
                    addRegWeightTail);
                if constexpr (AscendC::IsSameType<XType, half>::value) {
                    MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                              maskWeight);
                } else {
                    MicroAPI::RegTensor<half> weightF16Tail;
                    MicroAPI::Cast<half, typename RegTensorActualT<WType>::T, castTraitNorm>(weightF16Tail, weightIn,
                                                                                             maskWeight);
                    MicroAPI::Cast<XType, half, castTraitF162Bf16>(weightOut, weightF16Tail, maskWeight);
                }
            } else {
                MicroAPI::AddrReg addRegWeightTail = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    nIdx, param.wNStride, kIdx, param.wKStride);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightIn, (__local_mem__ typename RegTensorActualT<WType>::T *)(param.weightInGroupTailAddr),
                    addRegWeightTail);
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTraitNorm>(weightOut, weightIn,
                                                                                          maskWeight);
            }
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightOut, weightOut, offset, maskWeight);
            }
            MicroAPI::Mul(weightOut, weightOut, scale, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                param.weightOutBaseAddr, weightOut, param.dataBlockStride, param.repeatStride, maskWeight);
        }
        param.weightOutBaseAddr += param.weightOutStride;
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag>
__aicore__ inline void AntiquantW4Pergroup32NK(ParamsGroupSize32<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset0;
    MicroAPI::RegTensor<XType> offset1;
    MicroAPI::RegTensor<XType> scale0;
    MicroAPI::RegTensor<XType> scale1;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4Vl0;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4Vl1;
    MicroAPI::RegTensor<XType> weightB16Vl0;
    MicroAPI::RegTensor<XType> weightB16Vl1;

    MicroAPI::MaskReg maskRegB4 = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskRegB16 = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    MicroAPI::MaskReg maskWeight1 = MicroAPI::UpdateMask<XType>(p.maskWeight1);

    // (n, k) -> (k1, n1, n0, k0)
    for (uint16_t innerIdx = 0; innerIdx < (uint16_t)p.innerExtend; ++innerIdx) {
        // 按照一列一列处理
        for (uint16_t outerIdx = 0; outerIdx < (uint16_t)p.outerExtend; ++outerIdx) {
            if constexpr (useVag) {
                MicroAPI::AddrReg addrRegScale =
                    MicroAPI::CreateAddrReg<XType>(innerIdx, 8, outerIdx, p.outerStrideScale);
                MicroAPI::AddrReg addrRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    innerIdx, 128, outerIdx, p.outerStrideWeight);

                // 载入scale和offset
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_E2B_B16>(offset0, p.offsetBaseAddr0,
                                                                                addrRegScale);
                    MicroAPI::Interleave(offset0, offset1, offset0, offset0);
                }
                // DIST_E2B_B16 表示搬运模式为
                // SRC ： 0 1 2 3 4 5 6 7
                // DST ： 00000000000000001111111111111111222222222222222233333333333333333.............7777777777777777

                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_E2B_B16>(scale0, p.scaleBaseAddr0, addrRegScale);
                // Interleave后变为
                // scale0:
                // 00000000000000000000000000000000111111111111111111111111111111111.............333333333333333333333333333333333
                // scale1:
                // 44444444444444444444444444444444555555555555555555555555555555555.............777777777777777777777777777777777
                MicroAPI::Interleave(scale0, scale1, scale0, scale0);

                // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                // Vd 0 1 x x x x x x 2 3 x x x x x x

                // 对于256个数来说， 分两次处理， 每次处理128个数，即64B，应为地址按照int8存的，所以每次偏移64个int8的数
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4Vl0, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr),
                    addrRegWeight);

                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4Vl1, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + 64),
                    addrRegWeight);
            } else {
                // 载入scale和offset
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_E2B_B16>(
                        offset0, p.offsetBaseAddr0 + innerIdx * 8 + outerIdx * p.outerStrideScale);
                    MicroAPI::Interleave(offset0, offset1, offset0, offset0);
                }
                // DIST_E2B_B16 表示搬运模式为
                // SRC ： 0 1 2 3 4 5 6 7
                // DST ： 00000000000000001111111111111111222222222222222233333333333333333.............7777777777777777

                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_E2B_B16>(scale0, p.scaleBaseAddr0 + innerIdx * 8 +
                                                                                        outerIdx * p.outerStrideScale);
                // Interleave后变为
                // scale0:
                // 00000000000000000000000000000000111111111111111111111111111111111.............333333333333333333333333333333333
                // scale1:
                // 44444444444444444444444444444444555555555555555555555555555555555.............777777777777777777777777777777777
                MicroAPI::Interleave(scale0, scale1, scale0, scale0);

                // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                // Vd 0 1 x x x x x x 2 3 x x x x x x

                // 对于256个数来说， 分两次处理， 每次处理128个数，即64B，应为地址按照int8存的，所以每次偏移64个int8的数
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4Vl0,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + innerIdx * 128 +
                                                                          outerIdx * p.outerStrideWeight));

                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4Vl1,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + 64 + innerIdx * 128 +
                                                                          outerIdx * p.outerStrideWeight));
            }

            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16Vl0, weightB4Vl0, maskRegB4);
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16Vl1, weightB4Vl1, maskRegB4);

            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16Vl0, weightB16Vl0, offset0, maskRegB16);
                MicroAPI::Add(weightB16Vl1, weightB16Vl1, offset1, maskRegB16);
            }
            MicroAPI::Mul(weightB16Vl0, weightB16Vl0, scale0, maskRegB16);
            MicroAPI::Mul(weightB16Vl1, weightB16Vl1, scale1, maskRegB16);

            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr0, weightB16Vl0, p.dataBlockStride, p.repeatStride, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr1, weightB16Vl1, p.dataBlockStride, p.repeatStride, maskWeight1);
        }
        p.weightOutBaseAddr0 += p.outDimOffset;
        p.weightOutBaseAddr1 += p.outDimOffset;
        maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
        maskWeight1 = MicroAPI::UpdateMask<XType>(p.maskWeight1);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag>
__aicore__ inline void AntiquantW4Pergroup64NK(ParamsGroupSize64<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> oriOffset00;
    MicroAPI::RegTensor<XType> oriOffset01;
    MicroAPI::RegTensor<XType> offset0;
    MicroAPI::RegTensor<XType> oriScale00;
    MicroAPI::RegTensor<XType> oriScale01;
    MicroAPI::RegTensor<XType> scale0;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weight0B4;
    MicroAPI::RegTensor<XType> weight0B16;
    MicroAPI::MaskReg maskRegB4 = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    MicroAPI::MaskReg maskRegSelect = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::H>();

    for (uint16_t innerIdx = 0; innerIdx < (uint16_t)p.innerExtend; ++innerIdx) {
        for (uint16_t outerIdx = 0; outerIdx < (uint16_t)p.outerExtend; ++outerIdx) {
            if constexpr (useVag) {
                // 手写areg，避免部分场景编译器无法优化
                // 每次循环读取2个元素，通过vsel拼接成一个VECTOR_REG_WIDTH
                MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<XType>(innerIdx, 2, outerIdx, p.outerStrideScale);
                // DIST_UNPACK4_B8每次读取1/4 VECTOR_REG_WIDTH，也就是64B
                MicroAPI::AddrReg aregWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    innerIdx, 64, outerIdx, p.outerStrideWeight);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(oriOffset00, p.offsetBaseAddr00,
                                                                                aregScale);
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(oriOffset01, p.offsetBaseAddr01,
                                                                                aregScale);
                    MicroAPI::Select(offset0, oriOffset00, oriOffset01, maskRegSelect);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(oriScale00, p.scaleBaseAddr00, aregScale);
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(oriScale01, p.scaleBaseAddr01, aregScale);
                MicroAPI::Select(scale0, oriScale00, oriScale01, maskRegSelect);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weight0B4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr0), aregWeight);
            } else {
                // 不手写areg，避免出现areg限制导致的编译失败
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        oriOffset00, p.offsetBaseAddr00 + innerIdx * 2 + outerIdx * p.outerStrideScale);
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        oriOffset01, p.offsetBaseAddr01 + innerIdx * 2 + outerIdx * p.outerStrideScale);
                    MicroAPI::Select(offset0, oriOffset00, oriOffset01, maskRegSelect);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    oriScale00, p.scaleBaseAddr00 + innerIdx * 2 + outerIdx * p.outerStrideScale);
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    oriScale01, p.scaleBaseAddr01 + innerIdx * 2 + outerIdx * p.outerStrideScale);
                MicroAPI::Select(scale0, oriScale00, oriScale01, maskRegSelect);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weight0B4,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr0 + innerIdx * 64 +
                                                                          outerIdx * p.outerStrideWeight));
            }

            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weight0B16, weight0B4, maskRegB4);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weight0B16, weight0B16, offset0, maskWeight);
            }
            MicroAPI::Mul(weight0B16, weight0B16, scale0, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr0, weight0B16, p.dataBlockStride, p.repeatStride, maskWeight);
        }
        p.weightOutBaseAddr0 += p.outerStride;
        maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag>
__aicore__ inline void AntiquantW4Pergroup128NK(ParamsGroupSize128And256<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);

    for (uint16_t groupIdx = 0; groupIdx < p.groupNum; ++groupIdx) {
        for (uint16_t nBubIdx = 0; nBubIdx < p.bubNLen; ++nBubIdx) {
            if constexpr (useVag) {
                MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<XType>(groupIdx, 1, nBubIdx, p.outerStrideScale);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offset, p.offsetBaseAddr, aregScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr, aregScale);

                MicroAPI::AddrReg aregWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    groupIdx, p.innerStrideWeight, nBubIdx, p.outerStrideWeight);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr), aregWeight);
            } else {
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        offset, p.offsetBaseAddr + groupIdx + nBubIdx * p.outerStrideScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr + groupIdx +
                                                                                       nBubIdx * p.outerStrideScale);

                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr +
                                                                                    groupIdx * p.innerStrideWeight +
                                                                                    nBubIdx * p.outerStrideWeight));
            }

            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16, weightB16, offset, maskAll);
            }
            MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeight);
        }
        p.weightOutBaseAddr += p.weightOutAddrOffset;
        maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool useVag>
__aicore__ inline void AntiquantW4Pergroup256NK(ParamsGroupSize128And256<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weight1B4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::RegTensor<XType> weight1B16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
    MicroAPI::MaskReg maskWeight1 = MicroAPI::UpdateMask<XType>(p.maskWeight1);

    for (uint16_t groupIdx = 0; groupIdx < p.groupNum; ++groupIdx) {
        for (uint16_t nBubIdx = 0; nBubIdx < p.bubNLen; ++nBubIdx) {
            if constexpr (useVag) {
                MicroAPI::AddrReg aregScale = MicroAPI::CreateAddrReg<XType>(groupIdx, 1, nBubIdx, p.outerStrideScale);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offset, p.offsetBaseAddr, aregScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr, aregScale);

                MicroAPI::AddrReg aregWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    groupIdx, p.innerStrideWeight, nBubIdx, p.outerStrideWeight);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr), aregWeight);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weight1B4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr1), aregWeight);
            } else {
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                        offset, p.offsetBaseAddr + groupIdx + nBubIdx * p.outerStrideScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr + groupIdx +
                                                                                       nBubIdx * p.outerStrideScale);

                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr +
                                                                                    groupIdx * p.innerStrideWeight +
                                                                                    nBubIdx * p.outerStrideWeight));
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weight1B4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr1 +
                                                                                     groupIdx * p.innerStrideWeight +
                                                                                     nBubIdx * p.outerStrideWeight));
            }

            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weight1B16, weight1B4, maskAll);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16, weightB16, offset, maskAll);
                MicroAPI::Add(weight1B16, weight1B16, offset, maskAll);
            }
            MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
            MicroAPI::Mul(weight1B16, weight1B16, scale, maskAll);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeight);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr1, weight1B16, p.dataBlockStride, p.repeatStride, maskWeight1);
        }
        p.weightOutBaseAddr += p.weightOutAddrOffset;
        p.weightOutBaseAddr1 += p.weightOutAddrOffset;
        maskWeight = MicroAPI::UpdateMask<XType>(p.maskWeight);
        maskWeight1 = MicroAPI::UpdateMask<XType>(p.maskWeight1);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool tailInMainGroup, bool tailInTailGroup>
__aicore__ inline void AntiquantW4PergroupGt128NKCase1(ParamsGroupSizeGt128<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskGrpMod128 = MicroAPI::UpdateMask<XType>(p.resGrpMod128);
    MicroAPI::MaskReg maskRemainGrpMod128 = MicroAPI::UpdateMask<XType>(p.resRemainGrpMod128);

    for (uint16_t nBubIdx = 0; nBubIdx < p.bubNLen; ++nBubIdx) {
        for (uint16_t groupIdx = 0; groupIdx < p.mainGroupNum; ++groupIdx) {
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    offset, p.offsetBaseAddr + nBubIdx * p.groupNumBub + groupIdx);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr +
                                                                                   nBubIdx * p.groupNumBub + groupIdx);
            // 处理一个 group_size 中 128 对齐部分
            for (uint16_t vlIdx = 0; vlIdx < p.numVLInGroup; ++vlIdx) {
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                          groupIdx * p.groupSizeInByte +
                                                                          vlIdx * p.vlB4SizeInByte));
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offset, maskAll);
                }
                MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBaseAddr0, weightB16, p.dataBlockStride, p.repeatStride0, maskAll);
            }
            // 处理一个 group_size 相对于 128 的尾块
            if constexpr (tailInMainGroup) {
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                          groupIdx * p.groupSizeInByte +
                                                                          p.numVLInGroup * p.vlB4SizeInByte));
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4,
                                                                                      maskGrpMod128);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offset, maskGrpMod128);
                }
                MicroAPI::Mul(weightB16, weightB16, scale, maskGrpMod128);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBaseAddr0, weightB16, p.dataBlockStride, p.repeatStride1, maskGrpMod128);
            }
        }
        // 处理 bubKLen 相对于 group_size 的尾块
        for (uint16_t tailGroup = 0; tailGroup < p.tailGroupInBubKLen; ++tailGroup) {
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                    offset, p.offsetBaseAddr + nBubIdx * p.groupNumBub + p.mainGroupNum);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(
                scale, p.scaleBaseAddr + nBubIdx * p.groupNumBub + p.mainGroupNum);
            // 处理 bubKLen 相对于 group_size 的尾块中 128 对齐部分
            for (uint16_t vlIdx = 0; vlIdx < p.numVLInRemainGroup; ++vlIdx) {
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                          p.mainGroupNum * p.groupSizeInByte +
                                                                          vlIdx * p.vlB4SizeInByte));
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offset, maskAll);
                }
                MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBaseAddr1, weightB16, p.dataBlockStride, p.repeatStride0, maskAll);
            }
            // 处理一个不完整的 group 相对于 128 的尾块
            if constexpr (tailInTailGroup) {
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4,
                    (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                          p.mainGroupNum * p.groupSizeInByte +
                                                                          p.numVLInRemainGroup * p.vlB4SizeInByte));
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4,
                                                                                      maskRemainGrpMod128);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offset, maskRemainGrpMod128);
                }
                MicroAPI::Mul(weightB16, weightB16, scale, maskRemainGrpMod128);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(p.weightOutBaseAddr1, weightB16,
                                                                            p.dataBlockStride, 0, maskRemainGrpMod128);
            }
        }
        // 校正从 reg 到 ub 的搬出地址，递增 C0 (32 Byte) 大小的地址
        p.weightOutBaseAddr0 = p.oriWeightOutBaseAddr0 + 16 * (nBubIdx + 1);
        p.weightOutBaseAddr1 = p.oriWeightOutBaseAddr1 + 16 * (nBubIdx + 1);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantW4PergroupGt128NKCase2(ParamsGroupSizeGt128<XType, WType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    MicroAPI::RegTensor<XType> offset;
    MicroAPI::RegTensor<XType> scale;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskKLenMod128 = MicroAPI::UpdateMask<XType>(p.tailKLen);

    for (uint16_t nBubIdx = 0; nBubIdx < p.bubNLen; ++nBubIdx) {
        if constexpr (hasAntiquantOffset) {
            // 16 即 BLOCK_CUBE
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offset, p.offsetBaseAddr + nBubIdx * 16);
        }
        MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scale, p.scaleBaseAddr + nBubIdx * 16);
        // 处理 bubKLen 中 128 对齐部分
        for (uint16_t vlIdx = 0; vlIdx < p.numVLInKLen; ++vlIdx) {
            MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightB4,
                (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                      vlIdx * p.vlB4SizeInByte));
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16, weightB16, offset, maskAll);
            }
            MicroAPI::Mul(weightB16, weightB16, scale, maskAll);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr0, weightB16, p.dataBlockStride, p.repeatStride0, maskAll);
        }
        // 处理 bubKLen 相对于 128 的尾块
        for (uint16_t tailVL = 0; tailVL < p.tailVLInKLen; ++tailVL) {
            MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightB4,
                (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBaseAddr + nBubIdx * p.innerExtend +
                                                                      p.numVLInKLen * p.vlB4SizeInByte));
            MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskKLenMod128);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::Add(weightB16, weightB16, offset, maskKLenMod128);
            }
            MicroAPI::Mul(weightB16, weightB16, scale, maskKLenMod128);
            MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr0, weightB16, p.dataBlockStride, p.repeatStride0, maskKLenMod128);
        }
        // 校正从 reg 到 ub 的搬出地址，递增 C0 (32 Byte) 大小的地址
        p.weightOutBaseAddr0 = p.oriWeightOutBaseAddr0 + 16 * (nBubIdx + 1);
    }
}

template <typename XType, typename WType, bool hasAntiquantOffset, bool reusePrevGroup, bool crossBorder, bool useVag>
__aicore__ inline void AntiquantW4Pergroup32OddNK(ParamsGroupSize32OddNK<XType> &p)
{
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    MicroAPI::RegTensor<XType> offsetOdd;
    MicroAPI::RegTensor<XType> scaleOdd;
    MicroAPI::RegTensor<XType> offsetEven;
    MicroAPI::RegTensor<XType> scaleEven;
    MicroAPI::RegTensor<typename RegTensorActualT<WType>::T> weightB4;
    MicroAPI::RegTensor<XType> weightB16;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskWeightBorder = MicroAPI::CreateMask<XType, MicroAPI::MaskPattern::Q>();
    // 生成跨group计算时的mask，低32位为0，高96位为1
    MicroAPI::MaskNot(maskWeightBorder, maskWeightBorder, maskAll);

    uint32_t maskWeightOddValue;
    uint32_t maskWeightEvenValue;
    MicroAPI::MaskReg maskWeightOdd;
    MicroAPI::MaskReg maskWeightEven;

    for (uint16_t groupIdx = 0; groupIdx < p.groupPairNum; ++groupIdx) {
        for (uint16_t nIdx = 0; nIdx < p.bubNLen; ++nIdx) {
            // odd group part
            MicroAPI::AddrReg addRegScale =
                MicroAPI::CreateAddrReg<XType>(groupIdx, p.scaleGroupPairStride, nIdx, p.scaleNStride);
            MicroAPI::AddrReg addRegWeightBorder = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                groupIdx, p.weightGroupPairStride, nIdx, p.weightNStride);
            if constexpr (hasAntiquantOffset) {
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offsetOdd, p.offsetOddAddr, addRegScale);
            }
            MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scaleOdd, p.scaleOddAddr, addRegScale);
            maskWeightOddValue = p.maskWeightOdd;

            for (uint16_t oddIdx = 0; oddIdx < p.oddGroupVLNum; ++oddIdx) {
                MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                    groupIdx, p.weightGroupPairStride, nIdx, p.weightNStride, oddIdx, p.weightVLStride);
                MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInOddAddr), addRegWeight);
                maskWeightOdd = MicroAPI::UpdateMask<XType>(maskWeightOddValue);
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offsetOdd, maskWeightOdd);
                }
                MicroAPI::Mul(weightB16, weightB16, scaleOdd, maskWeightOdd);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutOddAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeightOdd);
            }
            p.weightOutOddAddr -= p.offsetNWeightOutOdd;

            // corss border part
            if constexpr (crossBorder) {
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(offsetEven, p.offsetEvenAddr,
                                                                                addRegScale);
                }
                MicroAPI::DataCopy<XType, MicroAPI::LoadDist::DIST_BRC_B16>(scaleEven, p.scaleEvenAddr, addRegScale);

                if constexpr (!reusePrevGroup) {
                    MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInBorderAddr),
                        addRegWeightBorder);
                }
                MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
                if constexpr (hasAntiquantOffset) {
                    MicroAPI::Add(weightB16, weightB16, offsetEven, maskWeightBorder);
                }
                MicroAPI::Mul(weightB16, weightB16, scaleEven, maskWeightBorder);
                MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    p.weightOutBorderAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeightBorder);

                // even group part
                maskWeightEvenValue = p.maskWeightEven;
                for (uint16_t evenIdx = 0; evenIdx < p.evenGroupVLNum; ++evenIdx) {
                    if constexpr (useVag) {
                        MicroAPI::AddrReg addRegWeight = MicroAPI::CreateAddrReg<typename RegTensorActualT<WType>::T>(
                            groupIdx, p.weightGroupPairStride, nIdx, p.weightNStride, evenIdx, p.weightVLStride);
                        MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            weightB4, (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInEvenAddr),
                            addRegWeight);
                    } else {
                        MicroAPI::DataCopy<typename RegTensorActualT<WType>::T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                            weightB4,
                            (__local_mem__ typename RegTensorActualT<WType>::T *)(p.weightInEvenAddr +
                                                                                  groupIdx * p.weightGroupPairStride +
                                                                                  nIdx * p.weightNStride));
                    }

                    maskWeightEven = MicroAPI::UpdateMask<XType>(maskWeightEvenValue);
                    MicroAPI::Cast<XType, typename RegTensorActualT<WType>::T, castTrait>(weightB16, weightB4, maskAll);
                    if constexpr (hasAntiquantOffset) {
                        MicroAPI::Add(weightB16, weightB16, offsetEven, maskWeightEven);
                    }
                    MicroAPI::Mul(weightB16, weightB16, scaleEven, maskWeightEven);
                    MicroAPI::DataCopy<XType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                                       MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        p.weightOutEvenAddr, weightB16, p.dataBlockStride, p.repeatStride, maskWeightEven);
                }
                p.weightOutBorderAddr -= p.offsetNWeightOutBorder;
                p.weightOutEvenAddr -= p.offsetNWeightOutEven;
            }
        }
        p.weightOutOddAddr += p.offsetKWeightOut;
        if constexpr (crossBorder) {
            p.weightOutBorderAddr += p.offsetKWeightOut;
            p.weightOutEvenAddr += p.offsetKWeightOut;
        }
    }
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35

#endif // WEIGHT_QUANT_BATCHMATMUL_V2_VF_H
