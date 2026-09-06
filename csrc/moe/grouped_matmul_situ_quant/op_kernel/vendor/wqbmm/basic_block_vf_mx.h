/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file basic_block_vf_mx.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType>
struct MxFp4NdScaleParams {
    uint64_t ubLoopExternalAxis;
    __local_mem__ uint8_t *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantScaleF16PhyAddr0;
    __local_mem__ xType *antiQuantScaleF16PhyAddr1;
};

template <typename xType, typename wType>
struct MxFp4NdWeightParams {
    uint64_t ubLoopExternalAxis;
    __local_mem__ wType *weightLowBitPhyAddr0;
    __local_mem__ wType *weightLowBitPhyAddr1;
    __local_mem__ xType *weightF16PhyAddr0;
    __local_mem__ xType *weightF16PhyAddr1;
    __local_mem__ xType *antiQuantScaleF16PhyAddr0;
    __local_mem__ xType *antiQuantScaleF16PhyAddr1;
};

template <typename xType, typename wType>
struct Fp4NzParams {
    uint64_t loopN1;
    uint64_t loopGroupNum;
    uint64_t loopInnerNum;
    uint64_t innerDstStride;
    uint64_t groupDstStride;
    uint64_t loopN1DstStride;
    __local_mem__ xType *antiQuantScaleBasePhyAddr;
    __local_mem__ wType *weightLowBitPhyAddr;
    __local_mem__ xType *weightHighBitPhyAddr;
};

template <typename xType, typename wType, typename biasType>
struct MxA8W4NzParams {
    uint64_t loopKNum;
    uint64_t innerLoopNum;
    uint64_t loopKDstStride;
    uint64_t innerDstStride;
    uint64_t biasLoopNum;
    uint64_t nRealSizeAlign;
    __ubuf__ wType *weightLowBitPhyAddr;
    __ubuf__ xType *weightHighBitPhyAddr;
    __ubuf__ biasType *biasInUbAddr;
    __ubuf__ biasType *biasOutUbAddr;
};

static constexpr MicroAPI::CastTrait CAST_BF16_TO_FP16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING,
                                                                AscendC::RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait CAST_FP4_TO_BF16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::UNKNOWN};
static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
static constexpr uint32_t E2M1_AND_MASK = 0x9C;

template <typename xType>
__aicore__ inline void MxScaleVf(RegTensor<uint8_t> &antiQuantScaleE8m0Vreg0,
                                 RegTensor<uint8_t> &antiQuantScaleE8m0Vreg1, RegTensor<xType> &antiQuantScaleF16Vreg0,
                                 RegTensor<xType> &antiQuantScaleF16Vreg1, MicroAPI::MaskReg &maskAll)
{
    RegTensor<uint8_t> zeroVreg;
    MicroAPI::Duplicate(zeroVreg, 0);

    // 通过数据重排指令， 交织 antiQuantScaleE8m0Vreg0 和 zeroVreg , Interleave后变为
    // antiQuantScaleE8m0Vreg0
    // Vn s1 0 s2 0 s3 0 s4 0 s5 0 s6 0 s7 0 s8 0....... s127 0 s128 0
    // antiQuantScaleE8m0Vreg1
    // Vd s128 0 s129 0 s130 0 s131 0 s132 0 s133 0....... s255 0 s256 0
    MicroAPI::Interleave(antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, zeroVreg, antiQuantScaleE8m0Vreg0);

    if constexpr (!IsSameType<typename MicroAPI::TypeGet<xType>::T, vector_f16>::value) {
        // 逻辑右移一位,得到BF16{S1E8M7}的排列,比如s1 0 =[8bit, 8bit]= [10101011 00000000]
        // 变为 S1 = [16bit] = [0101010110000000]
        // antiQuantScaleF16Vreg0：
        // S1 S2 S3 S4 S5 S6 S7 S8 ..... S127 S128
        // antiQuantScaleF16Vreg1:
        // S128 S129 S130 S131 S132 S133 S134 S135 ..... S255 S256
        MicroAPI::ShiftRights((MicroAPI::RegTensor<uint16_t> &)antiQuantScaleF16Vreg0,
                              (MicroAPI::RegTensor<uint16_t> &)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
        MicroAPI::ShiftRights((MicroAPI::RegTensor<uint16_t> &)antiQuantScaleF16Vreg1,
                              (MicroAPI::RegTensor<uint16_t> &)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
    } else {
        // 需要转换为FP16格式时，先转换为BF16再转换为FP16
        RegTensor<uint16_t> antiQuantScaleBF16Vreg0;
        RegTensor<uint16_t> antiQuantScaleBF16Vreg1;
        MicroAPI::ShiftRights((MicroAPI::RegTensor<uint16_t> &)antiQuantScaleBF16Vreg0,
                              (MicroAPI::RegTensor<uint16_t> &)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
        MicroAPI::ShiftRights((MicroAPI::RegTensor<uint16_t> &)antiQuantScaleBF16Vreg1,
                              (MicroAPI::RegTensor<uint16_t> &)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
        MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
            antiQuantScaleF16Vreg0, (MicroAPI::RegTensor<bfloat16_t> &)antiQuantScaleBF16Vreg0, maskAll);
        MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
            antiQuantScaleF16Vreg1, (MicroAPI::RegTensor<bfloat16_t> &)antiQuantScaleBF16Vreg1, maskAll);
    }
}

template <typename xType, typename wType>
__aicore__ inline void CastFp4ToF16(RegTensor<xType> &f16Vreg, RegTensor<wType> &fp4Vreg, MicroAPI::MaskReg &maskAll)
{
    if constexpr (!IsSameType<typename MicroAPI::TypeGet<xType>::T, vector_f16>::value) {
        // CAST_FP4_TO_BF16_TRAIT中设置RegLayout::ZERO 表示按照如下形式处理做cast：
        // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
        // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
        MicroAPI::Cast<xType, wType, CAST_FP4_TO_BF16_TRAIT>(f16Vreg, fp4Vreg, maskAll);
    } else {
        // FP4-->FP16指令不支持，需要转换为BF16再转换为FP16
        RegTensor<bfloat16_t> bF16Vreg;
        MicroAPI::Cast<bfloat16_t, wType, CAST_FP4_TO_BF16_TRAIT>(bF16Vreg, fp4Vreg, maskAll);
        MicroAPI::Cast<xType, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(f16Vreg, bF16Vreg, maskAll);
    }
}

template <typename xType>
__aicore__ inline void MxNkScaleVf(MxFp4NdScaleParams<xType> &mxFp4NdNkScaleParams)
{
    // NK mte2搬运的antiquantscale的标准大小为(128, 32), 按照4行的粒度处理，每次处理128个E8M0, 得到256个F16的数字
    RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
    RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
    RegTensor<xType> antiQuantScaleF16Vreg0;
    RegTensor<xType> antiQuantScaleF16Vreg1;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < mxFp4NdNkScaleParams.ubLoopExternalAxis; ubLoopNIdx++) {
        // 搬运128个E8M0的antiquantscale, 通过两倍上采样变成256个E8M0， DIST_US_B8表示搬运模式如下：
        // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s125 s126 s127 s128
        // Vd s1 s1 s2 s2 s3 s3 s4 s4 s5 ...... s125 s125 s126 s126 s127 s127 s128 s128
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
            antiQuantScaleE8m0Vreg0, mxFp4NdNkScaleParams.antiQuantScaleBasePhyAddr + ubLoopNIdx * 128);

        MxScaleVf(antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, antiQuantScaleF16Vreg0, antiQuantScaleF16Vreg1,
                  maskAll);

        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
            mxFp4NdNkScaleParams.antiQuantScaleF16PhyAddr0 + ubLoopNIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0,
            maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
            mxFp4NdNkScaleParams.antiQuantScaleF16PhyAddr1 + ubLoopNIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1,
            maskAll);
    }
}

template <typename xType>
__aicore__ inline void MxKnScaleVf(MxFp4NdScaleParams<xType> &mxFp4NdScaleParams)
{
    // KN mte2搬运的antiquantscale的标准大小为(4，256), 按照一行的粒度处理
    RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
    RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
    RegTensor<xType> antiQuantScaleF16Vreg0;
    RegTensor<xType> antiQuantScaleF16Vreg1;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < mxFp4NdScaleParams.ubLoopExternalAxis; ubLoopKIdx++) {
        // 搬运256个E8M0的antiquantscale, DIST_NORM表示搬运模式如下：
        // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
        // Vd s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
        MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantScaleE8m0Vreg0, mxFp4NdScaleParams.antiQuantScaleBasePhyAddr + ubLoopKIdx * VECTOR_REG_WIDTH);

        MxScaleVf(antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, antiQuantScaleF16Vreg0, antiQuantScaleF16Vreg1,
                  maskAll);

        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
            mxFp4NdScaleParams.antiQuantScaleF16PhyAddr0 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0,
            maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
            mxFp4NdScaleParams.antiQuantScaleF16PhyAddr1 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1,
            maskAll);
    }
}

template <typename xType, typename wType, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void MxNkWeightVf(MxFp4NdWeightParams<xType, wType> &mxFp4NdNkWeightParams)
{
    // 每次处理一行， 一行为256个FP4的数， 分两条指令处理，每条指令处理128个FP4的数
    RegTensor<xType> antiQuantScaleF16Vreg0;
    RegTensor<xType> antiQuantScaleF16Vreg1;
    RegTensor<wType> weightFp4Vreg0;
    RegTensor<wType> weightFp4Vreg1;
    RegTensor<xType> weightF16Vreg0;
    RegTensor<xType> weightF16Vreg1;

    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < mxFp4NdNkWeightParams.ubLoopExternalAxis; ubLoopNIdx++) {
        // DIST_E2B_B16 表示搬运模式如下, 将一个f16的数扩展成16个
        // Vn  1 2 3 4 5 6 7 8
        // Vd
        // 11111111111111112222222222222233333333333333334444444444444444455555555555555666666666666666677777777777777
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_E2B_B16>(
            antiQuantScaleF16Vreg0,
            mxFp4NdNkWeightParams.antiQuantScaleF16PhyAddr0 + ubLoopNIdx * (VECTOR_REG_WIDTH >> 2));
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_E2B_B16>(
            antiQuantScaleF16Vreg1,
            mxFp4NdNkWeightParams.antiQuantScaleF16PhyAddr1 + ubLoopNIdx * (VECTOR_REG_WIDTH >> 2));

        // DIST_UNPK_B8 表示按照如下形式载入, 其中Vn中一个数字为4bit:
        // Vn  1 2 3 4 5 6 7 8 9 a b c d e f g .....
        // Vd  1 2 x x x x x x 3 4 x x x x x x .....
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            weightFp4Vreg0, (__local_mem__ wType *)(mxFp4NdNkWeightParams.weightLowBitPhyAddr0 +
                                                    ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            weightFp4Vreg1, (__local_mem__ wType *)(mxFp4NdNkWeightParams.weightLowBitPhyAddr1 +
                                                    ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

        CastFp4ToF16<xType, wType>(weightF16Vreg0, weightFp4Vreg0, maskAll);
        CastFp4ToF16<xType, wType>(weightF16Vreg1, weightFp4Vreg1, maskAll);
        MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleF16Vreg0, maskAll);
        MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleF16Vreg1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            mxFp4NdNkWeightParams.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            mxFp4NdNkWeightParams.weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

template <typename xType, typename wType, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void MxKnWeightVf(MxFp4NdWeightParams<xType, wType> &mxFp4NdKnWeightParams)
{
    // 每次处理一行， 一行为256个FP4的数， 分两条指令处理，每条指令处理128个FP4的数
    RegTensor<xType> antiQuantScaleF16Vreg0;
    RegTensor<xType> antiQuantScaleF16Vreg1;
    RegTensor<wType> weightFp4Vreg0;
    RegTensor<wType> weightFp4Vreg1;
    RegTensor<xType> weightF16Vreg0;
    RegTensor<xType> weightF16Vreg1;

    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < mxFp4NdKnWeightParams.ubLoopExternalAxis; ubLoopKIdx++) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantScaleF16Vreg0, mxFp4NdKnWeightParams.antiQuantScaleF16PhyAddr0 + ubLoopKIdx * VECTOR_REG_WIDTH);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantScaleF16Vreg1, mxFp4NdKnWeightParams.antiQuantScaleF16PhyAddr1 + ubLoopKIdx * VECTOR_REG_WIDTH);

        for (uint16_t groupIdx = 0; groupIdx < MX_GROUPSIZE; groupIdx++) {
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightFp4Vreg0, (__local_mem__ wType *)(mxFp4NdKnWeightParams.weightLowBitPhyAddr0 +
                                                        ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1) * MX_GROUPSIZE +
                                                        groupIdx * (vecConfig.ubMte2InnerSize >> 1)));

            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightFp4Vreg1, (__local_mem__ wType *)(mxFp4NdKnWeightParams.weightLowBitPhyAddr1 +
                                                        ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1) * MX_GROUPSIZE +
                                                        groupIdx * (vecConfig.ubMte2InnerSize >> 1)));

            CastFp4ToF16<xType, wType>(weightF16Vreg0, weightFp4Vreg0, maskAll);
            CastFp4ToF16<xType, wType>(weightF16Vreg1, weightFp4Vreg1, maskAll);
            MicroAPI::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleF16Vreg0, maskAll);
            MicroAPI::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleF16Vreg1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxFp4NdKnWeightParams.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxFp4NdKnWeightParams.weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        }
    }
}

template <typename xType, typename wType, uint64_t ubMte2KSize>
__aicore__ inline void AntiQuantFp4NzKnVf(Fp4NzParams<xType, wType> &fp4NzParams)
{
    RegTensor<xType> antiQuantScaleVreg;
    RegTensor<wType> weightFp4Vreg;
    RegTensor<xType> weightF16Vreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    __local_mem__ xType *antiQuantScaleBasePhyAddr;

    for (uint16_t loopN1Idx = 0; loopN1Idx < fp4NzParams.loopN1; loopN1Idx++) {
        for (uint16_t loopGroupIdx = 0; loopGroupIdx < fp4NzParams.loopGroupNum; loopGroupIdx++) {
            // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
            antiQuantScaleBasePhyAddr = fp4NzParams.antiQuantScaleBasePhyAddr + loopN1Idx * BLOCK_CUBE +
                                        loopGroupIdx * 128;  // 当前方案一次对齐到128
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);

            for (uint16_t loopGroupInnerIdx = 0; loopGroupInnerIdx < fp4NzParams.loopInnerNum; loopGroupInnerIdx++) {
                // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                // Vd 0 1 x x x x x x 2 3 x x x x x x
                // 4bit物理地址位移 = 逻辑索引 >> 1
                AddrReg weightFp4AddrReg = MicroAPI::CreateAddrReg<wType>(loopN1Idx, BLOCK_CUBE * ubMte2KSize >> 1,
                                                                          loopGroupIdx, MX_GROUPSIZE * BLOCK_CUBE >> 1,
                                                                          loopGroupInnerIdx, VEC_MAX_ELEM_B16 >> 1);
                MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightFp4Vreg, (__local_mem__ wType *)fp4NzParams.weightLowBitPhyAddr, weightFp4AddrReg);

                CastFp4ToF16<xType, wType>(weightF16Vreg, weightFp4Vreg, maskAll);
                MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

                AddrReg weightHighBitPhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                    loopN1Idx, fp4NzParams.loopN1DstStride, loopGroupIdx, fp4NzParams.groupDstStride, loopGroupInnerIdx,
                    fp4NzParams.innerDstStride);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                    fp4NzParams.weightHighBitPhyAddr, weightF16Vreg, weightHighBitPhyAddrReg, maskAll);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, bool calcMxBias, bool isBiasSingleVector>
__simd_callee__ inline void MxA8W4BiasCompute(MxA8W4NzParams<xType, wType, biasType> &mxA8W4NzParams)
{
    if constexpr (calcMxBias) {
        static constexpr biasType MX_BIAS_FACTOR = static_cast<biasType>(0.015625f);
        MicroAPI::RegTensor<biasType> biasVreg, biasFactorVreg;
        MicroAPI::MaskReg maskBiasAll = MicroAPI::CreateMask<biasType, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasFactorVreg, MX_BIAS_FACTOR,
                                                                                 maskBiasAll);
        if constexpr (isBiasSingleVector) {
            for (uint16_t loopBiasIdx = 0; loopBiasIdx < mxA8W4NzParams.biasLoopNum; ++loopBiasIdx) {
                MicroAPI::AddrReg biasAreg = MicroAPI::CreateAddrReg<biasType>(loopBiasIdx, VEC_MAX_ELEM_B16);
                MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, mxA8W4NzParams.biasInUbAddr,
                                                                             biasAreg);
                MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                                   maskBiasAll);
                MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(mxA8W4NzParams.biasOutUbAddr,
                                                                                   biasVreg, biasAreg, maskBiasAll);
            }
        } else {
            MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, mxA8W4NzParams.biasInUbAddr);
            MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                               maskBiasAll);
            MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(mxA8W4NzParams.biasOutUbAddr, biasVreg,
                                                                               maskBiasAll);
        }
    }
}

template <typename xType, typename wType, typename biasType, bool calcMxBias, bool isBiasSingleVector>
__simd_vf__ inline void AntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    MxA8W4BiasCompute<xType, wType, biasType, calcMxBias, isBiasSingleVector>(mxA8W4NzParams);
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            // DIST_US_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7
            // Vd 0 1 0 1 2 3 2 3 4 5 4 5 6 7 6 7
            // 4bit物理地址位移 = 逻辑索引 >> 1
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t> &)wLoad, (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
            MicroAPI::And(wAnd, wSel, wAndReg, preg);

            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride, innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (MicroAPI::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

template <uint16_t groupNumDefaultLen, uint16_t dstInnerStride>
__simd_vf__ inline void TransMxScaleNdToDnVf(__ubuf__ uint16_t *scaleInAddr, __ubuf__ uint16_t *scaleOutAddr,
                                             __ubuf__ uint16_t *transIdAddr, uint16_t nCount, uint16_t groupNumCount,
                                             uint16_t nStride)
{
    MicroAPI::RegTensor<uint16_t> scaleVreg;
    MicroAPI::RegTensor<uint16_t> ubTransIdVreg;
    MicroAPI::LoadAlign(ubTransIdVreg, transIdAddr); // 0, 127,256,...1,128,...
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        for (uint16_t groupNumIdx = 0; groupNumIdx < groupNumCount; groupNumIdx++) {
            MicroAPI::Gather(scaleVreg, scaleInAddr + nIdx * BLOCK_CUBE * groupNumDefaultLen + groupNumIdx * 8,
                             ubTransIdVreg, maskAll);

            MicroAPI::AddrReg aregScaleOut =
                MicroAPI::CreateAddrReg<uint16_t>(nIdx, nStride, groupNumIdx, dstInnerStride);
            MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM_B16>(scaleOutAddr, scaleVreg, aregScaleOut,
                                                                               maskAll);
        }
    }
}

} // namespace WeightQuantBatchMatmulV2::Arch35
#endif // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H
