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
 * \file basic_block_vf_nz.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_NZ_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_NZ_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "weight_quant_tool.h"

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename antiQuantScaleType>
struct Int4NzParams {
    uint64_t loopN1;
    uint64_t loopGroupNum;
    uint64_t loopInnerNum;
    uint64_t innerDstStride;
    uint64_t groupDstStride;
    uint64_t loopN1DstStride;
    uint64_t antiQuantGroupSize;
    __local_mem__ antiQuantScaleType *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr;
    __local_mem__ wType *weightLowBitPhyAddr;
    __local_mem__ xType *weightHighBitPhyAddr;
    __local_mem__ uint8_t *antiQuantScaleMaskPhyAddr;
};

static constexpr MicroAPI::CastTrait CAST_S4_TO_F16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait CAST_F16_TO_S8_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::CAST_RINT};

static constexpr int64_t ONE_BLK_ELEM_B16 = ONE_BLK_SIZE / sizeof(half);

template <typename xType, typename wType, typename antiQuantScaleType, bool hasAntiQuantOffset,
          uint64_t ubMte2InnerSize, bool useVag>
__aicore__ inline void AntiQuantInt4NzKnVf(Int4NzParams<xType, wType, antiQuantScaleType> &int4NzParams)
{
    RegTensor<antiQuantScaleType> antiQuantScaleVreg;
    RegTensor<xType> antiQuantOffsetVreg;
    RegTensor<int4x2_t> weightS4Vreg;
    RegTensor<xType> weightF16Vreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t LoopN1Idx = 0; LoopN1Idx < int4NzParams.loopN1; LoopN1Idx++) {
        // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(
            antiQuantScaleVreg, int4NzParams.antiQuantScaleBasePhyAddr + LoopN1Idx * BLOCK_CUBE);
        if constexpr (hasAntiQuantOffset) {
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(
                antiQuantOffsetVreg, int4NzParams.antiQuantOffsetBasePhyAddr + LoopN1Idx * BLOCK_CUBE);
        }

        for (uint16_t LoopInnerNumIdx = 0; LoopInnerNumIdx < int4NzParams.loopInnerNum; LoopInnerNumIdx++) {
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg, (__local_mem__ int4x2_t *)(int4NzParams.weightLowBitPhyAddr +
                                                         LoopN1Idx * (BLOCK_CUBE >> 1) * ubMte2InnerSize +
                                                         LoopInnerNumIdx * (VECTOR_REG_WIDTH >> 2)));
            // PART_P0 表示按照如下形式处理做cast：
            // Vn 0 1 x x x x x x 2 3 x x x x x x
            // Vd 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
            MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(weightF16Vreg, weightS4Vreg, maskAll);
            if constexpr (hasAntiQuantOffset) {
                MicroAPI::Add(weightF16Vreg, weightF16Vreg, antiQuantOffsetVreg, maskAll);
            }
            MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

            if constexpr (useVag) {
                AddrReg weightF16PhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                    LoopN1Idx, int4NzParams.loopN1DstStride, LoopInnerNumIdx, int4NzParams.innerDstStride);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                    int4NzParams.weightHighBitPhyAddr, weightF16Vreg, weightF16PhyAddrReg, maskAll);
            } else {
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                    int4NzParams.weightHighBitPhyAddr + LoopN1Idx * int4NzParams.loopN1DstStride +
                        LoopInnerNumIdx * int4NzParams.innerDstStride,
                    weightF16Vreg, maskAll);
            }
        }
    }
}

template <typename xType, typename wType, typename antiQuantScaleType, bool hasAntiQuantOffset,
          uint64_t ubMte2InnerSize>
__aicore__ inline void AntiQuantS8S4NzKnGroupVf(Int4NzParams<xType, wType, antiQuantScaleType> &int4NzParams)
{
    RegTensor<antiQuantScaleType> antiQuantScaleVreg;
    RegTensor<antiQuantScaleType> antiQuantScaleVreg1;
    RegTensor<int4x2_t> weightS4Vreg;
    RegTensor<half> weightF16Vreg;
    RegTensor<xType> weightS8Vreg;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskSelect = MicroAPI::CreateMask<uint8_t>();
    MicroAPI::DataCopy(maskSelect, int4NzParams.antiQuantScaleMaskPhyAddr);
    __local_mem__ antiQuantScaleType *antiQuantScaleBasePhyAddr;

    for (uint16_t loopN1Idx = 0; loopN1Idx < int4NzParams.loopN1; loopN1Idx++) {
        for (uint16_t loopGroupIdx = 0; loopGroupIdx < int4NzParams.loopGroupNum; loopGroupIdx++) {
            // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
            antiQuantScaleBasePhyAddr =
                int4NzParams.antiQuantScaleBasePhyAddr + loopN1Idx * C0_SIZE_B8 + loopGroupIdx * VEC_MAX_ELEM_B16;
            MicroAPI::DataCopy<antiQuantScaleType, MicroAPI::LoadDist::DIST_BLK>(antiQuantScaleVreg,
                                                                                 antiQuantScaleBasePhyAddr);
            MicroAPI::DataCopy<antiQuantScaleType, MicroAPI::LoadDist::DIST_BLK>(
                antiQuantScaleVreg1, antiQuantScaleBasePhyAddr + ONE_BLK_ELEM_B16);
            MicroAPI::Select(antiQuantScaleVreg, antiQuantScaleVreg, antiQuantScaleVreg1, maskSelect);
            for (uint16_t loopGroupInnerIdx = 0; loopGroupInnerIdx < int4NzParams.loopInnerNum; loopGroupInnerIdx++) {
                // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                // Vd 0 1 x x x x x x 2 3 x x x x x x
                // 地址偏移以B记，对C0_SIZE_B8和VEC_MAX_ELEM_B16除以2实现正确偏移
                MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightS4Vreg,
                    (__local_mem__ int4x2_t *)(int4NzParams.weightLowBitPhyAddr +
                                               loopN1Idx * (C0_SIZE_B8 >> 1) * ubMte2InnerSize +
                                               loopGroupIdx * int4NzParams.antiQuantGroupSize * (C0_SIZE_B8 >> 1) +
                                               loopGroupInnerIdx * (VEC_MAX_ELEM_B16 >> 1)));
                // S4_TO_F16按如下模式cast
                // Vn 00000012 00000034
                // Vd 3c004000 42004400
                MicroAPI::Cast<half, int4x2_t, CAST_S4_TO_F16_TRAIT>(weightF16Vreg, weightS4Vreg, maskAll);
                MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);
                // F16_TO_S8按如下模式cast
                // Vn 3c004000 42004400
                // Vd 00010002 00030004
                MicroAPI::Cast<xType, half, CAST_F16_TO_S8_TRAIT>(weightS8Vreg, weightF16Vreg, maskAll);

                AddrReg weightHighBitPhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                    loopN1Idx, int4NzParams.loopN1DstStride,
                    loopGroupIdx, int4NzParams.groupDstStride,
                    loopGroupInnerIdx, int4NzParams.innerDstStride);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_PACK_B16>(
                    int4NzParams.weightHighBitPhyAddr, weightS8Vreg, weightHighBitPhyAddrReg, maskAll);
            }
        }
    }
}
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_VF_NZ_H