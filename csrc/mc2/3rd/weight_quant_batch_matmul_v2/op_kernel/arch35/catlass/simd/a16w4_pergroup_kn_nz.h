/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_SIMD_A16W4_PERGROUP_KN_NZ_H
#define ARCH35_CATLASS_SIMD_A16W4_PERGROUP_KN_NZ_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"

using AscendC::IsSameType;
using AscendC::VECTOR_REG_WIDTH;
namespace MicroAPI = AscendC::MicroAPI;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

template <typename T>
struct ONE_REG_ELEM {
    static constexpr uint32_t value = VECTOR_REG_WIDTH / sizeof(T);
};

template <typename T>
struct ONE_BLK_ELEM {
    // block is 32Byte
    static constexpr uint32_t value = 32 / sizeof(T);
};

template <typename T>
struct C0 {
    // block is 32Byte
    static constexpr uint32_t value = 32 / sizeof(T);
};

static constexpr MicroAPI::CastTrait CAST_S4_TO_F16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING,
                                                             AscendC::RoundMode::UNKNOWN};

template <typename DtypeDst>
struct Regs {
    MicroAPI::RegTensor<DtypeDst> wNzF16;
    MicroAPI::RegTensor<DtypeDst> scale;
    MicroAPI::RegTensor<DtypeDst> offset;
    MicroAPI::RegTensor<int4x2_t> wNzS4;
};

struct VfParams {
    uint16_t n1;
    uint16_t mainGroupNum;
    uint16_t regNumInMainGroup;
    uint16_t regNumInTailGroup;
    uint32_t n1SrcExtend;
    uint32_t groupNumSrcExtend;
    uint32_t innerSrcExtend;
    uint32_t n1DstExtend;
    uint32_t groupNumDstExtend;
    uint32_t innerDstExtend;
    uint64_t n;
};

template <typename DtypeDst>
struct MainAddr {
    __local_mem__ DtypeDst *scaleBaseAddr;
    __local_mem__ DtypeDst *offsetBaseAddr;
    __local_mem__ int8_t *weightInUbBaseAddr;
    __local_mem__ DtypeDst *weightOutUbBaseAddr;
};

template <typename DtypeDst>
struct TailAddr {
    __local_mem__ DtypeDst *tailScaleBaseAddr;
    __local_mem__ DtypeDst *tailOffsetBaseAddr;
    __local_mem__ int8_t *tailWeightInUbBaseAddr;
    __local_mem__ DtypeDst *tailWeightOutUbBaseAddr;
};

template <typename DtypeDst, bool hasAntiQuantOffset>
DEVICE void Dequant(MicroAPI::RegTensor<DtypeDst> &wNzF16, MicroAPI::RegTensor<DtypeDst> &scale,
                    MicroAPI::RegTensor<DtypeDst> &offset, MicroAPI::RegTensor<int4x2_t> &wNzS4,
                    MicroAPI::MaskReg &preg)
{
    MicroAPI::Cast<DtypeDst, int4x2_t, CAST_S4_TO_F16_TRAIT>(wNzF16, wNzS4, preg);
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::Add(wNzF16, wNzF16, offset, preg);
    }
    MicroAPI::Mul(wNzF16, wNzF16, scale, preg);
}

template <typename DtypeDst, bool hasAntiQuantOffset>
DEVICE void ProcessMainGroup(uint16_t n1Idx, Regs<DtypeDst> &reg, const VfParams &param, MainAddr<DtypeDst> &addr,
                             MicroAPI::MaskReg &preg)
{
    for (uint16_t groupIdx = 0; groupIdx < (uint16_t)param.mainGroupNum; ++groupIdx) {
        MicroAPI::AddrReg aregScale =
            MicroAPI::CreateAddrReg<DtypeDst>(n1Idx, ONE_BLK_ELEM<DtypeDst>::value, groupIdx, param.n);
        // 每次处理 128 个数, scale broadcast 为 128 个数 (256B)
        MicroAPI::DataCopy<DtypeDst, MicroAPI::LoadDist::DIST_BLK>(reg.scale, addr.scaleBaseAddr, aregScale);
        if constexpr (hasAntiQuantOffset) {
            MicroAPI::DataCopy<DtypeDst, MicroAPI::LoadDist::DIST_BLK>(reg.offset, addr.offsetBaseAddr, aregScale);
        }
        for (uint16_t regIdx = 0; regIdx < (uint16_t)param.regNumInMainGroup; ++regIdx) { // 按 128 个数迭代
            // UNPK4_B8 表示按照如下形式载入：
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g
            // Vd 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            MicroAPI::AddrReg aregWeightIn = MicroAPI::CreateAddrReg<uint8_t>(
                n1Idx, param.n1SrcExtend, groupIdx, param.groupNumSrcExtend, regIdx, param.innerSrcExtend);
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                reg.wNzS4, (__local_mem__ int4x2_t *)(addr.weightInUbBaseAddr), aregWeightIn);

            Dequant<DtypeDst, hasAntiQuantOffset>(reg.wNzF16, reg.scale, reg.offset, reg.wNzS4, preg);

            MicroAPI::AddrReg aregWeightOut = MicroAPI::CreateAddrReg<DtypeDst>(
                n1Idx, param.n1DstExtend, groupIdx, param.groupNumDstExtend, regIdx, param.innerDstExtend);
            MicroAPI::DataCopy<DtypeDst, MicroAPI::StoreDist::DIST_NORM_B16>(addr.weightOutUbBaseAddr, reg.wNzF16,
                                                                             aregWeightOut, preg);
        }
    }
}

template <typename DtypeDst, bool hasAntiQuantOffset>
DEVICE void ProcessTailGroup(uint16_t n1Idx, Regs<DtypeDst> &reg, const VfParams &param, TailAddr<DtypeDst> &addr,
                             MicroAPI::MaskReg &preg)
{
    MicroAPI::DataCopy<DtypeDst, MicroAPI::LoadDist::DIST_BLK>(reg.scale, addr.tailScaleBaseAddr +
                                                                              n1Idx * ONE_BLK_ELEM<DtypeDst>::value);
    if constexpr (hasAntiQuantOffset) {
        MicroAPI::DataCopy<DtypeDst, MicroAPI::LoadDist::DIST_BLK>(
            reg.offset, addr.tailOffsetBaseAddr + n1Idx * ONE_BLK_ELEM<DtypeDst>::value);
    }
    for (uint16_t regIdx = 0; regIdx < (uint16_t)param.regNumInTailGroup; ++regIdx) { // 按 128 个数迭代
        MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            reg.wNzS4, (__local_mem__ int4x2_t *)(addr.tailWeightInUbBaseAddr + n1Idx * param.n1SrcExtend +
                                                  regIdx * param.innerSrcExtend));
        Dequant<DtypeDst, hasAntiQuantOffset>(reg.wNzF16, reg.scale, reg.offset, reg.wNzS4, preg);
        MicroAPI::DataCopy<DtypeDst, MicroAPI::StoreDist::DIST_NORM_B16>(
            addr.tailWeightOutUbBaseAddr + n1Idx * param.n1DstExtend + regIdx * param.innerDstExtend, reg.wNzF16, preg);
    }
}

template <typename DtypeDst, bool hasAntiQuantOffset>
DEVICE void AntiQuantComputeKNGroupWeightNzMultiGroupWithoutTail(const VfParams &param, MainAddr<DtypeDst> &mainAddr)
{
    __VEC_SCOPE__
    {
        // (n1,    k1, k0, n0)
        // n1, gn * regNumInMainGroup / 2, 16, 16
        MicroAPI::RegTensor<DtypeDst> wNzF16, scale, offset;
        MicroAPI::RegTensor<int4x2_t> wNzS4;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<DtypeDst, MicroAPI::MaskPattern::ALL>();
        Regs<DtypeDst> reg{wNzF16, scale, offset, wNzS4};
        // 对一个 kBubSize 中 group 的个数迭代
        for (uint16_t n1Idx = 0; n1Idx < (uint16_t)param.n1; ++n1Idx) { // 对 nBub1 迭代 NOTE 8
            // 对一个 kBubSize 中 group 的个数迭代
            ProcessMainGroup<DtypeDst, hasAntiQuantOffset>(n1Idx, reg, param, mainAddr, preg);
        }
    }
}

template <typename DtypeDst, bool hasAntiQuantOffset>
DEVICE void AntiQuantComputeKNGroupWeightNzMultiGroupWithTail(const VfParams &param, MainAddr<DtypeDst> &mainAddr,
                                                              TailAddr<DtypeDst> &tailAddr)
{
    __VEC_SCOPE__
    {
        // (n1,    k1, k0, n0)
        // n1, gn * regNumInMainGroup / 2, 16, 16
        MicroAPI::RegTensor<DtypeDst> wNzF16, scale, offset;
        MicroAPI::RegTensor<int4x2_t> wNzS4;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<DtypeDst, MicroAPI::MaskPattern::ALL>();
        Regs<DtypeDst> reg{wNzF16, scale, offset, wNzS4};
        // 对一个 kBubSize 中 group 的个数迭代
        for (uint16_t n1Idx = 0; n1Idx < (uint16_t)param.n1; ++n1Idx) { // 对 nBub1 迭代 NOTE 8
            // 对一个 kBubSize 中 group 的个数迭代
            ProcessMainGroup<DtypeDst, hasAntiQuantOffset>(n1Idx, reg, param, mainAddr, preg);
            ProcessTailGroup<DtypeDst, hasAntiQuantOffset>(n1Idx, reg, param, tailAddr, preg);
        }
    }
}

template <typename DtypeDst, bool hasAntiQuantOffset, uint64_t kUb, uint32_t BufferNum>
DEVICE void AntiQuantComputeKNGroupWeightNz(LocalTensor<int8_t> const &weightIn, LocalTensor<DtypeDst> const &scale,
                                            LocalTensor<DtypeDst> const &offset, LocalTensor<DtypeDst> const &weightOut,
                                            uint64_t tileSizeN, uint64_t tileSizeK, uint64_t n, uint64_t groupSize)
{
    // 存储 (n1, kUb/k0, k0, n0)
    // 实际 (n1, kTile/k0, k0, n0)

    // 处理一块 (kBub, nBub), 数据格式为 (nBub1, kBub1, kBub0, nBub0)
    // 每次处理 128 个数，按照 128 个数 (256B) 输出, 对于每个 256B, 间隔 BufferNum * 256B 存放, 避免 bank 冲突
    uint16_t n1 = CeilDiv(tileSizeN, static_cast<uint64_t>(BLOCK_CUBE));
    uint16_t mainGroupNum = tileSizeK / groupSize; // 一个 kBubSize 中 group 的个数
    // 对于 kBubSize 的一个 group 中, 需要处理 innerNum 次 128 个数（对应 256B）, 目前只支持常数
    // NOTE groupSize是32的倍数，所以天然满足16的倍数，不需要做对齐
    uint16_t regNumInMainGroup = groupSize * C0<DtypeDst>::value / ONE_REG_ELEM<DtypeDst>::value;

    // extend以byte为单位
    // UNPK4_B8读取1/4寄存器大小(单次处理后 src 偏移为 128 个数（64B），偏移1/4)
    constexpr uint32_t innerSrcExtend = VECTOR_REG_WIDTH >> 2; // （8, 16) 128 Elements 64B
    uint32_t groupNumSrcExtend = regNumInMainGroup * innerSrcExtend;
    // (n1, k1, k0, n0)
    // NOTE kUb是512或1024，所以天然满足16的倍数，不需要做对齐
    uint32_t n1SrcExtend = kUb * (C0<DtypeDst>::value >> 1); // 按照UINT8类型计算

    // 单次处理后 dst 偏移为 128 * buffer数量，当buffer数量为4时，偏移1024B
    constexpr uint32_t innerDstExtend = ONE_REG_ELEM<DtypeDst>::value * BufferNum;
    uint32_t groupNumDstExtend = regNumInMainGroup * innerDstExtend;
    uint32_t n1DstExtend =
        CeilAlign(tileSizeK, static_cast<uint64_t>(C0<DtypeDst>::value)) * ONE_BLK_ELEM<DtypeDst>::value * BufferNum;
    // 获取vf使用的地址
    __local_mem__ DtypeDst *scaleBaseAddr = (__local_mem__ DtypeDst *)scale.GetPhyAddr();
    __local_mem__ DtypeDst *offsetBaseAddr = nullptr;
    if constexpr (hasAntiQuantOffset) {
        offsetBaseAddr = (__local_mem__ DtypeDst *)offset.GetPhyAddr();
    }
    __local_mem__ int8_t *weightInUbBaseAddr = (__local_mem__ int8_t *)weightIn.GetPhyAddr();
    __local_mem__ DtypeDst *weightOutUbBaseAddr = (__local_mem__ DtypeDst *)weightOut.GetPhyAddr();

    uint32_t mainGroupSize = mainGroupNum * groupSize;
#ifndef __CCE_KT_TEST__
    MainAddr<DtypeDst> mainAddr{scaleBaseAddr, offsetBaseAddr, weightInUbBaseAddr, weightOutUbBaseAddr};
    VfParams params{
        n1,          mainGroupNum,      regNumInMainGroup, 0, n1SrcExtend, groupNumSrcExtend, innerSrcExtend,
        n1DstExtend, groupNumDstExtend, innerDstExtend,    n};
    if (tileSizeK == mainGroupSize) {
        AntiQuantComputeKNGroupWeightNzMultiGroupWithoutTail<DtypeDst, hasAntiQuantOffset>(params, mainAddr);
    } else {
        uint16_t regNumInTailGroup = (tileSizeK - mainGroupSize) / (VECTOR_REG_WIDTH / sizeof(DtypeDst) / 16);
        __local_mem__ DtypeDst *tailScaleBaseAddr = scaleBaseAddr + mainGroupNum * n;
        __local_mem__ DtypeDst *tailOffsetBaseAddr = nullptr;
        if constexpr (hasAntiQuantOffset) {
            tailOffsetBaseAddr = offsetBaseAddr + mainGroupNum * n;
        }
        __local_mem__ int8_t *tailWeightInUbBaseAddr = weightInUbBaseAddr + mainGroupSize * (C0<DtypeDst>::value >> 1);
        __local_mem__ DtypeDst *tailWeightOutUbBaseAddr =
            weightOutUbBaseAddr + CeilAlign(mainGroupSize * BLOCK_CUBE, VECTOR_REG_WIDTH) * BufferNum;
        TailAddr<DtypeDst> tailAddr{tailScaleBaseAddr, tailOffsetBaseAddr, tailWeightInUbBaseAddr,
                                    tailWeightOutUbBaseAddr};
        params.regNumInTailGroup = regNumInTailGroup;
        AntiQuantComputeKNGroupWeightNzMultiGroupWithTail<DtypeDst, hasAntiQuantOffset>(params, mainAddr, tailAddr);
    }
#endif
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
