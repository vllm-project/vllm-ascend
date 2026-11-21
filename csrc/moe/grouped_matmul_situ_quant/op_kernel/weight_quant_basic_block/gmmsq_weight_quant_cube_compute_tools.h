/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmmsq_weight_quant_cube_compute_tools.h
 * \brief
 */

#ifndef GMMSQ_WEIGHT_QUANT_CUBE_COMPUTE_TOOLS_H
#define GMMSQ_WEIGHT_QUANT_CUBE_COMPUTE_TOOLS_H

#include "kernel_cube_intf.h"
#include "gmmsq_math_common.h"

using AscendC::BLOCK_CUBE;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;

namespace GMMSQWeightQuant {

static constexpr uint64_t L0AB_OPERATE_UNIT = 512L; // L0上搬运单位512Byte
static constexpr int32_t UNIT_FLAG_UPDATE = 3;      // 检查并更新标记位
static constexpr int32_t UNIT_FLAG_CHECK_ONLY = 2;  // 只检查标记位
static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

static constexpr uint64_t K_ALIGNMENT64 = 64UL; // 处理K轴非64对齐的场景

struct L0CopyAndCalcParams {
    uint64_t mL0Size; // M方向当前在L0上实际计算的大小
    uint64_t mL1Size; // M方向当前在L1上实际载入的大小
    uint64_t kL0Size; // K方向当前在L0上实际计算的大小
    uint64_t kL1Size; // K方向当前在L1上实际载入的大小
    uint64_t nL0Size; // N方向当前在L0上实际计算的大小
    uint64_t nL1Size; // N方向当前在L1上实际载入的大小
    uint64_t
        scaleKL1Size; // scale的K方向在L1上实际载入的大小，mx group数为scaleKL1Size/32，认为A和B的scale在L1载入量一致
    uint64_t aScaleL1GroupNum = 0; // ASW的A scale在L1上实际载入的分组数
    uint64_t bScaleL1GroupNum = 0; // ASW的B scale在L1上实际载入的分组数
    bool isFirstKLoop;             // 是否是第一次K方向的循环，用于控制L0C的初始值
    bool isLastKLoop;              // 是否是最后一次K方向的循环，用于控制unitFlag
};

struct FixL0CToDstParams {
    uint64_t mL0Size;         // M方向当前在L0上实际计算的大小
    uint64_t nL0Size;         // N方向当前在L0上实际计算的大小
    uint64_t outNSize;        // 输出N方向的总大小，用于dstStride
    uint64_t splitNSize = 0;  // fixp输出多个矩阵时，N方向切分的大小
    uint64_t dstNdStride = 0; // fixp输出多个矩阵时，目的相邻DN矩阵起始地址间的偏移
};

template <typename T>
__aicore__ inline constexpr uint64_t GetC0Size()
{
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        return ONE_BLK_SIZE + ONE_BLK_SIZE;
    } else {
        return ONE_BLK_SIZE / sizeof(T);
    }
}

/**
 * @brief 将A矩阵和scaleA矩阵从L1加载到L0A
 *
 * @param aL0Tensor L0A tensor
 * @param aL1Tensor L1A tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param scaleL1Tensor L1scale tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */

template <typename DstType, typename SrcType, typename XScaleType, bool UseIndependentScaleGroups = false>
__aicore__ inline void LoadAAndScaleL1ToL0(const LocalTensor<DstType>& aL0Tensor, const LocalTensor<SrcType>& aL1Tensor,
                                           const LocalTensor<XScaleType>& scaleL1Tensor,
                                           const L0CopyAndCalcParams& l0CopyAndCalcParams)
{
    uint64_t mL0AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t mL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition = 0;
    aL0Load2dParams.mStep = mL0AlignSize / static_cast<uint64_t>(BLOCK_CUBE);
    if constexpr (UseIndependentScaleGroups) {
        aL0Load2dParams.kStep =
            Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size * sizeof(SrcType), static_cast<uint64_t>(ONE_BLK_SIZE));
    } else {
        aL0Load2dParams.kStep =
            Ops::Base::CeilDiv(static_cast<uint64_t>(Ops::Base::CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64)),
                               GetC0Size<SrcType>());
    }
    aL0Load2dParams.srcStride = Ops::Base::CeilDiv(mL1AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);
    aL0Load2dParams.dstStride = Ops::Base::CeilDiv(mL0AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);

    AscendC::LoadData2DMxParams aL0Load2dMx;
    aL0Load2dMx.xStartPosition = 0;
    aL0Load2dMx.yStartPosition = 0;
    aL0Load2dMx.xStep = mL0AlignSize / static_cast<uint64_t>(BLOCK_CUBE);
    // 64: 表示K方向跨过了几个32B分形(k 不足 64 时按 64 对齐）
    if constexpr (UseIndependentScaleGroups) {
        // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
        aL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL);
        // 2：两个 B8 scale 合为一个 B16 搬运单元。
        aL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.aScaleL1GroupNum, 2UL);
    } else {
        aL0Load2dMx.yStep =
            // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
            Ops::Base::CeilDiv(static_cast<uint64_t>(Ops::Base::CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64)),
                               64UL);
        // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
        aL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.scaleKL1Size, 64UL);
    }
    aL0Load2dMx.dstStride = aL0Load2dMx.yStep;
    AscendC::LoadData(aL0Tensor, aL1Tensor, scaleL1Tensor, aL0Load2dParams, aL0Load2dMx);
}

/**
 * @brief 将B矩阵和scaleB矩阵从L1加载到L0B
 *
 * @param bL0Tensor L0B tensor
 * @param bL1Tensor L1B tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param scaleL1Tensor L1scale tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */

template <typename DstType, typename SrcType, typename WScaleType, bool UseIndependentScaleGroups = false,
          bool IsHif4Scale = false>
__aicore__ inline void LoadBAndScaleL1ToL0(const LocalTensor<DstType>& bL0Tensor, const LocalTensor<SrcType>& bL1Tensor,
                                           const LocalTensor<WScaleType>& scaleL1Tensor,
                                           const L0CopyAndCalcParams& l0CopyAndCalcParams)
{
    uint64_t nL0AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t nL1AlignSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition = 0;
    bL0Load2dParams.mStep = nL0AlignSize / static_cast<uint64_t>(BLOCK_CUBE);
    if constexpr (IsHif4Scale) {
        // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
        bL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL);
    } else {
        bL0Load2dParams.kStep =
            Ops::Base::CeilDiv(static_cast<uint64_t>(Ops::Base::CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64)),
                               GetC0Size<SrcType>());
    }
    bL0Load2dParams.srcStride = Ops::Base::CeilDiv(nL1AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);
    bL0Load2dParams.dstStride = Ops::Base::CeilDiv(nL0AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);

    AscendC::LoadData2DMxParams bL0Load2dMx;
    bL0Load2dMx.xStartPosition = 0;
    bL0Load2dMx.yStartPosition = 0;
    bL0Load2dMx.xStep = nL0AlignSize / static_cast<uint64_t>(BLOCK_CUBE);
    // yStep的含义是源矩阵Y轴（横向）搬运的32B的个数，不是Y轴本身的长度，而是包含了分形
    // mxfp4的scale是e8m0，yStep的计算方式为：
    // ((kL0 / mx_gsize) / 2) * (2 * 16 * sizeof(e8m0)) / 32[B] = kL0 / 64
    // 其中mx_gsize固定为32，除以2是表示2个e8m0构成一个分形， 2 * 16 * sizeof(e8m0)表示一个分形的大小，32B是单位
    // srcStride和dstStride计算过程类似
    if constexpr (UseIndependentScaleGroups && IsHif4Scale) {
        constexpr uint64_t B32_IN_B16_NUM = 2UL;
        // 64：一个 HIF4 scale 对应的 K 分组长度。
        bL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL) * B32_IN_B16_NUM;
        bL0Load2dMx.srcStride = l0CopyAndCalcParams.bScaleL1GroupNum * B32_IN_B16_NUM;
        bL0Load2dMx.dstStride = bL0Load2dMx.yStep;
    } else if constexpr (UseIndependentScaleGroups) {
        // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
        bL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL);
        // 2：两个 B8 scale 合为一个 B16 搬运单元。
        bL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.bScaleL1GroupNum, 2UL);
        bL0Load2dMx.dstStride = bL0Load2dMx.yStep;
    } else {
        bL0Load2dMx.yStep =
            // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
            Ops::Base::CeilDiv(static_cast<uint64_t>(Ops::Base::CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64)),
                               64UL);
        // 64：MX/HIF4 scale 搬运在 K 方向的分组粒度。
        bL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.scaleKL1Size, 64UL);
        bL0Load2dMx.dstStride = bL0Load2dMx.yStep;
    }
    AscendC::LoadData(bL0Tensor, bL1Tensor, scaleL1Tensor, bL0Load2dParams, bL0Load2dMx);
}

/**
 * @brief Mmad计算
 *
 * @param cL0Tensor L0C tensor
 * @param aL0Tensor L0A tensor
 * @param bL0Tensor L0B tensor
 * @param l0CopyAndCalcParams 搬运和计算参数
 */

template <typename DstType, typename SrcAType, typename SrcBType, bool AlignMmad = true>
__aicore__ inline void MmadCompute(const LocalTensor<DstType>& cL0Tensor, const LocalTensor<SrcAType>& aL0Tensor,
                                   const LocalTensor<SrcBType>& bL0Tensor,
                                   const L0CopyAndCalcParams& l0CopyAndCalcParams)
{
    AscendC::MmadParams mmadParams;
    if constexpr (AlignMmad) {
        // 2：Fixpipe/MMAD 的 M 维偶数对齐要求。
        mmadParams.m = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL0Size, 2UL);
        mmadParams.k = Ops::Base::CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64);
    } else {
        mmadParams.m = l0CopyAndCalcParams.mL0Size;
        mmadParams.k = l0CopyAndCalcParams.kL0Size;
    }
    mmadParams.n = l0CopyAndCalcParams.nL0Size;
    mmadParams.cmatrixInitVal = l0CopyAndCalcParams.isFirstKLoop;
    mmadParams.disableGemv = true;
    mmadParams.unitFlag = l0CopyAndCalcParams.isLastKLoop ? UNIT_FLAG_UPDATE : UNIT_FLAG_CHECK_ONLY;
    AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
}

/**
 * @brief 将L0C的数据切M分两路输出到UB，不做量化
 *
 * @param outTensor 输出 UB Tensor
 * @param cL0Tensor L0C tensor
 * @param fixL0CToDstParams 搬运参数
 * @tparam DstType UB数据类型
 * @tparam SrcType L0C数据类型
 */

template <typename DstType, typename SrcType, bool AlignMSize = true>
__aicore__ inline void FixL0CToDst(const LocalTensor<DstType>& outTensor, const LocalTensor<SrcType>& cL0Tensor,
                                   const FixL0CToDstParams& fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    if constexpr (AlignMSize) {
        fixParams.params.ndNum = 1;
        // 2：Fixpipe/MMAD 的 M 维偶数对齐要求。
        fixParams.mSize = Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, 2UL);
    } else {
        fixParams.mSize = fixL0CToDstParams.mL0Size;
    }
    if constexpr (AlignMSize) {
        fixParams.nSize = Ops::Base::CeilAlign(fixL0CToDstParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
    } else {
        fixParams.nSize = fixL0CToDstParams.nL0Size;
    }
    fixParams.srcStride = Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE));
    fixParams.dstStride = fixL0CToDstParams.outNSize;
    fixParams.unitFlag = UNIT_FLAG_UPDATE;
    fixParams.quantPre = QuantMode_t::NoQuant;
    if constexpr (AlignMSize) {
        fixParams.dualDstCtl = 1;
        AscendC::Fixpipe<DstType, SrcType, CFG_ROW_MAJOR_UB>(outTensor, cL0Tensor, fixParams);
    } else {
        AscendC::Fixpipe<DstType, SrcType, AscendC::Impl::CFG_ROW_MAJOR_UB>(outTensor, cL0Tensor, fixParams);
    }
}

} // namespace GMMSQWeightQuant
#endif
