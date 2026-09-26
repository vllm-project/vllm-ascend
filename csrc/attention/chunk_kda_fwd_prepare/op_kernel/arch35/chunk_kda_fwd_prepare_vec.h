/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H
#define ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <type_traits>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace KdaPrepare::Arch35 {

namespace Detail {

using namespace AscendC::MicroAPI;

constexpr static CastTrait kFp32ToBf16RintZero = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr static CastTrait kFp32ToBf16RintOne = {
    RegLayout::ONE,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr static CastTrait kFp32ToBf16RintZeroing = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_callee__ inline void CastFp32ToBf16Rint(
    RegTensor<bfloat16_t> &dst, RegTensor<float> &low,
    RegTensor<float> &high,
    MaskReg &mask)
{
    Cast<bfloat16_t, float, kFp32ToBf16RintOne>(dst, high, mask);
    Cast<bfloat16_t, float, kFp32ToBf16RintZero>(dst, low, mask);
}

template <typename T>
__simd_callee__ inline void Load128AsFp32(
    RegTensor<float> &low, RegTensor<float> &high, __ubuf__ T *src)
{
    if constexpr (std::is_same<T, float>::value) {
        // 连续的 128 个 FP32 元素按偶、奇维拆入两个 64-lane 寄存器，
        // 与 BF16 的 CastHalf2Float 结果保持同一维度语义。
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(low, high, src);
    } else {
        RegTensor<T> packed;
        LoadIn<T, false>(packed, src);
        MaskReg packedMask = CreateMask<T, MaskPattern::ALL>();
        CastHalf2Float<T>(low, high, packed, packedMask);
    }
}

__simd_callee__ inline void Store128FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &low,
    RegTensor<float> &high)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<bfloat16_t> packed;
    CastFp32ToBf16Rint(packed, low, high, floatMask);
    MaskReg packedMask = CreateMask<bfloat16_t, MaskPattern::ALL>();
    StoreAlign(dst, packed, packedMask);
}

__simd_callee__ inline void Store64FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &value)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<bfloat16_t> packed;
    Cast<bfloat16_t, float, kFp32ToBf16RintZeroing>(
        packed, value, floatMask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
        dst, packed, floatMask);
}

__simd_callee__ inline void Store32FromFp32(
    __ubuf__ bfloat16_t *dst, RegTensor<float> &value)
{
    uint32_t active = 32;
    MaskReg floatMask = UpdateMask<float>(active);
    RegTensor<bfloat16_t> packed;
    Cast<bfloat16_t, float, kFp32ToBf16RintZeroing>(
        packed, value, floatMask);
    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
        dst, packed, floatMask);
}

template <typename T>
__simd_callee__ inline void LoadScalarAsFp32(
    RegTensor<float> &dst, __ubuf__ T *src)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadIn<float, true>(dst, src);
    } else {
        RegTensor<T> raw;
        RegTensor<float> unused;
        MaskReg inputMask = CreateMask<T, MaskPattern::ALL>();
        LoadIn<T, true>(raw, src);
        CastHalf2Float<T>(dst, unused, raw, inputMask);
    }
}

template <bool USE_EXP2>
__simd_callee__ inline void ExpPair(
    RegTensor<float> &low, RegTensor<float> &high, float lower, float upper)
{
    using Domain = ExpDomainTraits<USE_EXP2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    Maxs(low, low, lower, mask);
    Maxs(high, high, lower, mask);
    Mins(low, low, upper, mask);
    Mins(high, high, upper, mask);
    if constexpr (Domain::useExp2) {
        Muls(low, low, Domain::expInputScale, mask);
        Muls(high, high, Domain::expInputScale, mask);
    }
    Exp(low, low, mask);
    Exp(high, high, mask);
}

// V0 的循环和指令均属于一次 VF。有效行和尾行分开执行，
// 保证循环体中只有编译期模式分支。
template <typename GateT, typename BetaT, typename CompilePolicy,
          bool BETA_SEQUENCE_MAJOR>
__simd_vf__ inline void StageV0Vf(
    __ubuf__ bfloat16_t *q, __ubuf__ bfloat16_t *k,
    __ubuf__ GateT *rawGate,
    __ubuf__ BetaT *betaRaw, __ubuf__ float *dtBias, __ubuf__ float *aLog,
    __ubuf__ float *g, __ubuf__ float *gRef, __ubuf__ float *gLast,
    __ubuf__ float *qRstd, __ubuf__ float *kRstd,
    __ubuf__ float *betaEff, uint16_t validRows, float epsilon,
    float lowerBound, bool hasDtBias, bool hasALog)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    RegTensor<float> carryLow;
    RegTensor<float> carryHigh;
    RegTensor<float> biasLow;
    RegTensor<float> biasHigh;
    RegTensor<float> a;
    Duplicate(carryLow, 0.0F, mask);
    Duplicate(carryHigh, 0.0F, mask);

    // dt_bias 和 A_log 都是 head 常量，在行循环前只判断和读取一次。
    if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
        if (hasDtBias) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                biasLow, biasHigh, dtBias);
        } else {
            Duplicate(biasLow, 0.0F, mask);
            Duplicate(biasHigh, 0.0F, mask);
        }
        if (hasALog) {
            LoadScalarAsFp32(a, aLog);
            Exp(a, a, mask); // 计算 a_h=exp(A_log[h])。
        } else {
            Duplicate(a, 1.0F, mask);
        }
    }

    // 有效行内直接展示 Q/K 归一化、gate 变换和前缀和。
    for (uint16_t row = 0; row < validRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        Load128AsFp32(qLow, qHigh, q + row * Shape::kHeadDim);
        Load128AsFp32(kLow, kHigh, k + row * Shape::kHeadDim);
        if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
            RegTensor<float> qSquareLow;
            RegTensor<float> qSquareHigh;
            RegTensor<float> kSquareLow;
            RegTensor<float> kSquareHigh;
            RegTensor<float> qSumLow;
            RegTensor<float> qSumHigh;
            RegTensor<float> kSumLow;
            RegTensor<float> kSumHigh;
            Mul(qSquareLow, qLow, qLow, mask);
            Mul(qSquareHigh, qHigh, qHigh, mask);
            Mul(kSquareLow, kLow, kLow, mask);
            Mul(kSquareHigh, kHigh, kHigh, mask);
            ReduceSum(qSumLow, qSquareLow, mask);
            ReduceSum(qSumHigh, qSquareHigh, mask);
            ReduceSum(kSumLow, kSquareLow, mask);
            ReduceSum(kSumHigh, kSquareHigh, mask);
            // ReduceSum 只保证首 lane 有效。rstd 供当前行归一化广播，
            // 需要反向保存量时再落入 UB 并由 MTE3 写回。
            Add(qSumLow, qSumLow, qSumHigh, scalarMask);
            Add(kSumLow, kSumLow, kSumHigh, scalarMask);
            Adds(qSumLow, qSumLow, epsilon, scalarMask);
            Adds(kSumLow, kSumLow, epsilon, scalarMask);
            Sqrt(qSumLow, qSumLow, scalarMask);
            Sqrt(kSumLow, kSumLow, scalarMask);
            RegTensor<float> one;
            Duplicate(one, 1.0F, scalarMask);
            Div(qSumLow, one, qSumLow, scalarMask);
            Div(kSumLow, one, kSumLow, scalarMask);
            if constexpr (CompilePolicy::outputMode != OutputMode::None) {
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    qRstd + row, qSumLow, scalarMask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    kRstd + row, kSumLow, scalarMask);
            }
            // ReduceSum 的最低 lane 已是最终 rstd，直接在寄存器内广播，
            // 不再为当前行计算做 UB 往返。
            RegTensor<float> qScale;
            RegTensor<float> kScale;
            Duplicate(qScale, qSumLow, mask);
            Duplicate(kScale, kSumLow, mask);
            Mul(qLow, qLow, qScale, mask);
            Mul(qHigh, qHigh, qScale, mask);
            Mul(kLow, kLow, kScale, mask);
            Mul(kHigh, kHigh, kScale, mask);
        } else {
            // Identity 模式仅在需要反向保存量时产生公开的 rstd=1。
            if constexpr (CompilePolicy::outputMode != OutputMode::None) {
                RegTensor<float> one;
                Duplicate(one, 1.0F, scalarMask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    qRstd + row, one, scalarMask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    kRstd + row, one, scalarMask);
            }
        }
        Store128FromFp32(q + row * Shape::kHeadDim, qLow, qHigh);
        Store128FromFp32(k + row * Shape::kHeadDim, kLow, kHigh);

        RegTensor<float> gateLow;
        RegTensor<float> gateHigh;
        Load128AsFp32(gateLow, gateHigh, rawGate + row * Shape::kHeadDim);
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            Add(gateLow, gateLow, biasLow, mask);
            Add(gateHigh, gateHigh, biasHigh, mask);
            if constexpr (CompilePolicy::safeGate ||
                          CompilePolicy::gateMode == GateMode::SafeSigmoid) {
                RegTensor<float> one;
                Duplicate(one, 1.0F, mask);
                Mul(gateLow, gateLow, a, mask);
                Mul(gateHigh, gateHigh, a, mask);
                Muls(gateLow, gateLow, -1.0F, mask);
                Muls(gateHigh, gateHigh, -1.0F, mask);
                Exp(gateLow, gateLow, mask);
                Exp(gateHigh, gateHigh, mask);
                Adds(gateLow, gateLow, 1.0F, mask);
                Adds(gateHigh, gateHigh, 1.0F, mask);
                Div(gateLow, one, gateLow, mask);
                Div(gateHigh, one, gateHigh, mask);
                Muls(gateLow, gateLow, lowerBound, mask);
                Muls(gateHigh, gateHigh, lowerBound, mask);
            } else {
                RegTensor<float> positiveLow;
                RegTensor<float> positiveHigh;
                RegTensor<float> softplusLow;
                RegTensor<float> softplusHigh;
                Maxs(positiveLow, gateLow, 0.0F, mask);
                Maxs(positiveHigh, gateHigh, 0.0F, mask);
                Abs(softplusLow, gateLow, mask);
                Abs(softplusHigh, gateHigh, mask);
                Muls(softplusLow, softplusLow, -1.0F, mask);
                Muls(softplusHigh, softplusHigh, -1.0F, mask);
                Exp(softplusLow, softplusLow, mask);
                Exp(softplusHigh, softplusHigh, mask);
                Adds(softplusLow, softplusLow, 1.0F, mask);
                Adds(softplusHigh, softplusHigh, 1.0F, mask);
                Ln(softplusLow, softplusLow, mask);
                Ln(softplusHigh, softplusHigh, mask);
                Add(gateLow, positiveLow, softplusLow, mask);
                Add(gateHigh, positiveHigh, softplusHigh, mask);
                Mul(gateLow, gateLow, a, mask);
                Mul(gateHigh, gateHigh, a, mask);
                Muls(gateLow, gateLow, -1.0F, mask);
                Muls(gateHigh, gateHigh, -1.0F, mask);
            }
        }
        // USE_EXP2=true 时 G 保存 log2 值；false 时保存自然对数值。
        if constexpr (Domain::useExp2) {
            Muls(gateLow, gateLow, Domain::stepScale, mask);
            Muls(gateHigh, gateHigh, Domain::stepScale, mask);
        }
        Add(carryLow, carryLow, gateLow, mask);
        Add(carryHigh, carryHigh, gateHigh, mask);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(
            g + row * Shape::kHeadDim, carryLow, carryHigh, mask);
    }

    // G 先完整落到 UB，然后在循环外取四个局部参考行。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> refLow;
    RegTensor<float> refHigh;
    if (validRows == 0) {
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, mask);
        StoreAlign(gRef, zero, mask);
        StoreAlign(gRef + 64, zero, mask);
        StoreAlign(gRef + Shape::kHeadDim, zero, mask);
        StoreAlign(gRef + Shape::kHeadDim + 64, zero, mask);
        StoreAlign(gRef + 2 * Shape::kHeadDim, zero, mask);
        StoreAlign(gRef + 2 * Shape::kHeadDim + 64, zero, mask);
        StoreAlign(gRef + 3 * Shape::kHeadDim, zero, mask);
        StoreAlign(gRef + 3 * Shape::kHeadDim + 64, zero, mask);
        StoreAlign(gLast, zero, mask);
        StoreAlign(gLast + 64, zero, mask);
    }
    if (validRows > 0) {
        const uint16_t end = validRows < 16 ? validRows : 16;
        const uint16_t refRow = end / 2;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            refLow, refHigh, g + refRow * Shape::kHeadDim);
        StoreAlign(gRef, refLow, mask);
        StoreAlign(gRef + 64, refHigh, mask);
    }
    if (validRows > 16) {
        const uint16_t end = validRows < 32 ? validRows : 32;
        const uint16_t refRow = (16 + end) / 2;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            refLow, refHigh, g + refRow * Shape::kHeadDim);
        StoreAlign(gRef + Shape::kHeadDim, refLow, mask);
        StoreAlign(gRef + Shape::kHeadDim + 64, refHigh, mask);
    }
    if (validRows > 32) {
        const uint16_t end = validRows < 48 ? validRows : 48;
        const uint16_t refRow = (32 + end) / 2;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            refLow, refHigh, g + refRow * Shape::kHeadDim);
        StoreAlign(gRef + 2 * Shape::kHeadDim, refLow, mask);
        StoreAlign(gRef + 2 * Shape::kHeadDim + 64, refHigh, mask);
    }
    if (validRows > 48) {
        const uint16_t refRow = (48 + validRows) / 2;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            refLow, refHigh, g + refRow * Shape::kHeadDim);
        StoreAlign(gRef + 3 * Shape::kHeadDim, refLow, mask);
        StoreAlign(gRef + 3 * Shape::kHeadDim + 64, refHigh, mask);
    }
    if (validRows > 0) {
        const uint16_t lastRow = validRows - 1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            refLow, refHigh, g + lastRow * Shape::kHeadDim);
        StoreAlign(gLast, refLow, mask);
        StoreAlign(gLast + 64, refHigh, mask);
    }

    for (uint16_t row = 0; row < validRows; ++row) {
        RegTensor<float> beta;
        RegTensor<float> one;
        constexpr uint16_t kBetaRowElements =
            BETA_SEQUENCE_MAJOR ? 32 / sizeof(BetaT) : 1;
        LoadScalarAsFp32(beta, betaRaw + row * kBetaRowElements);
        Duplicate(one, 1.0F, mask);
        // Raw 模式只把 BF16/FP32 输入统一转成 FP32。
        if constexpr (CompilePolicy::betaMode != BetaMode::Raw) {
            Muls(beta, beta, -1.0F, mask);
            Exp(beta, beta, mask);
            Adds(beta, beta, 1.0F, mask);
            Div(beta, one, beta, mask);
            if constexpr (CompilePolicy::betaMode == BetaMode::TwoSigmoid) {
                Muls(beta, beta, 2.0F, mask);
            }
        }
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            betaEff + row, beta, mask);
    }
}

template <uint16_t BAND, typename CompilePolicy>
__simd_callee__ inline void ComputeStageV1KMinusBand(
    __ubuf__ bfloat16_t *kHat, __ubuf__ float *g,
    __ubuf__ float *gRef, __ubuf__ bfloat16_t *kMinus,
    uint16_t validRows, MaskReg &mask)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    constexpr uint16_t prefixRows = Shape::kPrefixRows[BAND];
    constexpr uint16_t bandBegin = BAND * Shape::kSubChunkRows;
    constexpr uint32_t prefixBase =
        Shape::kSubChunkRows * Shape::kHeadDim * BAND * (BAND + 1) / 2;
    const uint16_t rowsInChunk = validRows < prefixRows
                                     ? validRows
                                     : prefixRows;
    const uint16_t computeRows = bandBegin < validRows ? rowsInChunk : 0;
    for (uint16_t row = 0; row < computeRows; ++row) {
        RegTensor<float> gateLow;
        RegTensor<float> gateHigh;
        RegTensor<float> refLow;
        RegTensor<float> refHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> outputLow;
        RegTensor<float> outputHigh;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            gateLow, gateHigh, g + row * Shape::kHeadDim);
        LoadAlign(refLow, gRef + BAND * Shape::kHeadDim);
        LoadAlign(refHigh, gRef + BAND * Shape::kHeadDim + 64);
        Sub(refLow, refLow, gateLow, mask);
        Sub(refHigh, refHigh, gateHigh, mask);
        constexpr float lower =
            Domain::StoredBound(ExpDomain::kV1Bf16LowerBase2);
        constexpr float upper =
            Domain::StoredBound(ExpDomain::kV1Bf16UpperBase2);
        ExpPair<CompilePolicy::useExp2>(refLow, refHigh, lower, upper);
        Load128AsFp32(kLow, kHigh, kHat + row * Shape::kHeadDim);
        Mul(outputLow, kLow, refLow, mask);
        Mul(outputHigh, kHigh, refHigh, mask);
        Store128FromFp32(
            kMinus + prefixBase + row * Shape::kHeadDim,
            outputLow, outputHigh);
    }
    for (uint16_t row = computeRows; row < prefixRows; ++row) {
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, mask);
        Store128FromFp32(
            kMinus + prefixBase + row * Shape::kHeadDim,
            zero, zero);
    }
}

template <uint16_t BAND, typename CompilePolicy>
__simd_callee__ inline void ComputeStageV1PlusBand(
    __ubuf__ bfloat16_t *qHat, __ubuf__ bfloat16_t *kHat,
    __ubuf__ float *g, __ubuf__ float *gRef,
    __ubuf__ bfloat16_t *qPlus, __ubuf__ bfloat16_t *kPlus,
    uint16_t validRows, MaskReg &mask)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    constexpr uint16_t kBandBegin = BAND * Shape::kSubChunkRows;
    const uint16_t bandEnd = validRows < kBandBegin + Shape::kSubChunkRows
                                 ? validRows
                                 : kBandBegin + Shape::kSubChunkRows;
    const uint16_t computeRows = validRows > kBandBegin
                                     ? bandEnd - kBandBegin
                                     : 0;
    for (uint16_t bandRow = 0; bandRow < computeRows; ++bandRow) {
        constexpr float lower =
            Domain::StoredBound(ExpDomain::kV1Bf16LowerBase2);
        constexpr float upper =
            Domain::StoredBound(ExpDomain::kV1Bf16UpperBase2);
        const uint16_t row = kBandBegin + bandRow;
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> expLow;
        RegTensor<float> expHigh;
        RegTensor<float> refLow;
        RegTensor<float> refHigh;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            expLow, expHigh, g + row * Shape::kHeadDim);
        LoadAlign(refLow, gRef + BAND * Shape::kHeadDim);
        LoadAlign(refHigh, gRef + BAND * Shape::kHeadDim + 64);
        Sub(expLow, expLow, refLow, mask);
        Sub(expHigh, expHigh, refHigh, mask);
        ExpPair<CompilePolicy::useExp2>(expLow, expHigh, lower, upper);
        Load128AsFp32(qLow, qHigh, qHat + row * Shape::kHeadDim);
        Load128AsFp32(kLow, kHigh, kHat + row * Shape::kHeadDim);
        Mul(qLow, qLow, expLow, mask);
        Mul(qHigh, qHigh, expHigh, mask);
        Mul(kLow, kLow, expLow, mask);
        Mul(kHigh, kHigh, expHigh, mask);
        Store128FromFp32(qPlus + row * Shape::kHeadDim, qLow, qHigh);
        Store128FromFp32(kPlus + row * Shape::kHeadDim, kLow, kHigh);
    }
    for (uint16_t bandRow = computeRows;
         bandRow < Shape::kSubChunkRows; ++bandRow) {
        const uint16_t row = kBandBegin + bandRow;
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, mask);
        Store128FromFp32(qPlus + row * Shape::kHeadDim, zero, zero);
        Store128FromFp32(kPlus + row * Shape::kHeadDim, zero, zero);
    }
}

template <typename CompilePolicy>
__simd_vf__ inline void StageV1Vf(
    __ubuf__ bfloat16_t *qHat, __ubuf__ bfloat16_t *kHat,
    __ubuf__ float *g, __ubuf__ float *gRef,
    __ubuf__ bfloat16_t *qPlus, __ubuf__ bfloat16_t *kPlus,
    __ubuf__ bfloat16_t *kMinus, uint16_t validRows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();

    // 四个 Kminus 前缀区固定展开，每区再分有效行和尾行。
    ComputeStageV1KMinusBand<0, CompilePolicy>(
        kHat, g, gRef, kMinus, validRows, mask);
    ComputeStageV1KMinusBand<1, CompilePolicy>(
        kHat, g, gRef, kMinus, validRows, mask);
    ComputeStageV1KMinusBand<2, CompilePolicy>(
        kHat, g, gRef, kMinus, validRows, mask);
    ComputeStageV1KMinusBand<3, CompilePolicy>(
        kHat, g, gRef, kMinus, validRows, mask);

    // Kminus 完成后再按四段生成 Qplus/Kplus，避免 Khat 被提前覆盖，
    // 同时让 Gref 地址在每个 VF callee 中保持编译期常量。
    ComputeStageV1PlusBand<0, CompilePolicy>(
        qHat, kHat, g, gRef, qPlus, kPlus, validRows, mask);
    ComputeStageV1PlusBand<1, CompilePolicy>(
        qHat, kHat, g, gRef, qPlus, kPlus, validRows, mask);
    ComputeStageV1PlusBand<2, CompilePolicy>(
        qHat, kHat, g, gRef, qPlus, kPlus, validRows, mask);
    ComputeStageV1PlusBand<3, CompilePolicy>(
        qHat, kHat, g, gRef, qPlus, kPlus, validRows, mask);
}

template <uint16_t BAND>
__simd_callee__ inline void UnpackStageV3Band(
    __ubuf__ float *rawScore, __ubuf__ float *betaEff,
    __ubuf__ bfloat16_t *aqk, __ubuf__ float *lkk, __ubuf__ float *b,
    uint16_t validBandRows, float scale, MaskReg &full,
    MaskReg &lowerColumnMask, MaskReg &upperColumnMask)
{
    constexpr uint16_t kBandBegin = BAND * Shape::kSubChunkRows;
    constexpr uint16_t kColumns = (BAND + 1) * Shape::kSubChunkRows;
    constexpr uint32_t kStackedBase =
        Shape::kSubChunkRows * Shape::kSubChunkRows * BAND * (BAND + 1);
    constexpr uint32_t kAkkBandOffset =
        Shape::kSubChunkRows * kColumns;

    for (uint16_t bandRow = 0; bandRow < validBandRows; ++bandRow) {
        const uint16_t row = kBandBegin + bandRow;
        RegTensor<float> aqkRow;
        RegTensor<float> akkRow;
        LoadAlign(aqkRow,
                  rawScore + kStackedBase + bandRow * kColumns);
        LoadAlign(akkRow,
                  rawScore + kStackedBase + kAkkBandOffset +
                      bandRow * kColumns);
        uint32_t aqkCount = static_cast<uint32_t>(row) + 1;
        uint32_t akkCount = static_cast<uint32_t>(row);
        MaskReg aqkMask = UpdateMask<float>(aqkCount);
        MaskReg akkMask = UpdateMask<float>(akkCount);
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, full);
        Select(aqkRow, aqkRow, zero, aqkMask);
        Select(akkRow, akkRow, zero, akkMask);
        Muls(aqkRow, aqkRow, scale, full);
        RegTensor<float> beta;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(
            beta, betaEff + row);
        Mul(akkRow, akkRow, beta, full);
        Store64FromFp32(aqk + row * Shape::kChunkRows, aqkRow);
        if constexpr (BAND < 2) {
            StoreAlign(lkk + row * Shape::kChunkRows,
                       akkRow, lowerColumnMask);
        } else {
            StoreAlign(b + (row - 32) * 32, akkRow, lowerColumnMask);
            StoreAlign(lkk + row * Shape::kChunkRows,
                       akkRow, upperColumnMask);
        }
    }
    for (uint16_t bandRow = validBandRows;
         bandRow < Shape::kSubChunkRows; ++bandRow) {
        const uint16_t row = kBandBegin + bandRow;
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, full);
        if constexpr (BAND < 2) {
            StoreAlign(lkk + row * Shape::kChunkRows,
                       zero, lowerColumnMask);
        } else {
            StoreAlign(b + (row - 32) * 32, zero, lowerColumnMask);
            StoreAlign(lkk + row * Shape::kChunkRows,
                       zero, upperColumnMask);
        }
    }
}

__simd_vf__ inline void StageV3Vf(
    __ubuf__ float *rawScore,
    __ubuf__ float *betaEff, __ubuf__ bfloat16_t *aqk,
    __ubuf__ float *lkk, __ubuf__ float *b, __ubuf__ float *x0,
    __ubuf__ float *x1, __ubuf__ float *negX1,
    __ubuf__ bfloat16_t *akk, uint16_t band0Rows,
    uint16_t band1Rows, uint16_t band2Rows, uint16_t band3Rows,
    float scale)
{
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    uint32_t lowerColumnCount = 32;
    MaskReg lowerColumnMask = UpdateMask<float>(lowerColumnCount);
    RegTensor<int32_t> column;
    MaskReg upperColumnMask;
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(column, 0);
    CompareScalar<int32_t, AscendC::CMPMODE::GE>(
        upperColumnMask, column, 32, full);
    // 四段 compact score 以编译期 BAND 展开，VF 内不做动态数组索引。
    UnpackStageV3Band<0>(rawScore, betaEff, aqk, lkk, b,
                         band0Rows, scale, full,
                         lowerColumnMask, upperColumnMask);
    UnpackStageV3Band<1>(rawScore, betaEff, aqk, lkk, b,
                         band1Rows, scale, full,
                         lowerColumnMask, upperColumnMask);
    UnpackStageV3Band<2>(rawScore, betaEff, aqk, lkk, b,
                         band2Rows, scale, full,
                         lowerColumnMask, upperColumnMask);
    UnpackStageV3Band<3>(rawScore, betaEff, aqk, lkk, b,
                         band3Rows, scale, full,
                         lowerColumnMask, upperColumnMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    uint32_t rowCount = 32;
    MaskReg rowMask = UpdateMask<float>(rowCount);
    RegTensor<float> xZero;
    RegTensor<float> one;
    Duplicate(xZero, 0.0F, full);
    Duplicate(one, 1.0F, full);

    // LoadAlign 每次读取 64 个 FP32 lane。第一次写同时生成 row0 的单位行
    // 并清零 row1，剩余物理空间继续清零，避免递推读取未初始化高 lane。
    MaskReg firstColumn;
    RegTensor<float> firstTwoRows;
    CompareScalar<int32_t, AscendC::CMPMODE::EQ>(
        firstColumn, column, 0, full);
    Select(firstTwoRows, one, xZero, firstColumn);
    StoreAlign(x0, firstTwoRows, full);
    StoreAlign(x1, firstTwoRows, full);
    for (uint16_t offset = 64; offset < 32 * 32; offset += 64) {
        StoreAlign(x0 + offset, xZero, full);
        StoreAlign(x1 + offset, xZero, full);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // X0=(I+L00)^-1，X1=(I+L11)^-1。两条递推链按相同 source
    // 交错下发，各自累加顺序不变；每行共同使用一次写后读屏障。
    for (uint16_t row = 1; row < 32; ++row) {
        RegTensor<float> result0;
        RegTensor<float> result1;
        MaskReg diagonal;
        CompareScalar<int32_t, AscendC::CMPMODE::EQ>(
            diagonal, column, static_cast<int32_t>(row), rowMask);
        Select(result0, one, xZero, diagonal);
        Select(result1, one, xZero, diagonal);
        for (uint16_t source = 0; source < row; ++source) {
            RegTensor<float> factor0;
            RegTensor<float> factor1;
            RegTensor<float> sourceRow0;
            RegTensor<float> sourceRow1;
            RegTensor<float> product0;
            RegTensor<float> product1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                factor0, lkk + row * 64 + source);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                factor1, lkk + (row + 32) * 64 + 32 + source);
            LoadAlign(sourceRow0, x0 + source * 32);
            LoadAlign(sourceRow1, x1 + source * 32);
            Mul(product0, sourceRow0, factor0, rowMask);
            Mul(product1, sourceRow1, factor1, rowMask);
            Sub(result0, result0, product0, rowMask);
            Sub(result1, result1, product1, rowMask);
        }
        StoreAlign(x0 + row * 32, result0, rowMask);
        StoreAlign(x1 + row * 32, result1, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> x1Row;
        LoadAlign(x1Row, x1 + row * 32);
        Muls(x1Row, x1Row, -1.0F, rowMask);
        StoreAlign(negX1 + row * 32, x1Row, rowMask);
    }
    // q00/q11 写入最终 64x64 行主序 Akk；q01/q10 先置零，C5 只补 q10。
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> diagonalRow;
        LoadAlign(diagonalRow, x0 + row * 32);
        Store32FromFp32(akk + row * 64, diagonalRow);
        Store32FromFp32(akk + row * 64 + 32, xZero);
    }
    for (uint16_t row = 0; row < 32; ++row) {
        RegTensor<float> diagonalRow;
        Store32FromFp32(akk + (row + 32) * 64, xZero);
        LoadAlign(diagonalRow, x1 + row * 32);
        Store32FromFp32(akk + (row + 32) * 64 + 32, diagonalRow);
    }
}

template <typename CompilePolicy>
__simd_vf__ inline void StageV6Vf(
    __ubuf__ bfloat16_t *qHat, __ubuf__ bfloat16_t *kHat,
    __ubuf__ bfloat16_t *v, __ubuf__ float *g, __ubuf__ float *gLast,
    __ubuf__ float *betaEff, __ubuf__ bfloat16_t *qg,
    __ubuf__ bfloat16_t *kg, __ubuf__ bfloat16_t *qgScaled,
    __ubuf__ bfloat16_t *kBetaG, __ubuf__ bfloat16_t *vBeta,
    uint16_t validRows, float scale)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();

    // 第一阶段只生成三个需要 BF16 舍入的中间结果。所有行完成后统一
    // 建立一次 VEC_STORE->VEC_LOAD 依赖，避免原实现每行两次屏障。
    for (uint16_t row = 0; row < validRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> gateLow;
        RegTensor<float> gateHigh;
        RegTensor<float> lastLow;
        RegTensor<float> lastHigh;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            gateLow, gateHigh, g + row * 128);
        LoadAlign(lastLow, gLast);
        LoadAlign(lastHigh, gLast + 64);
        Load128AsFp32(qLow, qHigh, qHat + row * 128);
        Load128AsFp32(kLow, kHigh, kHat + row * 128);

        RegTensor<float> posLow;
        RegTensor<float> posHigh;
        RegTensor<float> kPosLow;
        RegTensor<float> kPosHigh;
        Adds(posLow, gateLow, 0.0F, mask);
        Adds(posHigh, gateHigh, 0.0F, mask);
        constexpr float lower =
            Domain::StoredBound(ExpDomain::kV6LowerBase2);
        constexpr float upper =
            Domain::StoredBound(ExpDomain::kV6UpperBase2);
        ExpPair<CompilePolicy::useExp2>(posLow, posHigh, lower, upper);
        Mul(qLow, qLow, posLow, mask);
        Mul(qHigh, qHigh, posHigh, mask);
        Mul(kPosLow, kLow, posLow, mask);
        Mul(kPosHigh, kHigh, posHigh, mask);
        Store128FromFp32(qg + row * 128, qLow, qHigh);
        // 第一次舍入先落到最终 kBetaG 物理区，第二阶段再回读 FP32。
        Store128FromFp32(kBetaG + row * 128, kPosLow, kPosHigh);

        Sub(lastLow, lastLow, gateLow, mask);
        Sub(lastHigh, lastHigh, gateHigh, mask);
        ExpPair<CompilePolicy::useExp2>(lastLow, lastHigh, lower, upper);
        Mul(kLow, kLow, lastLow, mask);
        Mul(kHigh, kHigh, lastHigh, mask);
        Store128FromFp32(kg + row * 128, kLow, kHigh);
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 第二阶段统一完成 beta 和 scale。qg、K_beta_g 仍从 BF16 中间量
    // 回读，因此数值舍入顺序与原实现一致。
    for (uint16_t row = 0; row < validRows; ++row) {
        RegTensor<float> qLow;
        RegTensor<float> qHigh;
        RegTensor<float> kLow;
        RegTensor<float> kHigh;
        RegTensor<float> vLow;
        RegTensor<float> vHigh;
        RegTensor<float> beta;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(beta, betaEff + row);
        Load128AsFp32(qLow, qHigh, qg + row * 128);
        Load128AsFp32(kLow, kHigh, kBetaG + row * 128);
        Load128AsFp32(vLow, vHigh, v + row * 128);
        Mul(kLow, kLow, beta, mask);
        Mul(kHigh, kHigh, beta, mask);
        Mul(vLow, vLow, beta, mask);
        Mul(vHigh, vHigh, beta, mask);
        Muls(qLow, qLow, scale, mask);
        Muls(qHigh, qHigh, scale, mask);
        Store128FromFp32(qgScaled + row * 128, qLow, qHigh);
        Store128FromFp32(kBetaG + row * 128, kLow, kHigh);
        Store128FromFp32(vBeta + row * 128, vLow, vHigh);
    }

    // C7 按 32/64 行读取两个 RHS，只清零它实际会读取的尾行。
    const uint16_t rhsRows = validRows > 32 ? 64 : 32;
    for (uint16_t row = validRows; row < rhsRows; ++row) {
        RegTensor<float> zero;
        Duplicate(zero, 0.0F, mask);
        Store128FromFp32(kBetaG + row * 128, zero, zero);
        Store128FromFp32(vBeta + row * 128, zero, zero);
    }
}

} // namespace Detail

template <typename GateT, typename BetaT, typename CompilePolicy>
class ChunkKdaFwdPrepareVec {
public:
    __aicore__ inline void Init(const PrepareKernelArgs &args)
    {
        args_ = args;
        workgroup_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
        gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args.rawGate));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(args.beta));
        // rawGate/V 在每个 chunk/value head 只读取一次，绕过 L2 避免
        // 流式输入挤占后续阶段会再次读取的 workspace 与 gk 数据。
        vGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gateGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // HK=HV 时 Q/K 与 value head 一一对应，同样只读取一次；GVA 下
        // 同一个 QK head 会被多个 value head 复用，继续保留默认 L2 策略。
        if (args.tiling.qkHeadNum == args.tiling.valueHeadNum) {
            qGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            kGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (args.dtBias != nullptr) {
            dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dtBias));
        }
        if (args.aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.aLog));
        }
        if constexpr (CompilePolicy::outputMode == OutputMode::Save) {
            qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qg));
        }
        qgScaledGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(args.qgScaled));
        kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kg));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.gk));
        aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.aqk));
        if constexpr (CompilePolicy::outputAkk) {
            akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.akk));
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            qHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.qHat));
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            kHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.kHat));
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.qRstd));
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.kRstd));
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            betaEffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.betaEff));
        }
    }

    __aicore__ inline void Process()
    {
        if (args_.tiling.usedCoreNum == 0 ||
            workgroup_ >= args_.tiling.usedCoreNum) {
            return;
        }
        const uint32_t coreCount = args_.tiling.usedCoreNum;
        const uint32_t chunkWork = static_cast<uint32_t>(
            static_cast<uint64_t>(args_.tiling.batch) *
            args_.tiling.totalChunks);
        const uint32_t headsPerQk =
            args_.tiling.valueHeadNum / args_.tiling.qkHeadNum;
        const bool fourHeadGroupPreservesGva =
            headsPerQk != 0 && Shape::kHeadsPerGroup % headsPerQk == 0;
        const bool balanceDenseTail =
            !args_.tiling.isVarLen && HeadPartitionCount(args_.tiling) == 1 &&
            fourHeadGroupPreservesGva && chunkWork >= coreCount &&
            chunkWork % coreCount != 0;
        if (!balanceDenseTail) {
            const uint32_t total = TotalWorkItems(args_.tiling);
            const uint32_t workBegin = WorkBegin(
                total, workgroup_, coreCount);
            const uint32_t workEnd = WorkEnd(
                total, workgroup_, coreCount);
            for (uint32_t work = workBegin; work < workEnd; ++work) {
                uint32_t globalChunk = 0;
                uint32_t headPartition = 0;
                DecodeWorkItem(
                    args_.tiling, work, globalChunk, headPartition);
                ChunkRange chunk{};
                if (!ResolveChunk(args_, globalChunk, chunk)) {
                    continue;
                }
                uint32_t headBegin = 0;
                uint32_t headEnd = 0;
                HeadRange(
                    args_.tiling, headPartition, headBegin, headEnd);
                ProcessChunkHeadRange(chunk, headBegin, headEnd);
            }
        } else {
            // 主体 chunk 仍然只按 chunk 分核，每核处理相同数量的完整 head。
            const uint32_t chunksPerCore = chunkWork / coreCount;
            const uint32_t bodyChunkCount = chunksPerCore * coreCount;
            const uint32_t bodyBegin = workgroup_ * chunksPerCore;
            const uint32_t bodyEnd = bodyBegin + chunksPerCore;
            for (uint32_t globalChunk = bodyBegin;
                 globalChunk < bodyEnd; ++globalChunk) {
                ChunkRange chunk{};
                if (!ResolveChunk(args_, globalChunk, chunk)) {
                    continue;
                }
                ProcessChunkHeadRange(
                    chunk, 0, args_.tiling.valueHeadNum);
            }

            // 不足一轮的尾部 chunk 再按 4-head group 展开，均摊到全部核。
            const uint32_t headGroupCount = CeilDiv(
                args_.tiling.valueHeadNum, Shape::kHeadsPerGroup);
            const uint32_t tailChunkCount = chunkWork - bodyChunkCount;
            const uint64_t tailTaskCount =
                static_cast<uint64_t>(tailChunkCount) * headGroupCount;
            uint64_t tailTask = tailTaskCount * workgroup_ / coreCount;
            const uint64_t tailTaskEnd =
                tailTaskCount * (workgroup_ + 1) / coreCount;
            while (tailTask < tailTaskEnd) {
                const uint32_t tailChunk = static_cast<uint32_t>(
                    tailTask / headGroupCount);
                const uint32_t firstHeadGroup = static_cast<uint32_t>(
                    tailTask % headGroupCount);
                uint64_t segmentEnd =
                    static_cast<uint64_t>(tailChunk + 1) * headGroupCount;
                if (segmentEnd > tailTaskEnd) {
                    segmentEnd = tailTaskEnd;
                }
                const uint32_t segmentGroups = static_cast<uint32_t>(
                    segmentEnd - tailTask);
                const uint32_t headBegin =
                    firstHeadGroup * Shape::kHeadsPerGroup;
                uint32_t headEnd = (firstHeadGroup + segmentGroups) *
                    Shape::kHeadsPerGroup;
                if (headEnd > args_.tiling.valueHeadNum) {
                    headEnd = args_.tiling.valueHeadNum;
                }
                ChunkRange chunk{};
                if (ResolveChunk(
                        args_, bodyChunkCount + tailChunk, chunk)) {
                    ProcessChunkHeadRange(chunk, headBegin, headEnd);
                }
                tailTask = segmentEnd;
            }
        }
        // 消费最后一次 C7 free，保证每次 set 都有对应 wait。
        constexpr uint16_t kAicToAivSlotReusableFlagId[2] = {4, 5};
        for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
            if (usedLocalSlot_[localSlot]) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                    kAicToAivSlotReusableFlagId[localSlot]);
            }
        }
    }

private:
    __aicore__ inline void ProcessChunkHeadRange(
        const ChunkRange &chunk, uint32_t headBegin, uint32_t headEnd)
    {
        // 每个 AIV 都在自己的本地 flag 空间使用同一组固定编号：
        // localSlot0/1 的 ready=0/1，free=4/5。AIV1 不能写 16/17/20/21。
        constexpr uint16_t kAivToAicPayloadReadyFlagId[2] = {0, 1};
        constexpr uint16_t kAicToAivSlotReusableFlagId[2] = {4, 5};
        for (uint32_t groupBegin = headBegin; groupBegin < headEnd;) {
            uint32_t activeHeads = headEnd - groupBegin;
            if (activeHeads > Shape::kHeadsPerGroup) {
                activeHeads = Shape::kHeadsPerGroup;
            }
            for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                const uint32_t localHead = aiv_ * 2 + localSlot;
                if (localHead >= activeHeads) {
                    continue;
                }
                const uint32_t valueHead = groupBegin + localHead;
                usedLocalSlot_[localSlot] = true;
                // 初始 free 或上一组 C7 free；V0 首个消费者是 MTE2。
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                    kAicToAivSlotReusableFlagId[localSlot]);
                StageV0(chunk, valueHead, localHead, localSlot);
                StageV1(chunk, localHead, localSlot);
                // V1 的 72 KiB score payload 已经写入 workspace。
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    kAivToAicPayloadReadyFlagId[localSlot]);
            }
            for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                const uint32_t localHead = aiv_ * 2 + localSlot;
                if (localHead >= activeHeads) {
                    continue;
                }
                const uint32_t valueHead = groupBegin + localHead;
                // C2 已写回 raw Aqk/Akk，且不再读取 V1 payload。
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    kAicToAivSlotReusableFlagId[localSlot]);
                StageV3(chunk, valueHead, localHead, localSlot);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    kAivToAicPayloadReadyFlagId[localSlot]);
            }
            for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                const uint32_t localHead = aiv_ * 2 + localSlot;
                if (localHead >= activeHeads) {
                    continue;
                }
                const uint32_t valueHead = groupBegin + localHead;
                // C4 已一次性读完 B/X0/negX1/Akk，V6 可以原址换义。
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                    kAicToAivSlotReusableFlagId[localSlot]);
                StageV6(chunk, valueHead, localHead, localSlot);
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    kAivToAicPayloadReadyFlagId[localSlot]);
            }
            groupBegin += activeHeads;
        }
    }

    __aicore__ inline void StageV0(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto q = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQ);
        auto k = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kK);
        auto gate = resource_.ubBuf.template GetBufferByByte<GateT>(
            computeSlot + Arch35Ub::kGateInput);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kG);
        auto beta = resource_.ubBuf.template GetBufferByByte<BetaT>(
            state + Arch35Ub::kBetaRaw);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);
        auto qRstd = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kQRstd);
        auto kRstd = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kKRstd);
        auto gRef = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGRef[0]);
        auto gLast = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGLast);
        auto dtBias = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kV0Work);
        auto aLog = dtBias[Shape::kHeadDim];

        const uint32_t qkHead = QkHeadForValueHead(args_.tiling, valueHead);
        const uint64_t qkOffset = QkInputOffset(args_.tiling, chunk, qkHead);
        const uint64_t gateOffset =
            RawGateInputOffset(args_.tiling, chunk, valueHead);
        const uint64_t betaOffset =
            BetaInputOffset(args_.tiling, chunk, valueHead);
        const uint32_t qkStride = args_.tiling.inputSequenceMajor
                                      ? static_cast<uint32_t>(
                                            static_cast<uint64_t>(
                                                args_.tiling.qkHeadNum - 1) *
                                            Shape::kHeadDim * sizeof(bfloat16_t))
                                      : 0;
        const uint32_t gateStride = args_.tiling.inputSequenceMajor
                                        ? static_cast<uint32_t>(
                                              static_cast<uint64_t>(
                                                  args_.tiling.valueHeadNum - 1) *
                                              Shape::kHeadDim * sizeof(GateT))
                                        : 0;

        AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
        AscendC::DataCopyPad(q, qGm_[qkOffset],
            {static_cast<uint16_t>(chunk.validRows),
             static_cast<uint32_t>(Shape::kHeadDim * sizeof(bfloat16_t)),
             qkStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(k, kGm_[qkOffset],
            {static_cast<uint16_t>(chunk.validRows),
             static_cast<uint32_t>(Shape::kHeadDim * sizeof(bfloat16_t)),
             qkStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::DataCopyPad(gate, gateGm_[gateOffset],
            {static_cast<uint16_t>(chunk.validRows),
             static_cast<uint32_t>(Shape::kHeadDim * sizeof(GateT)),
             gateStride, 0, 0},
            {false, 0, 0, 0});
        if (args_.tiling.inputSequenceMajor) {
            const uint32_t betaStride = static_cast<uint32_t>(
                static_cast<uint64_t>(args_.tiling.valueHeadNum - 1) *
                sizeof(BetaT));
            AscendC::DataCopyPad(beta, betaGm_[betaOffset],
                {static_cast<uint16_t>(chunk.validRows),
                 static_cast<uint32_t>(sizeof(BetaT)), betaStride, 0, 0},
                {false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(beta, betaGm_[betaOffset],
                {1, static_cast<uint32_t>(chunk.validRows * sizeof(BetaT)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
            if (args_.tiling.hasDtBias) {
                AscendC::DataCopy(dtBias,
                    dtBiasGm_[valueHead * Shape::kHeadDim],
                    Shape::kHeadDim);
            }
            if (args_.aLog != nullptr) {
                AscendC::DataCopyExtParams scalarCopy{
                    1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> scalarPad{
                    false, 0, 0, 0};
                AscendC::DataCopyPad(aLog, aLogGm_[valueHead], scalarCopy,
                                     scalarPad);
            }
        }
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        if (args_.tiling.inputSequenceMajor) {
            AscendC::VF_CALL<Detail::StageV0Vf<
                GateT, BetaT, CompilePolicy, true>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
                reinterpret_cast<__ubuf__ GateT *>(gate.GetPhyAddr()),
                reinterpret_cast<__ubuf__ BetaT *>(beta.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(dtBias.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(aLog.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gRef.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gLast.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(kRstd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
                static_cast<uint16_t>(chunk.validRows), args_.tiling.epsilon,
                args_.tiling.lowerBound, args_.tiling.hasDtBias,
                args_.aLog != nullptr);
        } else {
            AscendC::VF_CALL<Detail::StageV0Vf<
                GateT, BetaT, CompilePolicy, false>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr()),
                reinterpret_cast<__ubuf__ GateT *>(gate.GetPhyAddr()),
                reinterpret_cast<__ubuf__ BetaT *>(beta.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(dtBias.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(aLog.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gRef.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gLast.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(qRstd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(kRstd.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
                static_cast<uint16_t>(chunk.validRows), args_.tiling.epsilon,
                args_.tiling.lowerBound, args_.tiling.hasDtBias,
                args_.aLog != nullptr);
        }
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        AscendC::GlobalTensor<bfloat16_t> qContext;
        AscendC::GlobalTensor<bfloat16_t> kContext;
        qContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kQHat));
        kContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kKHat));
        const AscendC::DataCopyExtParams scalarOutputCopy{
            1, static_cast<uint32_t>(chunk.validRows * sizeof(float)),
            0, 0, 0};
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(qContext, q,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kContext, k,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(gkGm_[HeadTensorOffset(
            args_.tiling, chunk, valueHead, Shape::kHeadDim)],
            g, chunk.validRows * Shape::kHeadDim);
        if constexpr (CompilePolicy::outputRecomputeAux) {
            AscendC::DataCopyPad(betaEffGm_[HeadScalarOffset(
                args_.tiling, chunk, valueHead)], betaEff, scalarOutputCopy);
        }
        if (IsQkOutputOwner(args_.tiling, valueHead)) {
            // q/k 归一化保存量按 HK 的 head-major 布局输出。
            // GVA 中只允许 QK 头组的首个 HV 写回，避免多 AIV
            // 对同一 QK head 发生重叠写。
            const uint64_t qkOut = QkHeadTensorOffset(
                args_.tiling, chunk, qkHead, Shape::kHeadDim);
            const uint64_t qkRstdOut = QkHeadScalarOffset(
                args_.tiling, chunk, qkHead);
            if constexpr (CompilePolicy::outputRecomputeAux) {
                AscendC::DataCopy(qHatGm_[qkOut], q,
                    chunk.validRows * Shape::kHeadDim);
            }
            if constexpr (CompilePolicy::outputRecomputeAux) {
                AscendC::DataCopy(kHatGm_[qkOut], k,
                    chunk.validRows * Shape::kHeadDim);
            }
            if constexpr (CompilePolicy::outputRecomputeAux) {
                AscendC::DataCopyPad(
                    qRstdGm_[qkRstdOut], qRstd, scalarOutputCopy);
            }
            if constexpr (CompilePolicy::outputRecomputeAux) {
                AscendC::DataCopyPad(
                    kRstdGm_[qkRstdOut], kRstd, scalarOutputCopy);
            }
        }
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV1(const ChunkRange &chunk,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto qPlus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQ);
        auto kPlus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kK);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kG);
        auto gRef = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGRef[0]);
        auto kMinus = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKMinus);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        // asc_vf_call 返回时 VF 已完成；Mutex 继续表达静态 UB 的 V 到 MTE3 生命周期。
        asc_vf_call<Detail::StageV1Vf<CompilePolicy>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(qPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gRef.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kPlus.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kMinus.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<bfloat16_t> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload));
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(payload, qPlus,
            2 * Shape::kChunkRows * Shape::kHeadDim);
        AscendC::DataCopy(payload[2 * Shape::kChunkRows * Shape::kHeadDim],
            kMinus, 40 * 1024 / sizeof(bfloat16_t));
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV3(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto rawScore = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kRawScore);
        auto aqk = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kAqk);
        auto lkk = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kLkk);
        auto b = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kB);
        auto x0 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kX0);
        auto x1 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kX1);
        auto negX1 = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kNegX1);
        auto akk = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kAkkPack);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);

        uint16_t remainingRows = static_cast<uint16_t>(chunk.validRows);
        const uint16_t band0Rows = remainingRows > Shape::kSubChunkRows
                                       ? Shape::kSubChunkRows
                                       : remainingRows;
        remainingRows -= band0Rows;
        const uint16_t band1Rows = remainingRows > Shape::kSubChunkRows
                                       ? Shape::kSubChunkRows
                                       : remainingRows;
        remainingRows -= band1Rows;
        const uint16_t band2Rows = remainingRows > Shape::kSubChunkRows
                                       ? Shape::kSubChunkRows
                                       : remainingRows;
        remainingRows -= band2Rows;
        const uint16_t band3Rows = remainingRows;

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        // asc_vf_call 返回时 VF 已完成；Mutex 继续表达静态 UB 的 V 到 MTE3 生命周期。
        asc_vf_call<Detail::StageV3Vf>(
            reinterpret_cast<__ubuf__ float *>(rawScore.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(aqk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(lkk.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(b.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(x0.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(x1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(negX1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(akk.GetPhyAddr()),
            band0Rows, band1Rows, band2Rows, band3Rows,
            args_.tiling.scale);
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<float> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload));
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        AscendC::DataCopy(aqkGm_[AOutputOffset(
            args_.tiling, chunk, valueHead)], aqk,
            chunk.validRows * Shape::kChunkRows);
        AscendC::DataCopy(payload[Workspace::kB / sizeof(float)], b, 1024);
        AscendC::DataCopy(payload[Workspace::kX0 / sizeof(float)], x0, 1024);
        AscendC::DataCopy(payload[Workspace::kNegX1 / sizeof(float)],
                          negX1, 1024);
        // C4 固定读取完整 64x64 矩阵，因此补零后的中转矩阵始终写入
        // 工作空间；公开 Akk 只有 T 行，尾 chunk 只能写有效行。
        AscendC::GlobalTensor<bfloat16_t> akkRelay;
        akkRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + WorkspaceSlotBase(
                workgroup_, localHead, Workspace::kArch35WorkgroupStride) +
            Workspace::kPayload + Workspace::kAkk));
        AscendC::DataCopy(akkRelay, akk,
            Shape::kChunkRows * Shape::kChunkRows);
        if constexpr (CompilePolicy::outputAkk) {
            AscendC::DataCopy(akkGm_[AOutputOffset(
                args_.tiling, chunk, valueHead)], akk,
                chunk.validRows * Shape::kChunkRows);
        }
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    __aicore__ inline void StageV6(const ChunkRange &chunk,
                                    uint32_t valueHead,
                                    uint32_t localHead,
                                    uint32_t localSlot)
    {
        const uint8_t mutex = static_cast<uint8_t>(localSlot); // slot0=0，slot1=1
        const uint32_t computeSlot = Arch35Ub::kComputeSlotBase[localSlot];
        const uint32_t state = Arch35Ub::kStateBase[localSlot];
        auto qg = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQg);
        auto kg = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKg);
        auto vBeta = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kVBeta);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(
            computeSlot + Arch35Ub::kGForPost);
        auto kBetaG = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kKBetaG);
        auto qgScaled = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            computeSlot + Arch35Ub::kQgScaled);
        auto betaEff = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kBetaEff);
        auto gLast = resource_.ubBuf.template GetBufferByByte<float>(
            state + Arch35Ub::kGLast);
        const uint64_t slot = WorkspaceSlotBase(
            workgroup_, localHead, Workspace::kArch35WorkgroupStride);
        AscendC::GlobalTensor<bfloat16_t> qContext;
        AscendC::GlobalTensor<bfloat16_t> kContext;
        qContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kQHat));
        kContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kKHat));

        AscendC::Mutex::Lock<PIPE_MTE2>(mutex);
        AscendC::DataCopy(qg, qContext,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kg, kContext,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(g,
            gkGm_[HeadTensorOffset(
                args_.tiling, chunk, valueHead, Shape::kHeadDim)],
            chunk.validRows * Shape::kHeadDim);
        const uint64_t vOffset =
            ValueInputOffset(args_.tiling, chunk, valueHead);
        const uint32_t vStride = args_.tiling.inputSequenceMajor
                                     ? static_cast<uint32_t>(
                                           static_cast<uint64_t>(
                                               args_.tiling.valueHeadNum - 1) *
                                           Shape::kValueDim * sizeof(bfloat16_t))
                                     : 0;
        AscendC::DataCopyPad(vBeta, vGm_[vOffset],
            {static_cast<uint16_t>(chunk.validRows),
             static_cast<uint32_t>(Shape::kValueDim * sizeof(bfloat16_t)),
             vStride, 0, 0},
            {false, 0, 0, 0});
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutex);

        AscendC::Mutex::Lock<PIPE_V>(mutex);
        AscendC::VF_CALL<Detail::StageV6Vf<CompilePolicy>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(qg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(vBeta.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gLast.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(betaEff.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kg.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(qgScaled.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(kBetaG.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(vBeta.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows), args_.tiling.scale);
        AscendC::Mutex::Unlock<PIPE_V>(mutex);

        AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
        AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
        kBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kKBetaG));
        vBetaRelay.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + slot + Workspace::kPayload +
            Workspace::kVBeta));
        const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
        AscendC::Mutex::Lock<PIPE_MTE3>(mutex);
        const uint64_t out = HeadTensorOffset(
            args_.tiling, chunk, valueHead, Shape::kHeadDim);
        if constexpr (CompilePolicy::outputMode == OutputMode::Save) {
            AscendC::DataCopy(qgGm_[out], qg,
                chunk.validRows * Shape::kHeadDim);
        }
        AscendC::DataCopy(qgScaledGm_[out], qgScaled,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kgGm_[out], kg,
            chunk.validRows * Shape::kHeadDim);
        AscendC::DataCopy(kBetaRelay, kBetaG,
            rhsRows * Shape::kHeadDim);
        AscendC::DataCopy(vBetaRelay, vBeta,
            rhsRows * Shape::kValueDim);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutex);
    }

    PrepareKernelArgs args_{};
    uint32_t workgroup_ = 0;
    uint32_t aiv_ = 0;
    bool usedLocalSlot_[2] = {false, false};
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_{};
    AscendC::GlobalTensor<bfloat16_t> qGm_{};
    AscendC::GlobalTensor<bfloat16_t> kGm_{};
    AscendC::GlobalTensor<bfloat16_t> vGm_{};
    AscendC::GlobalTensor<GateT> gateGm_{};
    AscendC::GlobalTensor<BetaT> betaGm_{};
    AscendC::GlobalTensor<float> dtBiasGm_{};
    AscendC::GlobalTensor<float> aLogGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgScaledGm_{};
    AscendC::GlobalTensor<bfloat16_t> kgGm_{};
    AscendC::GlobalTensor<float> gkGm_{};
    AscendC::GlobalTensor<bfloat16_t> aqkGm_{};
    AscendC::GlobalTensor<bfloat16_t> akkGm_{};
    AscendC::GlobalTensor<bfloat16_t> qHatGm_{};
    AscendC::GlobalTensor<bfloat16_t> kHatGm_{};
    AscendC::GlobalTensor<float> qRstdGm_{};
    AscendC::GlobalTensor<float> kRstdGm_{};
    AscendC::GlobalTensor<float> betaEffGm_{};
};

} // namespace KdaPrepare::Arch35

#endif // ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H
