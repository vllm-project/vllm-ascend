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
 * \file attn_res_fwd_regbase_common.h
 * \brief arch35 RegBase VF helpers（Reload / Resident 共用）
 */
#ifndef ATTN_RES_FWD_REGBASE_COMMON_H
#define ATTN_RES_FWD_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "reduce_common.h"

namespace AttnResFwd {
namespace RegBase {

using namespace AscendC;
using namespace AscendC::MicroAPI;

// Dump/PRINTF 插桩：默认关。需要时改为 1 并重编。
#ifndef ATTN_SOFTMAX_DUMP
#define ATTN_SOFTMAX_DUMP 0
#endif

constexpr AscendC::MicroAPI::CastTrait kCastB16ToB32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait kCastB32ToB16Rint = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr DivSpecificMode kAccurateDiv = {MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr SqrtSpecificMode kAccurateSqrt = {MaskMergeMode::ZEROING, false, SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr ExpSpecificMode kAccurateExp = {MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_FALSE};

constexpr uint32_t kVlFp32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));

// Add in FP32 and round to BF16 before residual score and value computation.
template <typename T>
__aicore__ inline void AddB16(const LocalTensor<T> &dst, const LocalTensor<T> &rhs, uint32_t count)
{
    __local_mem__ T *dstAddr = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *rhsAddr = (__local_mem__ T *)rhs.GetPhyAddr();
    uint32_t remaining = count;
    uint16_t repeats = static_cast<uint16_t>((count + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__ {
        RegTensor<T> lhs16, rhs16, out16;
        RegTensor<float> lhs32, rhs32, out32;
        MaskReg mask;
        for (uint16_t i = 0; i < repeats; ++i) {
            mask = UpdateMask<float>(remaining);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(lhs16, dstAddr + i * kVlFp32);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(rhs16, rhsAddr + i * kVlFp32);
            Cast<float, T, kCastB16ToB32>(lhs32, lhs16, mask);
            Cast<float, T, kCastB16ToB32>(rhs32, rhs16, mask);
            AscendC::MicroAPI::Add(out32, lhs32, rhs32, mask);
            Cast<T, float, kCastB32ToB16Rint>(out16, out32, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + i * kVlFp32, out16, mask);
        }
    }
}

/*!
 * outFp32 += Cast(srcB16) * broadcast(brcOneBlock[0])
 * even H：半分双发；odd H：单路。
 */
template <typename D_IN>
__aicore__ inline void WeightedMulAddFromB16(const LocalTensor<float> &outFp32, const LocalTensor<D_IN> &srcB16,
                                             const LocalTensor<float> &brcOneBlock, uint32_t hiddenSize)
{
    __local_mem__ D_IN *srcAddr = (__local_mem__ D_IN *)srcB16.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)outFp32.GetPhyAddr();
    __local_mem__ float *brcAddr = (__local_mem__ float *)brcOneBlock.GetPhyAddr();

    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ D_IN *srcAddr2 = srcAddr + halfCount;
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;

        __VEC_SCOPE__
        {
            RegTensor<float> wReg, x0, x1, acc0, acc1;
            MaskReg mask;
            DataCopy<float, LoadDist::DIST_BRC_B32>(wReg, brcAddr);
            if constexpr (IsSameType<D_IN, float>::value) {
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    mask = UpdateMask<float>(sreg);
                    DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                    DataCopy<float, LoadDist::DIST_NORM>(x1, srcAddr2 + i * kVlFp32);
                    DataCopy<float, LoadDist::DIST_NORM>(acc0, dstAddr + i * kVlFp32);
                    DataCopy<float, LoadDist::DIST_NORM>(acc1, dstAddr2 + i * kVlFp32);
                    Mul(x0, x0, wReg, mask);
                    Add(acc0, acc0, x0, mask);
                    Mul(x1, x1, wReg, mask);
                    Add(acc1, acc1, x1, mask);
                    DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                    DataCopy(dstAddr2 + i * kVlFp32, acc1, mask);
                }
            } else {
                RegTensor<D_IN> xIn0, xIn1;
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    mask = UpdateMask<float>(sreg);
                    DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn0, srcAddr + i * kVlFp32);
                    DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn1, srcAddr2 + i * kVlFp32);
                    Cast<float, D_IN, kCastB16ToB32>(x0, xIn0, mask);
                    Cast<float, D_IN, kCastB16ToB32>(x1, xIn1, mask);
                    DataCopy<float, LoadDist::DIST_NORM>(acc0, dstAddr + i * kVlFp32);
                    DataCopy<float, LoadDist::DIST_NORM>(acc1, dstAddr2 + i * kVlFp32);
                    Mul(x0, x0, wReg, mask);
                    Add(acc0, acc0, x0, mask);
                    Mul(x1, x1, wReg, mask);
                    Add(acc1, acc1, x1, mask);
                    DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                    DataCopy(dstAddr2 + i * kVlFp32, acc1, mask);
                }
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> wReg, x0, acc0;
            MaskReg mask;
            DataCopy<float, LoadDist::DIST_BRC_B32>(wReg, brcAddr);
            if constexpr (IsSameType<D_IN, float>::value) {
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    mask = UpdateMask<float>(sreg);
                    DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                    DataCopy<float, LoadDist::DIST_NORM>(acc0, dstAddr + i * kVlFp32);
                    Mul(x0, x0, wReg, mask);
                    Add(acc0, acc0, x0, mask);
                    DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                }
            } else {
                RegTensor<D_IN> xIn0;
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    mask = UpdateMask<float>(sreg);
                    DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn0, srcAddr + i * kVlFp32);
                    Cast<float, D_IN, kCastB16ToB32>(x0, xIn0, mask);
                    DataCopy<float, LoadDist::DIST_NORM>(acc0, dstAddr + i * kVlFp32);
                    Mul(x0, x0, wReg, mask);
                    Add(acc0, acc0, x0, mask);
                    DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                }
            }
        }
    }
}

/*! BF16/FP16 → FP32，半分双发写入 dstFp32 */
template <typename D_IN>
__aicore__ inline void CastB16ToFp32Dual(const LocalTensor<float> &dstFp32, const LocalTensor<D_IN> &srcB16,
                                         uint32_t hiddenSize)
{
    __local_mem__ D_IN *srcAddr = (__local_mem__ D_IN *)srcB16.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstFp32.GetPhyAddr();

    if constexpr (IsSameType<D_IN, float>::value) {
        DataCopy(dstFp32, srcB16, hiddenSize);
        return;
    }

    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ D_IN *srcAddr2 = srcAddr + halfCount;
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<D_IN> xIn0, xIn1;
            RegTensor<float> x0, x1;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn0, srcAddr + i * kVlFp32);
                DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn1, srcAddr2 + i * kVlFp32);
                Cast<float, D_IN, kCastB16ToB32>(x0, xIn0, mask);
                Cast<float, D_IN, kCastB16ToB32>(x1, xIn1, mask);
                DataCopy(dstAddr + i * kVlFp32, x0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, x1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<D_IN> xIn0;
            RegTensor<float> x0;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn0, srcAddr + i * kVlFp32);
                Cast<float, D_IN, kCastB16ToB32>(x0, xIn0, mask);
                DataCopy(dstAddr + i * kVlFp32, x0, mask);
            }
        }
    }
}

/*! FP32 → BF16/FP16 CAST_RINT，半分双发 */
template <typename D_OUT>
__aicore__ inline void CastFp32ToB16Dual(const LocalTensor<D_OUT> &dstB16, const LocalTensor<float> &srcFp32,
                                         uint32_t hiddenSize)
{
    __local_mem__ float *srcAddr = (__local_mem__ float *)srcFp32.GetPhyAddr();
    __local_mem__ D_OUT *dstAddr = (__local_mem__ D_OUT *)dstB16.GetPhyAddr();

    if constexpr (IsSameType<D_OUT, float>::value) {
        DataCopy(dstB16, srcFp32, hiddenSize);
        return;
    }

    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ float *srcAddr2 = srcAddr + halfCount;
        __local_mem__ D_OUT *dstAddr2 = dstAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> x0, x1;
            RegTensor<D_OUT> y0, y1;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(x1, srcAddr2 + i * kVlFp32);
                Cast<D_OUT, float, kCastB32ToB16Rint>(y0, x0, mask);
                Cast<D_OUT, float, kCastB32ToB16Rint>(y1, x1, mask);
                DataCopy<D_OUT, StoreDist::DIST_PACK_B32>(dstAddr + i * kVlFp32, y0, mask);
                DataCopy<D_OUT, StoreDist::DIST_PACK_B32>(dstAddr2 + i * kVlFp32, y1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> x0;
            RegTensor<D_OUT> y0;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                Cast<D_OUT, float, kCastB32ToB16Rint>(y0, x0, mask);
                DataCopy<D_OUT, StoreDist::DIST_PACK_B32>(dstAddr + i * kVlFp32, y0, mask);
            }
        }
    }
}

/*! dst = src * src（半分双发） */
__aicore__ inline void MulSquareDual(const LocalTensor<float> &dst, const LocalTensor<float> &src, uint32_t hiddenSize)
{
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ float *srcAddr2 = srcAddr + halfCount;
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> x0, x1, y0, y1;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(x1, srcAddr2 + i * kVlFp32);
                Mul(y0, x0, x0, mask);
                Mul(y1, x1, x1, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, y1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> x0, y0;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                Mul(y0, x0, x0, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
            }
        }
    }
}

/*! dst = src0 * src1（半分双发） */
__aicore__ inline void MulDual(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                               const LocalTensor<float> &src1, uint32_t hiddenSize)
{
    __local_mem__ float *aAddr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *bAddr = (__local_mem__ float *)src1.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ float *aAddr2 = aAddr + halfCount;
        __local_mem__ float *bAddr2 = bAddr + halfCount;
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> a0, a1, b0, b1, y0, y1;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(a0, aAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(a1, aAddr2 + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(b0, bAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(b1, bAddr2 + i * kVlFp32);
                Mul(y0, a0, b0, mask);
                Mul(y1, a1, b1, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, y1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> a0, b0, y0;
            MaskReg mask;
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(a0, aAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(b0, bAddr + i * kVlFp32);
                Mul(y0, a0, b0, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
            }
        }
    }
}

/*! dst = src * broadcast(scalarSrc[0])；scalar 来自已 Brcb 的 1 block 或直接 DIST_BRC */
__aicore__ inline void MulByBrcBlockDual(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                         const LocalTensor<float> &brcOneBlock, uint32_t hiddenSize)
{
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *brcAddr = (__local_mem__ float *)brcOneBlock.GetPhyAddr();
    if ((hiddenSize & 1U) == 0U) {
        const uint32_t halfCount = hiddenSize >> 1;
        uint32_t sreg = halfCount;
        const uint16_t repeatTimes = static_cast<uint16_t>((halfCount + kVlFp32 - 1U) / kVlFp32);
        __local_mem__ float *srcAddr2 = srcAddr + halfCount;
        __local_mem__ float *dstAddr2 = dstAddr + halfCount;
        __VEC_SCOPE__
        {
            RegTensor<float> wReg, x0, x1, y0, y1;
            MaskReg mask;
            DataCopy<float, LoadDist::DIST_BRC_B32>(wReg, brcAddr);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                DataCopy<float, LoadDist::DIST_NORM>(x1, srcAddr2 + i * kVlFp32);
                Mul(y0, x0, wReg, mask);
                Mul(y1, x1, wReg, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
                DataCopy(dstAddr2 + i * kVlFp32, y1, mask);
            }
        }
    } else {
        uint32_t sreg = hiddenSize;
        const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
        __VEC_SCOPE__
        {
            RegTensor<float> wReg, x0, y0;
            MaskReg mask;
            DataCopy<float, LoadDist::DIST_BRC_B32>(wReg, brcAddr);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = UpdateMask<float>(sreg);
                DataCopy<float, LoadDist::DIST_NORM>(x0, srcAddr + i * kVlFp32);
                Mul(y0, x0, wReg, mask);
                DataCopy(dstAddr + i * kVlFp32, y0, mask);
            }
        }
    }
}

/*!
 * dst = src * broadcast(scalarSrc[0])
 * 用 DIST_BRC 直接从 scalar 槽广播，省 Level2 Brcb（scalarSrc 需为 32B 对齐且 [0] 有效）。
 */
__aicore__ inline void BroadcastScalarMulDual(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                              const LocalTensor<float> &scalarSrc, uint32_t hiddenSize)
{
    MulByBrcBlockDual(dst, src, scalarSrc, hiddenSize);
}

// Division modes for the bounded RMS path below. Keep the full IEEE path
// for very small/invalid epsilon; softmax continues to use kAccurateDiv.
constexpr DivSpecificMode kNormalDiv = {MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_0ULP_FTZ_TRUE};
constexpr DivSpecificMode kHardwareDiv = {MaskMergeMode::ZEROING, false, DivAlgo::PRECISION_1ULP_FTZ_TRUE};

// Specialize CANN's residual-neighbor correction for numerator 1. The caller
// guarantees sqrt(mean + eps) >= 1e-10 or +inf, so all nonzero finite quotients and
// residuals are normal FP32 values and exponent scaling is unnecessary.
__simd_callee__ inline void ReciprocalRmsNormal(RegTensor<float> &value, MaskReg &mask)
{
    RegTensor<float> one, negative, quotient, original, previous, next, error, previousError, nextError;
    RegTensor<int32_t> previousBits, nextBits;
    MaskReg choose, finite;
    Duplicate(one, 1.0f, mask);
    Div<float, &kHardwareDiv>(quotient, one, value, mask);
    original = quotient;
    Muls(negative, value, -1.0f, mask);
    Adds(previousBits, (RegTensor<int32_t> &)quotient, -1, mask);
    Adds(nextBits, (RegTensor<int32_t> &)quotient, 1, mask);
    previous = (RegTensor<float> &)previousBits;
    next = (RegTensor<float> &)nextBits;
    error = one; previousError = one; nextError = one;
    MulAddDst(error, quotient, negative, mask);
    MulAddDst(previousError, previous, negative, mask);
    MulAddDst(nextError, next, negative, mask);
    Abs(error, error, mask); Abs(previousError, previousError, mask); Abs(nextError, nextError, mask);
    Compare<float, CMPMODE::LT>(choose, error, previousError, mask);
    Select(error, error, previousError, choose);
    Select(quotient, quotient, previous, choose);
    Compare<float, CMPMODE::LT>(choose, nextError, error, mask);
    Select(quotient, next, quotient, choose);
    // Any finite sqrt of an FP32 value is smaller than 1e30. Preserve the
    // hardware reciprocal for +inf/NaN instead of using their residuals.
    Compares<float, CMPMODE::LT>(finite, value, 1.0e30f, mask);
    Select(value, quotient, original, finite);
}

__simd_callee__ inline void InvRmsFromSum(RegTensor<float> &value, uint32_t hiddenSize,
    float epsilon, MaskReg &mask)
{
    RegTensor<float> divisor;
    Duplicate(divisor, static_cast<float>(hiddenSize), mask);
    if (epsilon >= 1.0e-20f) {
        // A flushed subnormal mean is <2^-126, far below half an ULP of
        // epsilon here. Adding epsilon therefore gives the same FP32 value.
        Div<float, &kNormalDiv>(value, value, divisor, mask);
        Adds(value, value, epsilon, mask);
        Sqrt<float, &kAccurateSqrt>(value, value, mask);
        ReciprocalRmsNormal(value, mask);
    } else {
        Div<float, &kAccurateDiv>(value, value, divisor, mask);
        Adds(value, value, epsilon, mask);
        Sqrt<float, &kAccurateSqrt>(value, value, mask);
        Duplicate(divisor, 1.0f, mask);
        Div<float, &kAccurateDiv>(value, divisor, value, mask);
    }
}

// Fixed 32-lane groups and a four-level carry tree avoid a long serial sum
// of already-reduced 64-element partials. Keep the FP32 operation boundaries.
template <bool SQUARE, bool NORMALIZE = false, bool CALC_INV = false, typename T = float>
__aicore__ inline void GroupedReduce(const LocalTensor<float> &dst,
    const LocalTensor<T> &lhs, const LocalTensor<float> &rhs,
    const LocalTensor<float> &inv, uint32_t hiddenSize, float epsilon,
    const LocalTensor<float> &rowOut)
{
    auto *a = (__local_mem__ T *)lhs.GetPhyAddr();
    auto *rowAddr = (__local_mem__ float *)rowOut.GetPhyAddr();
    auto *b = (__local_mem__ float *)rhs.GetPhyAddr();
    auto *r = (__local_mem__ float *)inv.GetPhyAddr();
    auto *out = (__local_mem__ float *)dst.GetPhyAddr();
    uint16_t repeats = static_cast<uint16_t>(hiddenSize / 64);
    __VEC_SCOPE__ {
        RegTensor<float> x, y, scale, part, a0, a1, a2, a3, total;
        RegTensor<T> input;
        RegTensor<uint32_t> index, lower, upper, constant;
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        Arange((RegTensor<int32_t> &)index, 0);
        Duplicate(constant, 31U, all);
        And(lower, index, constant, all);
        Adds(upper, lower, 32U, all);
        Duplicate(a0, 0.0f, all); Duplicate(a1, 0.0f, all);
        Duplicate(a2, 0.0f, all); Duplicate(a3, 0.0f, all);
        if constexpr (NORMALIZE) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(scale, r);
        }
        // Full 512-element groups keep the same FP32 addition order while
        // removing carry checks from every 64-element iteration.
        if ((hiddenSize % 512U) == 0U && hiddenSize <= 8192U) {
            uint16_t blocks = static_cast<uint16_t>(hiddenSize / 512U);
            for (uint16_t block = 0; block < blocks; ++block) {
                Duplicate(a0, 0.0f, all);
                for (uint16_t j = 0; j < 8; ++j) {
                    uint16_t i = block * 8U + j;
            if constexpr (IsSameType<T, float>::value) {
                DataCopy(x, a + i * 64U);
            } else {
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(input, a + i * 64U);
                Cast<float, T, kCastB16ToB32>(x, input, all);
                DataCopy(rowAddr + i * 64U, x, all);
            }
            if constexpr (SQUARE) {
                Mul(x, x, x, all);
            } else {
                DataCopy(y, b + i * 64U);
                if constexpr (NORMALIZE) { Mul(x, x, scale, all); }
                Mul(x, x, y, all);
            }
            Add(a0, a0, x, all);
            Gather(part, x, upper); Add(a0, a0, part, all);
                }
                Add(a1, a1, a0, all);
            }
            Duplicate(a0, 0.0f, all);
        } else {
        for (uint16_t i = 0; i < repeats; ++i) {
            if constexpr (IsSameType<T, float>::value) {
                DataCopy(x, a + i * 64U);
            } else {
                DataCopy<T, LoadDist::DIST_UNPACK_B16>(input, a + i * 64U);
                Cast<float, T, kCastB16ToB32>(x, input, all);
                DataCopy(rowAddr + i * 64U, x, all);
            }
            if constexpr (SQUARE) {
                Mul(x, x, x, all);
            } else {
                DataCopy(y, b + i * 64U);
                if constexpr (NORMALIZE) { Mul(x, x, scale, all); }
                Mul(x, x, y, all);
            }
            Add(a0, a0, x, all);
            Gather(part, x, upper); Add(a0, a0, part, all);
            uint32_t steps = (i + 1U) * 2U;
            if ((steps & 15U) == 0U) {
                Add(a1, a1, a0, all); Duplicate(a0, 0.0f, all);
                if ((steps & 255U) == 0U) {
                    Add(a2, a2, a1, all); Duplicate(a1, 0.0f, all);
                    if ((steps & 4095U) == 0U) {
                        Add(a3, a3, a2, all); Duplicate(a2, 0.0f, all);
                    }
                }
            }
        }
        }
        Add(a0, a0, a1, all); Add(a0, a0, a2, all); Add(a0, a0, a3, all);
        Duplicate(constant, 7U, all); And(lower, index, constant, all);
        Gather(total, a0, lower);
        for (uint16_t j = 1; j < 4; ++j) {
            Adds(upper, lower, static_cast<uint32_t>(j) * 8U, all);
            Gather(part, a0, upper); Add(total, total, part, all);
        }
        Duplicate(a0, 0.0f, all);
        for (uint16_t j = 0; j < 8; ++j) {
            Duplicate(index, static_cast<uint32_t>(j), all);
            Gather(part, total, index); Add(a0, a0, part, all);
        }
        if constexpr (CALC_INV) {
            InvRmsFromSum(a0, hiddenSize, epsilon, all);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(out, a0, all);
    }
}

/*! dst[0] = sum(src * src)，src 不被破坏 */
__aicore__ inline void ReduceSquareSum(const LocalTensor<float> &dstScalar, const LocalTensor<float> &src,
                                       uint32_t hiddenSize)
{
    if ((hiddenSize % 512U) == 0U && hiddenSize <= 8192U) {
        GroupedReduce<true>(dstScalar, src, src, dstScalar, hiddenSize, 0.0f, dstScalar);
        return;
    }
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstScalar.GetPhyAddr();
    uint32_t sreg = hiddenSize;
    const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> x, prod, part, acc;
        MaskReg mask;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        Duplicate(acc, 0.0f, maskAll);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(x, srcAddr + i * kVlFp32);
            Mul(prod, x, x, mask);
            ReduceSum(part, prod, mask);
            Add(acc, acc, part, maskAll);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskAll);
    }
}

/*! dst[0] = sum(src0 * src1)，src 不被破坏 */
__aicore__ inline void ReduceMulSum(const LocalTensor<float> &dstScalar, const LocalTensor<float> &src0,
                                    const LocalTensor<float> &src1, uint32_t hiddenSize)
{
    if ((hiddenSize % 512U) == 0U && hiddenSize <= 8192U) {
        GroupedReduce<false>(dstScalar, src0, src1, dstScalar, hiddenSize, 0.0f, dstScalar);
        return;
    }
    __local_mem__ float *aAddr = (__local_mem__ float *)src0.GetPhyAddr();
    __local_mem__ float *bAddr = (__local_mem__ float *)src1.GetPhyAddr();
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstScalar.GetPhyAddr();
    uint32_t sreg = hiddenSize;
    const uint16_t repeatTimes = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__
    {
        RegTensor<float> a, b, prod, part, acc;
        MaskReg mask;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        Duplicate(acc, 0.0f, maskAll);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = UpdateMask<float>(sreg);
            DataCopy<float, LoadDist::DIST_NORM>(a, aAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(b, bAddr + i * kVlFp32);
            Mul(prod, a, b, mask);
            ReduceSum(part, prod, mask);
            Add(acc, acc, part, maskAll);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, acc, maskAll);
    }
}

/*! invRms：dst[0] = 1 / sqrt(dst[0] * invH + eps)，与 Level2 InvRmsInPlace 同语义 */
__aicore__ inline void InvRmsScalar(const LocalTensor<float> &dstScalar, uint32_t hiddenSize, float normEps)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstScalar.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> v, one;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(v, dstAddr);
        Duplicate(one, static_cast<float>(hiddenSize), maskAll);
        Div<float, &kAccurateDiv>(v, v, one, maskAll);
        Adds(v, v, normEps, maskAll);
        Sqrt<float, &kAccurateSqrt>(v, v, maskAll);
        Duplicate(one, 1.0f, maskAll);
        Div<float, &kAccurateDiv>(v, one, v, maskAll);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, v, maskAll);
    }
}

// Keep the existing 64-element reduction order and BF16 rounding boundaries,
// but pass row intermediates directly through vector registers.
template <typename T>
__aicore__ inline void CastAndInvRms(const LocalTensor<float> &row, const LocalTensor<float> &inv,
    const LocalTensor<T> &input, uint32_t hiddenSize, float invHiddenSize, float epsilon)
{
    if ((hiddenSize % 512U) == 0U && hiddenSize <= 8192U) {
        GroupedReduce<true, false, true>(inv, input, row, inv, hiddenSize, epsilon, row);
        return;
    }
    __local_mem__ T *inputAddr = (__local_mem__ T *)input.GetPhyAddr();
    __local_mem__ float *rowAddr = (__local_mem__ float *)row.GetPhyAddr();
    __local_mem__ float *invAddr = (__local_mem__ float *)inv.GetPhyAddr();
    uint32_t remaining = hiddenSize;
    const uint16_t repeats = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__ {
        RegTensor<T> in;
        RegTensor<float> x, square, part, acc, one;
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        Duplicate(acc, 0.0f, all);
        for (uint16_t i = 0; i < repeats; ++i) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(in, inputAddr + i * kVlFp32);
            Cast<float, T, kCastB16ToB32>(x, in, mask);
            DataCopy(rowAddr + i * kVlFp32, x, mask);
            Mul(square, x, x, mask);
            ReduceSum(part, square, mask);
            Add(acc, acc, part, all);
        }
        Duplicate(one, static_cast<float>(hiddenSize), all);
        Div<float, &kAccurateDiv>(acc, acc, one, all);
        Adds(acc, acc, epsilon, all);
        Sqrt<float, &kAccurateSqrt>(acc, acc, all);
        Duplicate(one, 1.0f, all);
        Div<float, &kAccurateDiv>(acc, one, acc, all);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(invAddr, acc, all);
    }
}

__aicore__ inline void NormalizeAndReduceScore(const LocalTensor<float> &score,
    const LocalTensor<float> &row, const LocalTensor<float> &weight,
    const LocalTensor<float> &inv, uint32_t hiddenSize)
{
    if ((hiddenSize % 512U) == 0U && hiddenSize <= 8192U) {
        GroupedReduce<false, true>(score, row, weight, inv, hiddenSize, 0.0f, score);
        return;
    }
    __local_mem__ float *rowAddr = (__local_mem__ float *)row.GetPhyAddr();
    __local_mem__ float *weightAddr = (__local_mem__ float *)weight.GetPhyAddr();
    __local_mem__ float *invAddr = (__local_mem__ float *)inv.GetPhyAddr();
    __local_mem__ float *scoreAddr = (__local_mem__ float *)score.GetPhyAddr();
    uint32_t remaining = hiddenSize;
    const uint16_t repeats = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__ {
        RegTensor<float> x, w, invRms, part, acc;
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(invRms, invAddr);
        Duplicate(acc, 0.0f, all);
        for (uint16_t i = 0; i < repeats; ++i) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy<float, LoadDist::DIST_NORM>(x, rowAddr + i * kVlFp32);
            DataCopy<float, LoadDist::DIST_NORM>(w, weightAddr + i * kVlFp32);
            Mul(x, x, invRms, mask);
            Mul(x, x, w, mask);
            ReduceSum(part, x, mask);
            Add(acc, acc, part, all);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scoreAddr, acc, all);
    }
}

template <typename T>
__aicore__ inline void NormalizeAndCast(const LocalTensor<T> &output,
    const LocalTensor<float> &row, const LocalTensor<T> &weight,
    const LocalTensor<float> &inv, uint32_t hiddenSize)
{
    __local_mem__ T *outputAddr = (__local_mem__ T *)output.GetPhyAddr();
    __local_mem__ float *rowAddr = (__local_mem__ float *)row.GetPhyAddr();
    __local_mem__ T *weightAddr = (__local_mem__ T *)weight.GetPhyAddr();
    __local_mem__ float *invAddr = (__local_mem__ float *)inv.GetPhyAddr();
    uint32_t remaining = hiddenSize;
    const uint16_t repeats = static_cast<uint16_t>((hiddenSize + kVlFp32 - 1U) / kVlFp32);
    __VEC_SCOPE__ {
        RegTensor<T> w16, out;
        RegTensor<float> x, w, invRms;
        DataCopy<float, LoadDist::DIST_BRC_B32>(invRms, invAddr);
        for (uint16_t i = 0; i < repeats; ++i) {
            MaskReg mask = UpdateMask<float>(remaining);
            DataCopy<float, LoadDist::DIST_NORM>(x, rowAddr + i * kVlFp32);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(w16, weightAddr + i * kVlFp32);
            Cast<float, T, kCastB16ToB32>(w, w16, mask);
            Mul(x, x, invRms, mask);
            Mul(x, x, w, mask);
            Cast<T, float, kCastB32ToB16Rint>(out, x, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(outputAddr + i * kVlFp32, out, mask);
        }
    }
}

constexpr CastTrait kRoundFp32ToInt = {RegLayout::UNKNOWN, SatMode::NO_SAT,
        MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait kRoundIntToFp32 = {RegLayout::UNKNOWN, SatMode::UNKNOWN,
        MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// Exponential range reduction and polynomial adapted from SLEEF xexpf.
// Copyright Naoki Shibata and contributors 2010-2025.
// Boost Software License - Version 1.0 - August 17th, 2003
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// Preserve explicit FP32 FMA and rounding boundaries across vector backends.
__simd_callee__ inline void SoftmaxExpFp32(RegTensor<float> &x, MaskReg &mask)
{
    RegTensor<float> reduced, qf, poly, coefficient, next, square, fallback;
    RegTensor<int32_t> q, first, second;
    MaskReg usePoly;
    Compares<float, CMPMODE::GE>(usePoly, x, -80.0f, mask);
    Exp<float, &kAccurateExp>(fallback, x, mask);
    Maxs(reduced, x, -80.0f, mask);
    Muls(qf, reduced, 1.4426950408889634074f, mask);
    Cast<int32_t, float, kRoundFp32ToInt>(q, qf, mask);
    Cast<float, int32_t, kRoundIntToFp32>(qf, q, mask);
    Duplicate(coefficient, -0.693145751953125f, mask);
    MulAddDst(reduced, qf, coefficient, mask);
    Duplicate(coefficient, -1.428606765330187045e-6f, mask);
    MulAddDst(reduced, qf, coefficient, mask);
    Duplicate(poly, 0.000198527617612853646278381f, mask);
    Duplicate(next, 0.00139304355252534151077271f, mask);
    MulAddDst(next, poly, reduced, mask); poly = next;
    Duplicate(next, 0.00833336077630519866943359f, mask);
    MulAddDst(next, poly, reduced, mask); poly = next;
    Duplicate(next, 0.0416664853692054748535156f, mask);
    MulAddDst(next, poly, reduced, mask); poly = next;
    Duplicate(next, 0.166666671633720397949219f, mask);
    MulAddDst(next, poly, reduced, mask); poly = next;
    Duplicate(next, 0.5f, mask);
    MulAddDst(next, poly, reduced, mask); poly = next;
    Mul(square, reduced, reduced, mask);
    MulAddDst(reduced, square, poly, mask);
    Adds(poly, reduced, 1.0f, mask);
    ShiftRights(first, q, static_cast<int16_t>(1), mask);
    Sub(second, q, first, mask);
    Adds(first, first, 127, mask); ShiftLefts(first, first, static_cast<int16_t>(23), mask);
    Adds(second, second, 127, mask); ShiftLefts(second, second, static_cast<int16_t>(23), mask);
    Mul(poly, poly, (RegTensor<float> &)first, mask);
    Mul(poly, poly, (RegTensor<float> &)second, mask);
    Select(x, poly, fallback, usePoly);
}

// A decode residual bank fits in a single vector register. Keep its small
// softmax entirely in registers instead of repeatedly staging scalars in UB.
__aicore__ inline void SoftmaxOneRegister(const LocalTensor<float> &scores, uint32_t count)
{
    __local_mem__ float *addr = (__local_mem__ float *)scores.GetPhyAddr();
    uint32_t remaining = count;
    __VEC_SCOPE__ {
        RegTensor<float> x, maximum, total, one;
        MaskReg mask = UpdateMask<float>(remaining);
        MaskReg all = CreateMask<float, MaskPattern::ALL>();
        DataCopy<float, LoadDist::DIST_NORM>(x, addr);
        ReduceMax(maximum, x, mask);
        Duplicate(maximum, maximum, all);
        Sub(x, x, maximum, mask);
        SoftmaxExpFp32(x, mask);
        RegTensor<float> lanes, part;
        RegTensor<uint32_t> indices, laneId, offsets, constant;
        Arange((RegTensor<int32_t> &)indices, 0);
        Duplicate(constant, 7U, all); And(laneId, indices, constant, all);
        if (count < 8U) {
            Duplicate(total, 0.0f, all);
            for (uint16_t j = 0; j < count; ++j) {
                Duplicate(indices, static_cast<uint32_t>(j), all);
                Gather(part, x, indices); Add(total, total, part, all);
            }
        } else {
            Gather(lanes, x, laneId);
            for (uint16_t j = 8; j < count; j += 8) {
                Adds(indices, laneId, static_cast<uint32_t>(j), all);
                Gather(part, x, indices); Add(lanes, lanes, part, all);
            }
            for (uint16_t shift = 4; shift > 0; shift /= 2) {
                Duplicate(constant, static_cast<uint32_t>(shift), all);
                Xor(offsets, laneId, constant, all);
                Gather(part, lanes, offsets); Add(lanes, lanes, part, all);
            }
            total = lanes;
        }
        Duplicate(one, 1.0f, all);
        Div<float, &kAccurateDiv>(total, one, total, all);
        Mul(x, x, total, mask);
        DataCopy(addr, x, mask);
    }
}

/*!
 * UB→UB 精确拷贝 count 个 float。dst/src 基址须 32B 对齐；count∈(0,64]。
 * RegBase UpdateMask；避免 Level2 DataCopy(非整段) 漏写 / 非对齐 CopyMetaScalar→507035。
 */
__aicore__ inline void CopyFloatsUbExact(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                         uint32_t count)
{
    if (count == 0U) {
        return;
    }
    __local_mem__ float *dstAddr = (__local_mem__ float *)dst.GetPhyAddr();
    __local_mem__ float *srcAddr = (__local_mem__ float *)src.GetPhyAddr();
    uint32_t sreg = count;
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        MaskReg mask = UpdateMask<float>(sreg);
        DataCopy<float, LoadDist::DIST_NORM>(x, srcAddr);
        DataCopy<float, StoreDist::DIST_NORM>(dstAddr, x, mask);
    }
    PipeBarrier<PIPE_V>();
}

/*!
 * 小 B Softmax（RegBase，支持 blockCount∈(0,128]）：
 * SoftmaxSmallVec 结构 + B>64 修正：
 * - Max：ReduceMaxHalfInterval
 * - Sum：两段 WholeReduceSum
 * - rem：CopyFloatsUbExact（VF UpdateMask）
 */
__aicore__ inline void SoftmaxSmallRegBase(const LocalTensor<float> &vecMeta, uint32_t blockCount,
                                          uint32_t metaAlign, const LocalTensor<float> &workScalar,
                                          const LocalTensor<float> &brcMeta, const LocalTensor<float> &brcPack)
{
    if (blockCount == 0U) {
        return;
    }
    const LocalTensor<float> brcScratch = brcPack;
    const int32_t curColNum = static_cast<int32_t>(blockCount);
    const uint32_t body = (blockCount / ELEM_PER_BLK_FP32) * ELEM_PER_BLK_FP32;
    const uint32_t remBlk = blockCount - body; // vs VL rem below（勿同名）

    Duplicate(brcMeta, SOFTMAX_PAD, metaAlign);
    PipeBarrier<PIPE_V>();
    if (body > 0U) {
        DataCopy(brcMeta, vecMeta, body);
        PipeBarrier<PIPE_V>();
    }
    CopyFloatsUbExact(brcMeta[body], vecMeta[body], remBlk);
    ReduceMaxHalfInterval(workScalar, brcMeta, curColNum);

    // half-interval 破坏 brcMeta，重新铺 score
    Duplicate(brcMeta, SOFTMAX_PAD, metaAlign);
    PipeBarrier<PIPE_V>();
    if (body > 0U) {
        DataCopy(brcMeta, vecMeta, body);
        PipeBarrier<PIPE_V>();
    }
    CopyFloatsUbExact(brcMeta[body], vecMeta[body], remBlk);

    BrcbScalarRow1(brcScratch, workScalar);
    SubLastDimRow1NoBrc(brcMeta, brcMeta, brcScratch, curColNum);
    Exp(brcMeta, brcMeta, metaAlign);
    PipeBarrier<PIPE_V>();

    if (blockCount > kVlFp32) {
        const uint32_t remVl = blockCount - kVlFp32;
        AscendCUtils::SetMask<float>(kVlFp32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if ASCEND_IS_AIV {
            WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 0, 1, 0);
        }
#else
        WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        // rem SetMask 按 8 对齐；pad 位为 exp(SOFTMAX_PAD)≈0
        AscendCUtils::SetMask<float>(RoundUpFp32(remVl));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if ASCEND_IS_AIV {
            WholeReduceSum<float, false>(brcScratch, brcMeta[kVlFp32], MASK_PLACEHOLDER, 1, 0, 1, 0);
        }
#else
        WholeReduceSum<float, false>(brcScratch, brcMeta[kVlFp32], MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        Add(workScalar, workScalar, brcScratch, 1);
        PipeBarrier<PIPE_V>();
    } else {
        AscendCUtils::SetMask<float>(blockCount);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if ASCEND_IS_AIV {
            WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 0, 1, 0);
        }
#else
        WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    Duplicate(brcScratch, 1.0f, 1);
    PipeBarrier<PIPE_V>();
    Div(workScalar, brcScratch, workScalar, 1);
    PipeBarrier<PIPE_V>();

    BrcbScalarRow1(brcScratch, workScalar);
    MulLastDimRow1NoBrc(brcMeta, brcMeta, brcScratch, curColNum);
    if (body > 0U) {
        DataCopy(vecMeta, brcMeta, body);
        PipeBarrier<PIPE_V>();
    }
    CopyFloatsUbExact(vecMeta[body], brcMeta[body], remBlk);
}


} // namespace RegBase
} // namespace AttnResFwd

#endif // ATTN_RES_FWD_REGBASE_COMMON_H
