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

constexpr uint32_t kVlFp32 = AscendC::VECTOR_REG_WIDTH / static_cast<uint32_t>(sizeof(float));

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
                    MulAddDst(acc0, x0, wReg, mask);
                    MulAddDst(acc1, x1, wReg, mask);
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
                    MulAddDst(acc0, x0, wReg, mask);
                    MulAddDst(acc1, x1, wReg, mask);
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
                    MulAddDst(acc0, x0, wReg, mask);
                    DataCopy(dstAddr + i * kVlFp32, acc0, mask);
                }
            } else {
                RegTensor<D_IN> xIn0;
                for (uint16_t i = 0; i < repeatTimes; ++i) {
                    mask = UpdateMask<float>(sreg);
                    DataCopy<D_IN, LoadDist::DIST_UNPACK_B16>(xIn0, srcAddr + i * kVlFp32);
                    Cast<float, D_IN, kCastB16ToB32>(x0, xIn0, mask);
                    DataCopy<float, LoadDist::DIST_NORM>(acc0, dstAddr + i * kVlFp32);
                    MulAddDst(acc0, x0, wReg, mask);
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

/*! dst[0] = sum(src * src)，src 不被破坏 */
__aicore__ inline void ReduceSquareSum(const LocalTensor<float> &dstScalar, const LocalTensor<float> &src,
                                       uint32_t hiddenSize)
{
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
__aicore__ inline void InvRmsScalar(const LocalTensor<float> &dstScalar, float invHiddenSize, float normEps)
{
    __local_mem__ float *dstAddr = (__local_mem__ float *)dstScalar.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> v, one;
        MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(v, dstAddr);
        Muls(v, v, invHiddenSize, maskAll);
        Adds(v, v, normEps, maskAll);
        Sqrt(v, v, maskAll);
        Duplicate(one, 1.0f, maskAll);
        Div(v, one, v, maskAll);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, v, maskAll);
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
