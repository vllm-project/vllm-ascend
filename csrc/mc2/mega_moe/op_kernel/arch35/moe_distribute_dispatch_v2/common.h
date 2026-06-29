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
 * \file common.h
 * \brief
 */

#ifndef MC2_MOE_DISPATCH_COMM_H
#define MC2_MOE_DISPATCH_COMM_H

constexpr uint32_t NEED_ONE_HUNDRED_AND_TWENTY_SEVEN = 127;
constexpr uint32_t RIGHT_SHIFT_BIT_SEVEN = 7;
constexpr uint32_t NEED_THIRTY_FIRST = 31;
constexpr uint32_t ALIGN_UP_TO_2_MASK = 1;
constexpr uint32_t ALIGN_UP_TO_32_MASK = 31;
constexpr uint32_t ALIGN_UP_TO_64_MASK = 63;
constexpr uint32_t ALIGN_UP_TO_128_MASK = 127;
constexpr uint32_t ALIGN_UP_TO_256_MASK = 255;
constexpr uint32_t ALIGN_UP_TO_512_MASK = 511;
constexpr uint32_t RIGHT_SHIFT_BIT_FIVE = 5;
constexpr uint32_t FIVE_HUNDRED_AND_ELEVEN = 511;
constexpr uint32_t RIGHT_SHIFT_BIT_NINE = 9;

namespace AscendC {
template <typename T1, typename T2>
__aicore__ inline T2 Ceil(T1 x, T1 y)
{
    return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline T Ceil32(T x)
{
    return (x + NEED_THIRTY_FIRST) >> RIGHT_SHIFT_BIT_FIVE;
}

template <typename T>
__aicore__ inline T Ceil128(T x)
{
    return (x + NEED_ONE_HUNDRED_AND_TWENTY_SEVEN) >> RIGHT_SHIFT_BIT_SEVEN;
}

template <typename T>
__aicore__ inline T Ceil512(T x)
{
    return (x + FIVE_HUNDRED_AND_ELEVEN) >> RIGHT_SHIFT_BIT_NINE;
}

template <typename T1, typename T2>
__aicore__ inline T2 Align(T1 x, T1 y)
{
    return Ceil<T1, T2>(x, y) * y;
}

template <typename T>
__aicore__ inline T Align2(T x)
{
    return (x + ALIGN_UP_TO_2_MASK) & (~ALIGN_UP_TO_2_MASK);
}

template <typename T>
__aicore__ inline T Align32(T x)
{
    return (x + ALIGN_UP_TO_32_MASK) & (~ALIGN_UP_TO_32_MASK);
}

template <typename T>
__aicore__ inline T Align64(T x)
{
    return (x + ALIGN_UP_TO_64_MASK) & (~ALIGN_UP_TO_64_MASK);
}

template <typename T>
__aicore__ inline T Align128(T x)
{
    return (x + ALIGN_UP_TO_128_MASK) & (~ALIGN_UP_TO_128_MASK);
}

template <typename T>
__aicore__ inline T Align256(T x)
{
    return (x + ALIGN_UP_TO_256_MASK) & (~ALIGN_UP_TO_256_MASK);
}

template <typename T>
__aicore__ inline T Align512(T x)
{
    return (x + ALIGN_UP_TO_512_MASK) & (~ALIGN_UP_TO_512_MASK);
}

template <MicroAPI::HistogramsType htype, typename T, typename U>
static __aicore__ inline void HistogramsVf(__local_mem__ U* dst, __local_mem__ T* src, uint16_t repeatElm,
                                           uint16_t halfRepeat, uint32_t totalElm, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::RegTensor<U> dst0Reg;
    AscendC::MicroAPI::RegTensor<U> dst1Reg;
    AscendC::MicroAPI::MaskReg pregOut = AscendC::MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(dst0Reg, 0);
    MicroAPI::Duplicate(dst1Reg, 0);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(totalElm);
        MicroAPI::DataCopy(srcReg, src + repeatElm * i);
        MicroAPI::Histograms<T, U, MicroAPI::HistogramsBinType::BIN0, htype>(dst0Reg, srcReg, preg);
        MicroAPI::Histograms<T, U, MicroAPI::HistogramsBinType::BIN1, htype>(dst1Reg, srcReg, preg);
    }
    MicroAPI::DataCopy(dst, dst0Reg, pregOut);
    MicroAPI::DataCopy(dst + halfRepeat, dst1Reg, pregOut);
}

__aicore__ inline void GetExpertFreq(LocalTensor<uint16_t>& dstLocal, LocalTensor<uint8_t>& srcLocal, uint32_t totalElm)
{
    uint32_t repeatElm = GetVecLen();
    uint16_t repeatTimes = Ceil<uint32_t, uint16_t>(totalElm, repeatElm);
    __local_mem__ uint8_t* src = (__local_mem__ uint8_t*)srcLocal.GetPhyAddr();
    __local_mem__ uint16_t* dst = (__local_mem__ uint16_t*)dstLocal.GetPhyAddr();
    VF_CALL<HistogramsVf<MicroAPI::HistogramsType::FREQUENCY, uint8_t, uint16_t>>(dst, src, repeatElm, repeatElm >> 1,
                                                                                  totalElm, repeatTimes);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetExpertCumSum(LocalTensor<uint16_t>& dstLocal, LocalTensor<uint8_t>& srcLocal,
                                       uint32_t totalElm)
{
    uint32_t repeatElm = GetVecLen();
    uint16_t repeatTimes = Ceil<uint32_t, uint16_t>(totalElm, repeatElm);
    __local_mem__ uint8_t *src = (__local_mem__ uint8_t *)srcLocal.GetPhyAddr();
    __local_mem__ uint16_t *dst = (__local_mem__ uint16_t *)dstLocal.GetPhyAddr();
    VF_CALL<HistogramsVf<MicroAPI::HistogramsType::ACCUMULATE, uint8_t, uint16_t>>(dst, src, repeatElm, repeatElm >> 1,
                                                                                   totalElm, repeatTimes);
    PipeBarrier<PIPE_V>();
}

static __aicore__ inline void ReduceLoop(__local_mem__ int32_t* dst, __local_mem__ int32_t* src, uint16_t repeat0Times,
                                         uint32_t repeat0SrcStride, uint16_t repeat1Times, uint32_t repeat1Stride,
                                         uint32_t repeat1Element, uint32_t repeat0DstStride)
{
    MicroAPI::MaskReg maskFirst = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::VL1>();
    for (uint16_t i0 = 0; i0 < repeat0Times; ++i0) {
        MicroAPI::RegTensor<int32_t> sumReg;
        uint32_t elements = repeat1Element;
        MicroAPI::Duplicate(sumReg, 0);
        for (uint16_t i1 = 0; i1 < repeat1Times; ++i1) {
            MicroAPI::RegTensor<int32_t> srcReg;
            MicroAPI::RegTensor<int32_t> dstReg;
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<int32_t>(elements);
            MicroAPI::DataCopy(srcReg, src + i0 * repeat0SrcStride + i1 * repeat1Stride);
            MicroAPI::ReduceSum(dstReg, srcReg, mask);
            MicroAPI::Add(sumReg, sumReg, dstReg, maskFirst);
        }
        MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + i0 * repeat0DstStride, sumReg,
                                                                                 maskFirst);
    }
}

__aicore__ inline void GetReduceSum(LocalTensor<int32_t>& dstLocal, LocalTensor<int32_t>& srcLocal,
                                    uint16_t repeat0Times, uint32_t repeat0SrcStride, uint32_t repeat1Element,
                                    uint32_t repeat0DstStride)
{
    if (repeat0Times == 0 || repeat1Element == 0) {
        return;
    }

    uint32_t repeat1Stride = GetVecLen() / sizeof(int32_t);
    uint16_t repeat1Times = Ceil<uint32_t, uint16_t>(repeat1Element, repeat1Stride);
    __local_mem__ int32_t* src = (__local_mem__ int32_t*)srcLocal.GetPhyAddr();
    __local_mem__ int32_t* dst = (__local_mem__ int32_t*)dstLocal.GetPhyAddr();
    VF_CALL<ReduceLoop>(dst, src, repeat0Times, repeat0SrcStride, repeat1Times, repeat1Stride, repeat1Element,
                        repeat0DstStride);
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif
