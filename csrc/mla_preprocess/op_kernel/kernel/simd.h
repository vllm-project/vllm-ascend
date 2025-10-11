/*  Adapted from
 *      https://gitee.com/ascend/ascend-transformer-boost.git
 *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INCLUDE_SIMD_H
#define INCLUDE_SIMD_H

#include "hardware.h"
#include "kernel_operator.h"

/////////////////////////////////////////////////////
// vcgadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cgadd_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, const int32_t repeat,
                               const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    AscendC::BlockReduceSum<DType, false>(dst, src, repeat, 0, dstRepStride, srcBlkStride, srcRepStride);
}

/////////////////////////////////////////////////////
// vadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void add_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1, uint8_t repeat, uint8_t dstBlockStride,
                             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    AscendC::Add<DType, false>(dst, src0, src1, (uint64_t)0, repeat,
                               AscendC::BinaryRepeatParams(dstBlockStride, src0BlockStride, src1BlockStride,
                                                           dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vadds
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void adds_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, DType scalarValue,
                              uint8_t repeat, uint8_t dstBlockStride, uint8_t srcBlockStride, uint8_t dstRepeatStride,
                              uint8_t srcRepeatStride)
{
    AscendC::Adds<DType, false>(
        dst, src, scalarValue, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vcadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cadd_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint8_t repeat,
                              uint16_t dstRepeatStride, uint16_t srcBlockStride, uint16_t srcRepeatStride)
{
    AscendC::RepeatReduceSum<DType, false>(dst, src, repeat, 0, 0, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
/////////////////////////////////////////////////////
// vbrcb
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void brcb_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint16_t dstBlockStride,
                              uint16_t dstRepeatStride, uint8_t repeat)
{
    AscendC::Brcb(dst, src, repeat, AscendC::BrcbRepeatParams(dstBlockStride, dstRepeatStride));
}

/////////////////////////////////////////////////////
// vcmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType, AscendC::ReduceOrder OrderType>
__aicore__ inline void cmax_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint8_t repeat,
                              uint16_t dstRepeatStride, uint16_t srcBlockStride, uint16_t srcRepeatStride)
{
#if defined(__DAV_C220_VEC__)
    AscendC::WholeReduceMax<DType, false>(dst, src, (int32_t)0, repeat, dstRepeatStride, srcBlockStride,
                                          srcRepeatStride, OrderType);
#else
    AscendC::WholeReduceMax<DType, false>(dst, src, (int32_t)0, repeat, dstRepeatStride, srcBlockStride,
                                          srcRepeatStride);
#endif
}

/////////////////////////////////////////////////////
// vconv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(AscendC::LocalTensor<DTypeOut> dst, AscendC::LocalTensor<DTypeIn> src, uint8_t repeat,
                              uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst, src, AscendC::RoundMode::CAST_RINT, (uint64_t)0, repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    } else {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst, src, AscendC::RoundMode::CAST_NONE, (uint64_t)0, repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    }
}

/////////////////////////////////////////////////////
// vconv_f322bf16r
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void convr_v(AscendC::LocalTensor<DTypeOut> dst, AscendC::LocalTensor<DTypeIn> src, uint8_t repeat,
                               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                               uint16_t srcRepeatStride)
{
    AscendC::Cast<DTypeOut, DTypeIn, false>(
        dst, src, AscendC::RoundMode::CAST_RINT, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vdiv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void div_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1, uint8_t repeat, uint8_t dstBlockStride,
                             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    AscendC::Div<DType, false>(dst, src0, src1, (uint64_t)0, repeat,
                               AscendC::BinaryRepeatParams(dstBlockStride, src0BlockStride, src1BlockStride,
                                                           dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vexp
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void exp_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint8_t repeat,
                             uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                             uint16_t srcRepeatStride)
{
    AscendC::Exp<DType, false>(
        dst, src, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void max_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1, uint8_t repeat, uint8_t dstBlockStride,
                             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    AscendC::Max<DType, false>(dst, src0, src1, (uint64_t)0, repeat,
                               AscendC::BinaryRepeatParams(dstBlockStride, src0BlockStride, src1BlockStride,
                                                           dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vmul
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mul_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1, uint8_t repeat, uint8_t dstBlockStride,
                             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    AscendC::Mul<DType, false>(dst, src0, src1, (uint64_t)0, repeat,
                               AscendC::BinaryRepeatParams(dstBlockStride, src0BlockStride, src1BlockStride,
                                                           dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vmuls
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void muls_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0, DType src1,
                              uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                              uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    AscendC::Muls<DType, false>(
        dst, src0, src1, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vsub
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void sub_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1, uint8_t repeat, uint8_t dstBlockStride,
                             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    AscendC::Sub<DType, false>(dst, src0, src1, (uint64_t)0, repeat,
                               AscendC::BinaryRepeatParams(dstBlockStride, src0BlockStride, src1BlockStride,
                                                           dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vmaxs
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void maxs_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0, DType src1,
                              uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                              uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    AscendC::Maxs<DType, false>(
        dst, src0, src1, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vmins
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mins_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src0, DType src1,
                              uint8_t repeat, uint16_t dstBlockStride, uint16_t srcBlockStride,
                              uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    AscendC::Mins<DType, false>(
        dst, src0, src1, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vsqrt
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void sqrt_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint8_t repeat,
                              uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
    AscendC::Sqrt<DType, false>(
        dst, src, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vln
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void ln_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, uint8_t repeat,
                            uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                            uint16_t srcRepeatStride)
{
    AscendC::Ln<DType, false>(
        dst, src, (uint64_t)0, repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}

/////////////////////////////////////////////////////
// vtranspose
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void tranpose_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src)
{
    AscendC::Transpose(dst, src);
}

/////////////////////////////////////////////////////
// vcgmax
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void cgmax_v(AscendC::LocalTensor<DType> dst, AscendC::LocalTensor<DType> src, const int32_t repeat,
                               const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    AscendC::BlockReduceMax<DType, false>(dst, src, repeat, 0, dstRepStride, srcBlkStride, srcRepStride);
}
#endif
